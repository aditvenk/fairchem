"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Split graph generation into its two halves and test removing its syncs.

generate_graph is 78% GPU-idle on the v3 backend: the GPU work is 0.58 ms but
the call takes 2.63 ms. It is made of radius_graph_pbc_* (the neighbor search)
followed by get_pbc_distances (pure tensor math). This times them separately
and measures three candidate fixes on the second half:

  - dropping the zero-distance filter, whose boolean mask has a data-dependent
    output shape and so forces a device-to-host sync plus four gathers over
    every edge (compute.py:96-104). The radius search already excludes
    zero-distance pairs, so the filter is defensive.
  - torch.compile on get_pbc_distances, which generate_graph currently forbids
    wholesale via @torch.compiler.disable().
  - both together.
"""

from __future__ import annotations

import argparse

import torch
from triton.testing import do_bench

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.common_structures import get_fcc_crystal_by_num_atoms
from fairchem.core.graph.compute import get_pbc_distances
from fairchem.core.graph.radius_graph_pbc import radius_graph_pbc_v2
from fairchem.core.graph.radius_graph_pbc_nvidia import radius_graph_pbc_nvidia
from fairchem.core.units.mlip_unit import MLIPPredictUnit
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

CKPT = "/home/avenkataraman/github/kUPS/examples/uma-s-1p2.pt"


def pbc_distances_nofilter(pos, edge_index, cell, cell_offsets, neighbors):
    """
    get_pbc_distances without the zero-distance filter.

    Keeps every edge, so no boolean mask, no data-dependent shape, no sync and
    no gathers. Returns the same keys as the filtered version.
    """
    row, col = edge_index
    distance_vectors = pos[row] - pos[col]
    neighbors = neighbors.to(cell.device)
    cell = torch.repeat_interleave(cell, neighbors, dim=0)
    offsets = cell_offsets.to(dtype=cell.dtype).view(-1, 1, 3).bmm(cell).view(-1, 3)
    distance_vectors = distance_vectors + offsets
    return {
        "edge_index": edge_index,
        "distances": distance_vectors.norm(dim=-1),
        "distance_vec": distance_vectors,
        "offsets": offsets,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--natoms", type=int, default=1000)
    ap.add_argument("--checkpoint", default=CKPT)
    args = ap.parse_args()

    settings = InferenceSettings(
        tf32=True, activation_checkpointing=False, merge_mole=True, compile=False,
        external_graph_gen=False, internal_graph_gen_version=2,
        execution_mode="umas_fast_gpu",
    )
    p = MLIPPredictUnit(args.checkpoint, "cuda", inference_settings=settings)
    bb = p.model.module.backbone
    cutoff, max_neigh = bb.cutoff, bb.max_neighbors
    del p
    torch.cuda.empty_cache()

    atoms = get_fcc_crystal_by_num_atoms(args.natoms)
    data = AtomicData.from_ase(atoms, task_name="omat").to("cuda")
    print(f"natoms={args.natoms} cutoff={cutoff} max_neighbors={max_neigh}\n")

    backends = {2: radius_graph_pbc_v2, 3: radius_graph_pbc_nvidia}
    for v, fn in backends.items():
        ms = do_bench(
            lambda f=fn: f(data, cutoff, max_neigh, False, data.pbc),
            warmup=25, rep=100,
        )
        print(f"  radius_graph v{v:<2}                     {ms:>7.3f} ms")

    edge_index, cell_offsets, neighbors = radius_graph_pbc_nvidia(
        data, cutoff, max_neigh, False, data.pbc
    )
    print(f"\n  (v3 produced {edge_index.shape[1]} edges)\n")

    def base():
        return get_pbc_distances(
            data.pos, edge_index, data.cell, cell_offsets, neighbors,
            return_offsets=True, return_distance_vec=True,
        )

    def nofilter():
        return pbc_distances_nofilter(
            data.pos, edge_index, data.cell, cell_offsets, neighbors
        )

    ref = base()
    alt = nofilter()
    same_n = ref["edge_index"].shape[1] == alt["edge_index"].shape[1]
    print(f"  filter drops {edge_index.shape[1] - ref['edge_index'].shape[1]} edges; "
          f"no-filter keeps all -> {'equivalent here' if same_n else 'DIFFERS'}")
    if same_n:
        for k in ("distances", "distance_vec", "offsets"):
            d = (ref[k] - alt[k]).abs().max().item()
            assert d == 0.0, f"{k} differs by {d}"
        print("  no-filter output bit-identical to filtered\n")

    compiled = torch.compile(get_pbc_distances, dynamic=True)
    compiled_nf = torch.compile(pbc_distances_nofilter, dynamic=True)

    def base_c():
        return compiled(
            data.pos, edge_index, data.cell, cell_offsets, neighbors,
            return_offsets=True, return_distance_vec=True,
        )

    def nofilter_c():
        return compiled_nf(
            data.pos, edge_index, data.cell, cell_offsets, neighbors
        )

    variants = [
        ("get_pbc_distances (current)", base),
        ("  + no zero-distance filter", nofilter),
        ("  + torch.compile", base_c),
        ("  + both", nofilter_c),
    ]
    results = {}
    for name, f in variants:
        try:
            f()
            torch.cuda.synchronize()
            results[name] = do_bench(f, warmup=25, rep=100)
        except Exception as exc:  # noqa: BLE001 - compile may reject this
            print(f"  {name:<34} FAILED: {type(exc).__name__}: {str(exc)[:90]}")
            continue
    b = results.get("get_pbc_distances (current)")
    for name, ms in results.items():
        tag = "" if b is None or ms == b else f"   {b / ms:.2f}x"
        print(f"  {name:<34} {ms:>7.3f} ms{tag}")


if __name__ == "__main__":
    main()
