"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Compare internal graph-generation backends for correctness and speed.

Graph generation is 5.9 ms of the 41.3 ms inference step at natoms=1000, the
largest single addressable item on that path. Three implementations exist
behind InferenceSettings.internal_graph_gen_version: 1 (radius_graph_pbc),
2 (radius_graph_pbc_v2, the default) and 3 (radius_graph_pbc_nvidia, a
cell-list search from nvalchemiops).

Switching backends changes the neighbor list, not just its speed, so this
checks the produced graph and the resulting energy/forces/stress against the
current default before reporting timings.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.common_structures import get_fcc_crystal_by_num_atoms
from fairchem.core.units.mlip_unit import MLIPPredictUnit
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

CKPT = "/home/avenkataraman/github/kUPS/examples/uma-s-1p2.pt"
BASELINE_VERSION = 2


def build(checkpoint: str, version: int, compile_model: bool):
    settings = InferenceSettings(
        tf32=True,
        activation_checkpointing=False,
        merge_mole=True,
        compile=compile_model,
        external_graph_gen=False,
        internal_graph_gen_version=version,
        execution_mode="umas_fast_gpu",
    )
    return MLIPPredictUnit(checkpoint, "cuda", inference_settings=settings)


def canonical_edges(edge_index: torch.Tensor, offsets: torch.Tensor) -> set:
    """
    Hashable (src, dst, cell offset) set, so edge ordering does not matter.
    """
    ei = edge_index.cpu().numpy()
    off = offsets.cpu().numpy().round().astype(int)
    return {
        (int(ei[0, i]), int(ei[1, i]), *map(int, off[i])) for i in range(ei.shape[1])
    }


def graph_for(checkpoint: str, version: int, atoms) -> tuple[dict, int]:
    """
    Run generate_graph directly, bypassing the model, to compare edge sets.
    """
    from fairchem.core.graph.compute import generate_graph

    predictor = build(checkpoint, version, compile_model=False)
    backbone = predictor.model.module.backbone
    data = AtomicData.from_ase(atoms, task_name="omat").to("cuda")
    out = generate_graph(
        data,
        cutoff=backbone.cutoff,
        max_neighbors=backbone.max_neighbors,
        enforce_max_neighbors_strictly=False,
        radius_pbc_version=version,
        pbc=data.pbc,
    )
    del predictor
    torch.cuda.empty_cache()
    return out, out["edge_index"].shape[1]


def timed(predictor, data, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        predictor.predict(data)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        predictor.predict(data)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--natoms", type=int, default=1000)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--versions", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--checkpoint", default=CKPT)
    args = ap.parse_args()

    torch.manual_seed(1)
    np.random.seed(1)
    atoms = get_fcc_crystal_by_num_atoms(args.natoms)

    print(f"\nnatoms={args.natoms}\n\ngraph comparison (generate_graph directly):")
    ref_edges = None
    for v in args.versions:
        out, nedges = graph_for(args.checkpoint, v, atoms)
        edges = canonical_edges(out["edge_index"], out["cell_offsets"])
        if ref_edges is None:
            ref_edges = edges
            print(f"  v{v} (reference): {nedges} edges, {len(edges)} unique")
        else:
            only_ref = len(ref_edges - edges)
            only_new = len(edges - ref_edges)
            same = "IDENTICAL" if not only_ref and not only_new else "DIFFERS"
            print(
                f"  v{v}: {nedges} edges, {len(edges)} unique  "
                f"missing {only_ref}, extra {only_new}  -> {same}"
            )

    print("\nend-to-end (compiled):")
    results = {}
    for v in args.versions:
        data = AtomicData.from_ase(atoms, task_name="omat")
        predictor = build(args.checkpoint, v, compile_model=True)
        outs = [
            {k: t.detach().clone() for k, t in predictor.predict(data).items()}
            for _ in range(3)
        ]
        ms = timed(predictor, data, args.iters, args.warmup)
        results[v] = {"outs": outs, "ms": ms}
        del predictor
        torch.cuda.empty_cache()

    base = results[BASELINE_VERSION]["outs"]
    for key in ("energy", "forces", "stress"):
        scale = base[0][key].abs().max().item()
        spread = max((base[0][key] - o[key]).abs().max().item() for o in base[1:])
        row = [f"  {key:<8} scale {scale:>11.4e}  v{BASELINE_VERSION} spread {spread:>9.3e}"]
        for v in args.versions:
            if v == BASELINE_VERSION:
                continue
            d = (base[0][key] - results[v]["outs"][0][key]).abs().max().item()
            row.append(f"  v{v} delta {d:>9.3e} ({d / max(scale, 1e-30):.2e} rel)")
        print("".join(row))

    print()
    ms_base = results[BASELINE_VERSION]["ms"]
    for v in args.versions:
        ms = results[v]["ms"]
        tag = " (baseline)" if v == BASELINE_VERSION else f"  {100 * (ms_base / ms - 1):+.2f}% qps"
        print(f"  v{v}: {ms:>7.3f} ms/step  ({1000 / ms:.2f} qps){tag}")


if __name__ == "__main__":
    main()
