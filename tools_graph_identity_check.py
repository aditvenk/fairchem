"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Do graph backends 2 and 3 produce the same neighbor list, on more than one cell?

Backend 3 (nvalchemiops cell list) is ~2x faster than the default backend 2 on
the neighbor search. Promoting it to the default is only safe if it returns the
same graph, otherwise results would silently depend on whether an optional
dependency happens to be installed. The speed benchmark only exercises one
1000-atom FCC crystal, so this sweeps molecules, bulk, slabs, non-periodic and
mixed-PBC cases and compares edge sets exactly.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk, fcc111, molecule

from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.datasets.common_structures import get_fcc_crystal_by_num_atoms
from fairchem.core.graph.compute import generate_graph

CUTOFF = 6.0
MAX_NEIGH = 300


def cases() -> list[tuple[str, Atoms]]:
    rng = np.random.default_rng(0)
    water = molecule("H2O")
    water.center(vacuum=6.0)
    water.pbc = False

    ch4 = molecule("CH4")
    ch4.center(vacuum=8.0)
    ch4.pbc = False

    cu = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((3, 3, 3))
    small = bulk("Cu", "fcc", a=3.6)  # tiny cell -> many periodic images
    pt = fcc111("Pt", size=(3, 3, 4), vacuum=10.0)

    rattled = get_fcc_crystal_by_num_atoms(500)
    rattled.positions += rng.normal(0, 0.15, rattled.positions.shape)

    tri = bulk("Ti", "hcp", a=2.95, c=4.68).repeat((3, 3, 2))

    return [
        ("H2O (no pbc)", water),
        ("CH4 (no pbc)", ch4),
        ("Cu bulk 3x3x3", cu),
        ("Cu primitive (tiny cell)", small),
        ("Pt(111) slab", pt),
        ("FCC 500 rattled", rattled),
        ("Ti hcp", tri),
        ("FCC 1000", get_fcc_crystal_by_num_atoms(1000)),
        ("FCC 2000", get_fcc_crystal_by_num_atoms(2000)),
    ]


def edge_set(out: dict) -> set:
    ei = out["edge_index"].cpu().numpy()
    off = out["cell_offsets"].cpu().numpy().round().astype(int)
    n = ei.shape[1]
    return {(int(ei[0, i]), int(ei[1, i]), *map(int, off[i])) for i in range(n)}


def run(data, version: int) -> dict:
    return generate_graph(
        data,
        cutoff=CUTOFF,
        max_neighbors=MAX_NEIGH,
        enforce_max_neighbors_strictly=False,
        radius_pbc_version=version,
        pbc=data.pbc,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", type=int, nargs="+", default=[2, 3])
    args = ap.parse_args()
    va, vb = args.versions

    print(f"comparing radius_pbc_version {va} (reference) vs {vb}")
    print(f"cutoff={CUTOFF} max_neighbors={MAX_NEIGH}\n")
    print(f"{'case':<28} {'v' + str(va) + ' edges':>10} {'v' + str(vb) + ' edges':>10} "
          f"{'missing':>8} {'extra':>7}  verdict")
    print("-" * 78)

    allsame = True
    items = cases()
    for name, atoms in items:
        data = AtomicData.from_ase(atoms, task_name="omat").to("cuda")
        a, b = edge_set(run(data, va)), edge_set(run(data, vb))
        miss, extra = len(a - b), len(b - a)
        ok = miss == 0 and extra == 0
        allsame &= ok
        print(f"{name:<28} {len(a):>10} {len(b):>10} {miss:>8} {extra:>7}  "
              f"{'IDENTICAL' if ok else 'DIFFERS'}")

    # A mixed batch: periodic and non-periodic systems together, which is the
    # case radius_graph_pbc_v2 exists to handle.
    batch = atomicdata_list_to_batch(
        [AtomicData.from_ase(a, task_name="omat") for _, a in items[:5]]
    ).to("cuda")
    a, b = edge_set(run(batch, va)), edge_set(run(batch, vb))
    miss, extra = len(a - b), len(b - a)
    ok = miss == 0 and extra == 0
    allsame &= ok
    print(f"{'mixed-PBC batch of 5':<28} {len(a):>10} {len(b):>10} {miss:>8} "
          f"{extra:>7}  {'IDENTICAL' if ok else 'DIFFERS'}")

    print("\n" + ("ALL IDENTICAL" if allsame else "SOME DIFFER -- v3 is not a "
                                                  "drop-in default"))


if __name__ == "__main__":
    main()
