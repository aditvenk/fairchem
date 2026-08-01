"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Interleaved A/B of the PBC scaffold cache, end to end.

Separate benchmark processes drift by more than 1% run to run, which is larger
than the effect being measured. This builds one predictor and alternates timing
blocks with the cache on and off, so drift hits both arms equally, and reports
a paired per-round difference against its own spread.
"""

from __future__ import annotations

import argparse
import statistics

import numpy as np
import torch

import fairchem.core.graph.radius_graph_pbc as R
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.common_structures import get_fcc_crystal_by_num_atoms
from fairchem.core.units.mlip_unit import MLIPPredictUnit
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

CKPT = "/home/avenkataraman/github/kUPS/examples/uma-s-1p2.pt"


def block(predictor, data, iters: int) -> float:
    """
    Wall-clock ms/step over one timing block, GPU-synchronised at both ends.
    """
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
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--graph-version", type=int, default=2)
    ap.add_argument("--checkpoint", default=CKPT)
    args = ap.parse_args()

    torch.manual_seed(1)
    np.random.seed(1)
    atoms = get_fcc_crystal_by_num_atoms(args.natoms)
    data = AtomicData.from_ase(atoms, task_name="omat")

    predictor = MLIPPredictUnit(
        args.checkpoint,
        "cuda",
        inference_settings=InferenceSettings(
            tf32=True,
            activation_checkpointing=False,
            merge_mole=True,
            compile=True,
            external_graph_gen=False,
            internal_graph_gen_version=args.graph_version,
            execution_mode="umas_fast_gpu",
        ),
    )
    for _ in range(15):
        predictor.predict(data)
    torch.cuda.synchronize()

    series: dict[bool, list[float]] = {False: [], True: []}
    for r in range(args.rounds):
        # Alternate order each round so ordering bias cancels.
        for on in ((False, True) if r % 2 == 0 else (True, False)):
            R._SCAFFOLD_CACHE_ENABLED = on
            R._SCAFFOLD_CACHE.clear()
            predictor.predict(data)  # repopulate before timing
            series[on].append(block(predictor, data, args.iters))
    R._SCAFFOLD_CACHE_ENABLED = True

    print(f"\nnatoms={args.natoms}  graph v{args.graph_version}  "
          f"iters/block={args.iters}  rounds={args.rounds}\n")
    for on in (False, True):
        s = series[on]
        print(
            f"  cache {'on ' if on else 'off'}: median {statistics.median(s):.3f} ms  "
            f"sd {statistics.pstdev(s):.3f}  min {min(s):.3f}  "
            f"({1000 / statistics.median(s):.2f} qps)"
        )

    diffs = [off - on for off, on in zip(series[False], series[True])]
    med = statistics.median(diffs)
    sd = statistics.pstdev(diffs)
    base = statistics.median(series[False])
    print(
        f"\n  paired off-on: median {med:+.3f} ms  sd {sd:.3f}  "
        f"({sum(d > 0 for d in diffs)}/{len(diffs)} rounds favour cache)"
    )
    print(f"  -> {100 * med / base:+.2f}% qps")
    if abs(med) < sd:
        print("  NOTE: |median| < sd, inside the noise floor")


if __name__ == "__main__":
    main()
