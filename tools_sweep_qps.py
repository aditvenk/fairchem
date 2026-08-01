"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Sweep system size and measure inference QPS.

Two things are reported per size. First, absolute ms/step and QPS, written to
JSON so the same harness can be run against a stashed tree and diffed -- that
captures every change, including the ones with no runtime switch (the sync
removals in radius_graph_pbc.py). Second, an in-process paired A/B of the two
optimisations that do have switches, the PBC scaffold cache and the Helion
rotate+scatter fusion, which is far less noisy than comparing processes:
separate benchmark runs on this machine drift by more than 1%.

Graph generation is a largely size-independent CPU cost while the model's work
grows with the system, so the relative win is expected to shrink as atoms grow.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics

import numpy as np
import torch

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.common_structures import get_fcc_crystal_by_num_atoms
from fairchem.core.units.mlip_unit import MLIPPredictUnit
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

CKPT = "/home/avenkataraman/github/kUPS/examples/uma-s-1p2.pt"


def get_toggles():
    """
    Return {name: setter} for optimisations switchable at runtime.

    Absent on an older tree, which is exactly when this harness is used to
    produce the baseline half of the comparison.
    """
    toggles = {}
    try:
        import fairchem.core.graph.radius_graph_pbc as R

        def set_cache(on, _R=R):
            _R._SCAFFOLD_CACHE_ENABLED = on
            _R._SCAFFOLD_CACHE.clear()

        if hasattr(R, "_SCAFFOLD_CACHE_ENABLED"):
            toggles["scaffold_cache"] = set_cache
    except (ImportError, AttributeError):
        pass
    try:
        import fairchem.core.models.uma.triton.helion_fused as HF

        def set_helion(on, _HF=HF):
            _HF.HELION_AVAILABLE = on

        toggles["helion_fusion"] = set_helion
    except ImportError:
        pass
    return toggles


def build(checkpoint: str, graph_version: int):
    return MLIPPredictUnit(
        checkpoint,
        "cuda",
        inference_settings=InferenceSettings(
            tf32=True,
            activation_checkpointing=False,
            merge_mole=True,
            compile=True,
            external_graph_gen=False,
            internal_graph_gen_version=graph_version,
            execution_mode="umas_fast_gpu",
        ),
    )


def block(predictor, data, iters: int) -> float:
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        predictor.predict(data)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--natoms", type=int, nargs="+", default=[100, 250, 500, 1000, 2000, 4000]
    )
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--graph-version", type=int, default=2)
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--out", default="/tmp/sweep_qps.json")
    ap.add_argument("--label", default="current")
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Toggle only these optimisations; any others are forced off, so "
        "the measured gain is attributable to the named ones alone.",
    )
    args = ap.parse_args()

    torch.manual_seed(1)
    np.random.seed(1)
    logging.getLogger().setLevel(logging.ERROR)
    toggles = get_toggles()
    forced_off = {}
    if args.only is not None:
        forced_off = {k: v for k, v in toggles.items() if k not in args.only}
        toggles = {k: v for k, v in toggles.items() if k in args.only}
        for setter in forced_off.values():
            setter(False)
    print(
        f"\nlabel={args.label}  switchable: {sorted(toggles) or 'none'}"
        f"  forced off: {sorted(forced_off) or 'none'}"
    )
    print(f"iters/block={args.iters} rounds={args.rounds}\n")

    header = f"{'natoms':>7} {'edges':>8} {'ms/step':>9} {'qps':>8}"
    if toggles:
        header += f"  {'all-off ms':>11} {'gain':>7}"
    print(header)
    print("-" * (len(header) + 4))

    results = {}
    for n in args.natoms:
        atoms = get_fcc_crystal_by_num_atoms(n)
        data = AtomicData.from_ase(atoms, task_name="omat")
        try:
            p = build(args.checkpoint, args.graph_version)
            for _ in range(12):
                out = p.predict(data)
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            print(f"{n:>7}  OOM")
            torch.cuda.empty_cache()
            continue

        nedges = int(out.get("energy").numel() * 0) or _edge_count(p, data)

        on, off = [], []
        for r in range(args.rounds):
            for state in (True, False) if r % 2 == 0 else (False, True):
                for setter in toggles.values():
                    setter(state)
                for setter in forced_off.values():
                    setter(False)
                p.predict(data)  # repopulate caches before timing
                (on if state else off).append(block(p, data, args.iters))
        for setter in toggles.values():
            setter(True)
        for setter in forced_off.values():
            setter(False)

        ms_on = statistics.median(on)
        row = {
            "natoms": n,
            "edges": nedges,
            "ms": ms_on,
            "qps": 1000 / ms_on,
            "sd": statistics.pstdev(on),
        }
        line = f"{n:>7} {nedges:>8} {ms_on:>9.3f} {1000 / ms_on:>8.2f}"
        if toggles:
            ms_off = statistics.median(off)
            diffs = [o - i for o, i in zip(off, on)]
            row["ms_all_off"] = ms_off
            row["gain_pct"] = 100 * (ms_off / ms_on - 1)
            row["rounds_favouring"] = sum(d > 0 for d in diffs)
            row["paired_sd"] = statistics.pstdev(diffs)
            line += f"  {ms_off:>11.3f} {row['gain_pct']:>6.2f}%"
        print(line)
        results[n] = row

        del p
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump({"label": args.label, "results": results}, f, indent=1)
    print(f"\nwrote {args.out}")


def _edge_count(predictor, data) -> int:
    """
    Edge count for the system, via one direct graph build.
    """
    from fairchem.core.graph.compute import generate_graph

    bb = predictor.model.module.backbone
    d = data.clone().to("cuda") if hasattr(data, "clone") else data
    out = generate_graph(
        d,
        cutoff=bb.cutoff,
        max_neighbors=bb.max_neighbors,
        enforce_max_neighbors_strictly=False,
        radius_pbc_version=2,
        pbc=d.pbc,
    )
    return int(out["edge_index"].shape[1])


if __name__ == "__main__":
    main()
