"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Attribute radius_graph_pbc_v2's CPU cost to source lines.

v2 spends ~3.6 ms per call at 1000 atoms with only ~1.1 ms of GPU kernel time;
the rest is 412 kernel launches and 99 device-to-host syncs. Optimising it
needs to know which lines issue those launches and syncs.

torch.profiler's with_stack does not populate stacks in this build, so this
uses a TorchFunctionMode to time every op on the CPU side and blame it on the
caller's line in radius_graph_pbc.py. CPU wall time per op is the right measure
here: it captures both launch overhead and the stall when an op has to wait for
the device.
"""

from __future__ import annotations

import argparse
import collections
import time
import traceback

import torch
from torch.overrides import TorchFunctionMode

TARGET = "radius_graph_pbc.py"
# Ops whose result or output shape lives on the host, so the CPU must wait.
SYNC_NAMES = {
    "nonzero", "masked_select", "item", "unique", "_unique2", "equal", "all",
    "any", "tolist", "__bool__", "__float__", "__int__", "__index__", "max",
    "min", "sum",
}


class LineAttribution(TorchFunctionMode):
    """
    Time each torch op and attribute it to the calling line in TARGET.
    """

    def __init__(self):
        self.cpu = collections.Counter()
        self.calls = collections.Counter()
        self.ops = collections.defaultdict(collections.Counter)
        self.enabled = False

    def __torch_function__(self, func, types, a=(), kw=None):
        kw = kw or {}
        if not self.enabled:
            return func(*a, **kw)
        line = None
        for frame in reversed(traceback.extract_stack()):
            if frame.filename.endswith(TARGET):
                line = frame.lineno
                break
        if line is None:
            return func(*a, **kw)
        t0 = time.perf_counter()
        out = func(*a, **kw)
        self.cpu[line] += time.perf_counter() - t0
        self.calls[line] += 1
        self.ops[line][getattr(func, "__name__", str(func))] += 1
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--natoms", type=int, default=1000)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--cutoff", type=float, default=6.0)
    ap.add_argument("--max-neigh", type=int, default=300)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    from fairchem.core.datasets.atomic_data import AtomicData
    from fairchem.core.datasets.common_structures import get_fcc_crystal_by_num_atoms
    from fairchem.core.graph.radius_graph_pbc import radius_graph_pbc_v2

    atoms = get_fcc_crystal_by_num_atoms(args.natoms)
    data = AtomicData.from_ase(atoms, task_name="omat").to("cuda")

    def run():
        return radius_graph_pbc_v2(data, args.cutoff, args.max_neigh, False, data.pbc)

    for _ in range(5):
        run()
    torch.cuda.synchronize()

    mode = LineAttribution()
    with mode:
        mode.enabled = True
        for _ in range(args.iters):
            run()
        torch.cuda.synchronize()
        mode.enabled = False

    src = open(
        "src/fairchem/core/graph/radius_graph_pbc.py", encoding="utf-8"
    ).readlines()
    n = args.iters
    total = sum(mode.cpu.values()) * 1000 / n
    print(f"\nnatoms={args.natoms}   attributed CPU {total:.3f} ms/call, "
          f"{sum(mode.calls.values()) / n:.0f} torch calls/call\n")
    print(f"{'line':>5} {'cpu ms':>8} {'%':>5} {'calls':>6}  {'ops':<26} source")
    print("-" * 116)
    cum = 0.0
    for line, t in mode.cpu.most_common(args.top):
        ms = t * 1000 / n
        cum += ms
        text = src[line - 1].strip() if line - 1 < len(src) else "?"
        top_ops = ",".join(k for k, _ in mode.ops[line].most_common(3))
        print(f"{line:>5} {ms:>8.3f} {100 * ms / total:>4.0f}% "
              f"{mode.calls[line] / n:>6.1f}  {top_ops[:26]:<26} {text[:44]}")
    print(f"\n  top {args.top} lines = {cum:.3f} ms ({100 * cum / total:.0f}% of "
          f"attributed CPU)")

    # Phase boundaries in radius_graph_pbc.py. Everything before the grid
    # build depends only on (cell, pbc, natoms) -- not on positions -- so it is
    # constant across steps of a fixed-cell, fixed-composition simulation.
    phases = [
        ("canonical_pbc", 370, 400),
        ("A rep / cells_per_image   [cell-only]", 420, 482),
        ("B unit_cell + pbc offsets [cell-only]", 483, 506),
        ("C source atom expansion   [cell-only]", 507, 545),
        ("D grid resolution + floor", 546, 565),
        ("E per-image grid min/max  [cell-only]", 566, 615),
        ("F grid cell mapping", 616, 700),
        ("G neighbour cell enumeration", 701, 750),
        ("H distances + filtering", 751, 790),
        ("I get_max_neighbors_mask etc", 40, 200),
    ]
    print("\nby phase:")
    tot_cache = 0.0
    for name, lo, hi in phases:
        ms = sum(v for k, v in mode.cpu.items() if lo <= k <= hi) * 1000 / n
        calls = sum(v for k, v in mode.calls.items() if lo <= k <= hi) / n
        if "[cell-only]" in name:
            tot_cache += ms
        print(f"  {name:<40} {ms:>7.3f} ms  {calls:>5.0f} calls")
    print(f"\n  position-independent (cacheable) total: {tot_cache:.3f} ms "
          f"({100 * tot_cache / total:.0f}% of attributed CPU)")

    print("\nlines calling likely-synchronising ops:")
    for line, ops in mode.ops.items():
        hit = {k: v for k, v in ops.items() if k in SYNC_NAMES}
        if not hit:
            continue
        text = src[line - 1].strip() if line - 1 < len(src) else "?"
        print(f"{line:>5} {mode.cpu[line] * 1000 / n:>7.3f} ms  "
              f"{str(hit)[:40]:<40} {text[:46]}")


if __name__ == "__main__":
    main()
