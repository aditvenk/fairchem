"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Break down the cost of internal graph generation.

Graph generation is ~5.9 ms of the 41.3 ms inference step (measured by
toggling InferenceSettings.external_graph_gen). generate_graph is
@torch.compiler.disable()d, so it runs eager, and radius_graph_pbc_v2 contains
several device-to-host syncs (tensor values used as tensor sizes, Python `if`
on GPU scalars, masked_select). This reports, per backend, how much of the cost
is GPU kernels, how much is CPU, and how many syncs occur.
"""

from __future__ import annotations

import argparse
import collections

import torch
from torch.profiler import ProfilerActivity, profile

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.common_structures import get_fcc_crystal_by_num_atoms
from fairchem.core.graph.compute import generate_graph
from fairchem.core.units.mlip_unit import MLIPPredictUnit
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

CKPT = "/home/avenkataraman/github/kUPS/examples/uma-s-1p2.pt"
# Ops whose output shape or control flow depends on device data, so each one
# blocks the CPU until the GPU catches up.
SYNC_OPS = (
    "aten::nonzero",
    "aten::masked_select",
    "aten::item",
    "aten::_local_scalar_dense",
    "aten::unique",
    "aten::_unique2",
    "aten::equal",
)


def get_backbone_params(checkpoint: str):
    settings = InferenceSettings(
        tf32=True, activation_checkpointing=False, merge_mole=True, compile=False,
        external_graph_gen=False, internal_graph_gen_version=2,
        execution_mode="umas_fast_gpu",
    )
    p = MLIPPredictUnit(checkpoint, "cuda", inference_settings=settings)
    bb = p.model.module.backbone
    out = (bb.cutoff, bb.max_neighbors)
    del p
    torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--natoms", type=int, default=1000)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--versions", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--checkpoint", default=CKPT)
    args = ap.parse_args()

    cutoff, max_neighbors = get_backbone_params(args.checkpoint)
    atoms = get_fcc_crystal_by_num_atoms(args.natoms)
    data = AtomicData.from_ase(atoms, task_name="omat").to("cuda")
    print(f"natoms={args.natoms} cutoff={cutoff} max_neighbors={max_neighbors}\n")

    for v in args.versions:
        def run(version=v):
            return generate_graph(
                data, cutoff=cutoff, max_neighbors=max_neighbors,
                enforce_max_neighbors_strictly=False,
                radius_pbc_version=version, pbc=data.pbc,
            )

        for _ in range(5):
            run()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            run()
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / args.iters

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(args.iters):
                run()
            torch.cuda.synchronize()

        ka = prof.key_averages()
        gpu = [
            e for e in ka
            if e.self_device_time_total > 0 and str(e.device_type) != "DeviceType.CPU"
        ]
        gpu_ms = sum(e.self_device_time_total for e in gpu) / 1000 / args.iters
        cpu_ops = [e for e in ka if str(e.device_type) == "DeviceType.CPU"]
        n_launches = sum(e.count for e in gpu) / args.iters
        syncs = [e for e in cpu_ops if e.key in SYNC_OPS]
        n_sync = sum(e.count for e in syncs) / args.iters

        print(f"=== radius_pbc_version={v} ===")
        print(f"  wall            {ms:>7.3f} ms/call")
        print(f"  gpu kernel time {gpu_ms:>7.3f} ms/call  ({100 * gpu_ms / ms:.0f}%)")
        print(f"  gpu idle        {ms - gpu_ms:>7.3f} ms/call  "
              f"({100 * (ms - gpu_ms) / ms:.0f}%)")
        print(f"  kernel launches {n_launches:>7.0f} /call")
        print(f"  sync ops        {n_sync:>7.0f} /call  "
              f"({', '.join(sorted({e.key for e in syncs})) or 'none'})")
        print("  top GPU kernels:")
        for e in sorted(gpu, key=lambda x: -x.self_device_time_total)[:6]:
            print(f"    {e.self_device_time_total / 1000 / args.iters:>7.3f} ms  "
                  f"{e.count / args.iters:>5.0f}x  {e.key[:60]}")
        print("  top CPU ops by self time:")
        agg = collections.Counter()
        for e in cpu_ops:
            agg[e.key] += e.self_cpu_time_total
        for k, t in agg.most_common(6):
            print(f"    {t / 1000 / args.iters:>7.3f} ms  {k[:60]}")
        print()


if __name__ == "__main__":
    main()
