"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Does torch.compile(mode="reduce-overhead") help this inference path?

About 7 ms of the ~39 ms step is GPU idle, and graph generation alone issues
hundreds of tiny kernels, so CUDA graphs look attractive. Two things work
against them here: predict.py compiles with dynamic=True (cudagraphs want
static shapes), and generate_graph is @torch.compiler.disable()d, which breaks
the step into several compiled regions with eager, sync-heavy code between them.

This patches the compile call in predict.py to try each mode, records whether
inductor actually captured cudagraphs or skipped them and why, and checks
outputs against the current setting.
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import statistics
from contextlib import redirect_stderr

import numpy as np
import torch

import fairchem.core.units.mlip_unit.predict as predict_mod
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.common_structures import get_fcc_crystal_by_num_atoms
from fairchem.core.units.mlip_unit import MLIPPredictUnit
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

CKPT = "/home/avenkataraman/github/kUPS/examples/uma-s-1p2.pt"
SKIP_RE = re.compile(r"skipping cudagraphs due to ([^\n\"]*)", re.I)

VARIANTS = [
    ("baseline dynamic=True", dict(dynamic=True)),
    ("dynamic=True  + reduce-overhead", dict(dynamic=True, mode="reduce-overhead")),
    ("dynamic=False + reduce-overhead", dict(dynamic=False, mode="reduce-overhead")),
    ("dynamic=None  + reduce-overhead", dict(dynamic=None, mode="reduce-overhead")),
]


def build(checkpoint: str, compile_kwargs: dict):
    """
    Build a predictor, overriding the torch.compile call inside predict.py.
    """
    real = torch.compile

    def patched(model, *a, **kw):
        return real(model, **compile_kwargs)

    predict_mod.torch.compile = patched
    try:
        return MLIPPredictUnit(
            checkpoint,
            "cuda",
            inference_settings=InferenceSettings(
                tf32=True,
                activation_checkpointing=False,
                merge_mole=True,
                compile=True,
                external_graph_gen=False,
                internal_graph_gen_version=2,
                execution_mode="umas_fast_gpu",
            ),
        )
    finally:
        predict_mod.torch.compile = real


def timed(predictor, data, iters: int) -> float:
    torch.cuda.synchronize()
    s, e = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    s.record()
    for _ in range(iters):
        predictor.predict(data)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--natoms", type=int, default=1000)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--checkpoint", default=CKPT)
    args = ap.parse_args()

    torch.manual_seed(1)
    np.random.seed(1)
    atoms = get_fcc_crystal_by_num_atoms(args.natoms)
    logging.getLogger().setLevel(logging.ERROR)

    ref = None
    print(f"\nnatoms={args.natoms}  iters/round={args.iters} rounds={args.rounds}\n")
    for name, kwargs in VARIANTS:
        torch._dynamo.reset()
        data = AtomicData.from_ase(atoms, task_name="omat")
        buf = io.StringIO()
        try:
            with redirect_stderr(buf):
                torch._logging.set_logs(perf_hints=True)
                p = build(args.checkpoint, kwargs)
                for _ in range(12):
                    out = p.predict(data)
                torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001 - a mode may simply not work
            print(f"  {name:<34} FAILED: {type(exc).__name__}: {str(exc)[:100]}")
            continue
        finally:
            torch._logging.set_logs()

        log = buf.getvalue()
        skips = sorted(set(SKIP_RE.findall(log)))
        ran = [timed(p, data, args.iters) for _ in range(args.rounds)]
        ms = statistics.median(ran)

        if ref is None:
            ref = {k: v.detach().clone() for k, v in out.items()}
            delta = "reference"
        else:
            d = max(
                (ref[k] - out[k]).abs().max().item() for k in ("energy", "forces")
            )
            delta = f"max |d| vs baseline {d:.2e}"
        print(f"  {name:<34} {ms:>7.3f} ms  ({1000 / ms:.2f} qps)  {delta}")
        if skips:
            for s in skips[:3]:
                print(f"       cudagraph skipped: {s.strip()[:88]}")
        elif "reduce-overhead" in str(kwargs):
            print("       (no skip message captured)")

        del p
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
