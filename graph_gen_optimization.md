# Internal graph generation: profiling and optimization

Measured 2026-07-31 on an H100 96 GB (132 SMs, 63 MB L2), torch 2.14.0a0,
model `uma-s-1p2`, `execution_mode=umas_fast_gpu`, `internal_graph_gen_version=2`.

## Summary

`radius_graph_pbc_v2` is **1.82× faster** at 1000 atoms (3.96 → 2.17 ms per
call), with kernel launches down 412 → 228 and device-to-host syncs down
99 → 43. The graph produced is bit-identical on every system tested.

Three changes, all local, none requiring a rewrite to static shapes:

1. Cache the position-independent part of the PBC tiling.
2. Remove three sources of device-to-host syncs.
3. Delete a dead `torch.unique` over every edge.

## Why graph generation was worth attacking

Toggling `InferenceSettings.external_graph_gen` isolates its cost: 24.20 qps
with internal graph generation versus 28.29 without, i.e. **5.9 ms of the
41.3 ms step (14.5%)** at 1000 atoms. That was the largest single addressable
item on the inference path.

The important finding is that it is **not GPU-bound**:

| backend | wall | GPU kernels | GPU idle | launches | syncs |
|---|---:|---:|---:|---:|---:|
| v2 (default) | 3.84 ms | 1.07 ms (28%) | **2.77 ms (72%)** | 412 | **99** |
| v3 (nvalchemiops) | 2.63 ms | 0.58 ms (22%) | **2.05 ms (78%)** | 127 | 22 |

Under 1 ms of real GPU work; the rest is `cudaLaunchKernel` (1.25 ms),
`cudaStreamSynchronize` (0.42 ms) and torch dispatch. Threading or SIMD do not
apply — the CPU time is not compute. The lever is issuing fewer ops.

A line-level profile (`tools_profile_v2_lines.py`, using a `TorchFunctionMode`
because `torch.profiler`'s `with_stack` does not populate in this build) showed
**314 tensor ops per call and no dominant line** — the top 25 lines were only
41% of CPU time. Death by a thousand tiny ops, so the fix had to be removing
ops rather than speeding any one up.

## 1. Cache the position-independent PBC scaffold

Bucketing the line profile by phase showed **46% of the CPU time (2.39 ms,
142 of 314 ops) depends only on `(cell, pbc, natoms, radius)`** — not on atom
positions:

| phase | CPU | ops | position-independent |
|---|---:|---:|:--:|
| rep / cells_per_image | 0.58 ms | 53 | yes |
| unit_cell + pbc offsets | 0.63 ms | 38 | yes |
| source atom expansion | 0.45 ms | 20 | yes |
| per-image scatter map | 0.73 ms | 31 | yes |
| grid build, neighbour enumeration, distances, trim | 2.69 ms | 163 | no |

A fixed-cell, fixed-composition simulation — exactly what
`inference_settings_turbo` is documented for — recomputes those identical
tensors on every step.

`_PbcScaffold` holds them, in a bounded LRU (`_SCAFFOLD_CACHE`, 4 entries)
keyed on the host-side values of cell, pbc, natoms, radius, device and batch
size. Building the key costs a few tiny syncs (~3 µs each), negligible against
the work skipped. Disable with `FAIRCHEM_PBC_SCAFFOLD_CACHE=0`.

Keying on values rather than tensor identity is deliberate: a changing cell
(NPT, cell relaxation) must miss the cache, and a stale scaffold would silently
produce a wrong graph.

## 2. Remove device-to-host syncs

Syncs come in two kinds, and only one needs the static-shape rewrite:

- **Shape syncs** — a device value used as a tensor dimension. Needs the host
  value or a fixed upper bound.
- **Control-flow syncs** — a Python `if` on a device scalar. Often removable.

Three removals, none needing static shapes:

- **Share one `nonzero` across masked selects.** Several `masked_select` calls
  using the same mask each sync separately to learn their own output size.
  Resolving the mask once and gathering with the result gives the gathers a
  known shape, so they add no further sync. **5 syncs → 2.**
- **`get_max_neighbors_mask` returns `None`** when no trimming is needed,
  instead of an all-True mask that every caller tested with `torch.all(...)` —
  a sync plus a full reduction. Callers now branch in Python. Three call sites,
  including `radius_graph_pbc_nvidia.py`.
- **`is_mixed_pbc`** did up to six `.item()` calls in a loop; now one
  `.tolist()` and pure-Python logic. Batched inputs only.

## 3. Delete a dead `torch.unique`

`num_neighbors_image` was computed by a `torch.unique` over every edge and then
unconditionally overwritten by `get_max_neighbors_mask` on the next line. A
full sort plus a sync, per call, for a discarded result.

## Results

### Component, `radius_graph_pbc_v2` at 1000 atoms

| | ms/call | vs original |
|---|---:|---:|
| original | 3.955 | — |
| + sync removals + dead code | 3.358 | 1.18× |
| + scaffold cache | **2.168** | **1.82×** |

Kernel launches 412 → 228; syncs 99 → 43.

### End-to-end sweep

In-process paired A/B, alternating cache on/off across 6 rounds of 30
iterations per size, so slow drift affects both arms equally. This isolates the
**scaffold cache only** — the sync removals and dead-code deletion have no
runtime switch, so their (positive) contribution is not included here and the
branch total is higher.

| natoms | edges | ms/step | qps | cache-off ms | gain | rounds |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 4,318 | 13.43 | 74.47 | 14.98 | **+11.6%** | 6/6 |
| 250 | 9,832 | 14.35 | 69.69 | 15.98 | **+11.3%** | 6/6 |
| 500 | 26,364 | 24.47 | 40.86 | 25.40 | **+3.8%** | 6/6 |
| 1000 | 54,000 | 40.62 | 24.62 | 42.24 | **+4.0%** | 6/6 |
| 2000 | 98,302 | 66.90 | 14.95 | 68.37 | **+2.2%** | 6/6 |
| 4000 | 210,940 | 132.62 | 7.54 | 134.06 | **+1.1%** | 6/6 |

The gain decays with system size because the work removed is CPU-side and
largely fixed while GPU work grows with atom count. Small systems benefit most:
before these changes, 100 and 250 atoms ran at effectively the same speed
(59.4 vs 59.7 qps), i.e. they were entirely overhead-bound.

Note the sweep uses FCC crystals. Molecules, slabs and mixed-PBC batches have
different edge densities and periodic-image counts, so the speedup curve will
differ; correctness was verified on those, the speedup curve was not.

## Correctness

- Graph bit-identical with the cache on versus off, cold and warm, on H2O,
  CH4, Cu bulk, Cu primitive, Pt(111) slab, rattled FCC, Ti hcp and FCC 1000.
- **Moving atoms at fixed cell** — the case the cache targets — correct on every
  hit across 4 MD-like steps.
- **Changing the cell invalidates correctly** (a = 3.6 → 3.8 → 3.6 Å).
- v2 output still matches `radius_graph_pbc_nvidia` (v3) exactly across 10
  system types including a mixed-PBC batch.
- `pytest tests/core/graph/`: 136 passed. The 2 failures
  (`TestMixedPBCBatch::test_inference_results_match_mixed_vs_individual`) fail
  identically on a clean tree — they download `uma-s-1p2` from HuggingFace.

## What is left

- `num_grid_cells` and `max_atoms_per_grid_cell` are device values used as
  tensor sizes. These are true shape syncs; removing them means allocating at
  an upper bound, i.e. the static-shape route.
- The `max_internal_cell > 200` guard costs a full reduction over the tiled
  positions plus a sync (~0.09 ms).
- `generate_graph` is `@torch.compiler.disable()`d, so all of this runs eager.
- `radius_graph_pbc_nvidia` (version 3, needs the optional `nvalchemiops`) runs
  the neighbour search in 1.81 ms versus v2's post-optimization 2.17 ms, and
  produces bit-identical graphs on all 10 system types tested. A graceful
  fallback to v2 when the dependency is absent is included here; promoting v3
  to a default is deliberately left as a separate decision.

### Why this also blocks CUDA graphs

`torch.compile(mode="reduce-overhead")` was measured and is **slower**
(38.61 → 39.52 ms at 1000 atoms). Cudagraphs are captured, but the tree manager
holds only 6 nodes against ~1131 kernels per step, with
`cudagraph_recorded_non_static_inputs: 33` — 33 tensors copied into static
buffers on every replay. Coverage is small and the copies are not.

The cause is the same as above: `generate_graph` is `@torch.compiler.disable()`d
and sits mid-step with syncs and data-dependent shapes, so no graph can span it
and the step fragments. Making graph generation shape-static and sync-free would
address both the remaining CPU cost and the ~6.3 ms (16%) GPU-idle gap.
`InferenceSettings.edge_chunk_size` looks like the edge-padding hook for that
but is currently dead code — plumbed to `escn_md.py` and never read.

## Reproducing

```bash
# component timing and sync/launch counts
python tools_profile_graph_gen.py --natoms 1000 --versions 2
# line-level CPU attribution
python tools_profile_v2_lines.py --natoms 1000
# graph equivalence across system types
python tools_graph_identity_check.py --versions 2 3
# end-to-end sweep, cache isolated
python tools_sweep_qps.py --natoms 100 250 500 1000 2000 4000 --only scaffold_cache
```

The speed benchmark needs two overrides, as the shipped yaml hardcodes a
non-writable `run_dir` and downloads the checkpoint:

```bash
HF_HUB_OFFLINE=1 fairchem -c configs/uma/speed/uma-speed.yaml \
  job.run_dir=/tmp/uma_speed_runs \
  '++uma_s_1p2=/path/to/uma-s-1p2.pt'
```
