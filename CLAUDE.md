# CLAUDE.md
FAIRChem is Meta FAIR Chemistry's ML framework for atomistic simulations. Core abstractions: foundation models (UMA) with backbone+heads architecture, ASE calculator integration, Hydra-based config, and multi-task training via TorchTNT.

## Development Commands

```bash
# Install
pip install -e packages/fairchem-core[dev]

# Tests (always pass -c flag)
pytest tests -c packages/fairchem-core/pyproject.toml
pytest tests/core/models/test_uma.py -vv
pytest tests/core -m "not gpu"

# Lint & format — REQUIRED for every modified file before committing
pre-commit run --files path/to/modified_file.py

# CLI
fairchem -c config.yaml [overrides...]
```

## Code Style

**IMPORTANT: You MUST run `pre-commit run --files /path/to/modified_file.py` on every file you modify, before considering the task complete. No exceptions.**

**Every file must start with:**
```python
"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations
```

**Line length**: 88 characters. **Linter**: Ruff (config in `ruff.toml`).

**Docstrings** use Google convention. No text on opening/closing quote lines:
```python
# WRONG
"""This is wrong."""

# RIGHT
"""
Short description.
"""

# RIGHT (with args)
"""
Short description.

Args:
    x: The input tensor.

Returns:
    The processed tensor.
"""
```

**Imports**: isort enforced via Ruff. `fairchem.core` is `known-first-party`. Use `TYPE_CHECKING` for type-only imports:
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fairchem.core.datasets.atomic_data import AtomicData
```

## Testing Conventions

All tests go in `tests/` (mirrors `src/fairchem/` structure). Always run with:
```bash
pytest tests -c packages/fairchem-core/pyproject.toml
```

### Test Markers
- `@pytest.mark.gpu`: GPU-only (auto-skipped when CUDA unavailable)
- `@pytest.mark.cpu_and_gpu`: Runs on both CPU and GPU
- `@pytest.mark.dgl`: Requires `fairchem_cpp`
- `@pytest.mark.inference_check`: Inference validation (skipped by default)

### Key Fixtures

**Root conftest (`tests/conftest.py`):**
- `seed_fixture` (function): Seeds all RNGs to 42
- `water_xyz_file` (session): Path to a minimal 3-atom water XYZ file
- `compile_reset_state` (function): Resets `torch.compiler` before/after test
- `setup_before_each_test` (autouse): Cleans up Ray, GPU memory, distributed state

**Core conftest (`tests/core/conftest.py`):**
- `dummy_binary_dataset` (session, parametrized): ASE dataset in both LMDB and CIF formats
- `fake_uma_dataset` (session): Full UMA training dataset path + config
- `direct_checkpoint` (session): Trained model checkpoint (inference + resume)
- `direct_mole_checkpoint` (session): Trained MOLE checkpoint
- `torch_deterministic` (function): Enables deterministic algorithms
- `snapshot` (function): Syrupy snapshot with approximate numpy comparison (`Approx`)

**Dataset conftest (`tests/core/datasets/conftest.py`):**
- `structures` (module): List of test atoms [H2O molecule, Cu bulk, Pt slab]

### Test Patterns

GPU/CPU dual tests:
```python
@pytest.mark.gpu
def test_something_gpu():
    _test_something("cuda")


def test_something_cpu():
    _test_something("cpu")


def _test_something(device):
    # shared implementation
    ...
```

Snapshot testing with approximate comparison:
```python
def test_values(snapshot):
    result = compute_something()
    assert pytest.approx(result.numpy(), abs=1e-3) == snapshot
```

Integration tests using the CLI:
```python
from tests.core.testing_utils import launch_main


def test_training(fake_uma_dataset):
    sys_args = [
        "--config",
        "tests/core/units/mlip_unit/test_mlip_train.yaml",
        f"datasets.data_root_dir={fake_uma_dataset}",
        "job.device_type=CPU",
        "max_steps=2",
    ]
    launch_main(sys_args)
```

## Architecture

### Model System (Backbone + Heads)

Models use `HydraModel` (registered as `"hydra"`): one backbone extracts features, multiple heads predict properties.

```
BackboneInterface.forward(data: AtomicData) -> dict[str, Tensor]   # features
HeadInterface.forward(data: AtomicData, emb: dict) -> dict[str, Tensor]  # predictions
```

Primary backbone: `escnmd_backbone` (SO(3)-equivariant eSCN with MD modifications).
Heads: `MLP_Energy_Head`, `Linear_Force_Head`, `DatasetSpecificSingleHeadWrapper`.

### Registry Pattern

Components are registered for dynamic Hydra instantiation:
```python
@registry.register_model("my_backbone")
class MyBackbone(nn.Module, BackboneInterface): ...
```

Available decorators: `register_model`, `register_dataset`, `register_loss`, `register_task`, `register_logger`, `register_trainer`.

Lookup: `registry.get_model_class("my_backbone")` or by full import path `"fairchem.core.models.my_module.MyBackbone"`.

### Data Flow

```
ASE Atoms -> AtomicData.from_ase() -> graph generation -> backbone -> heads -> predictions
```

`AtomicData` required fields: `pos, atomic_numbers, cell, pbc, natoms, edge_index, cell_offsets, nedges, charge, spin, fixed, tags`.
Optional targets: `energy, forces, stress`.

Batching via `atomicdata_list_to_batch()`. Multi-task collation via `MTCollater` (fills missing targets with `inf` for loss masking).

### Configuration (Hydra)

YAML configs use `_target_` keys for component instantiation:
```yaml
runner:
  _target_: fairchem.core.components.train.train_runner.TrainEvalRunner
  train_eval_unit:
    _target_: fairchem.core.units.mlip_unit.mlip_unit.MLIPTrainEvalUnit
    model:
      _target_: fairchem.core.models.base.HydraModel
      backbone: ${backbone}
```

Config sections: `job`, `runner`, `datasets`, `tasks`, `backbone`, `optimizer`.
Default configs in `configs/`. Overrides via CLI: `fairchem -c config.yaml key=value`.

### Task Names
- `oc20`: Catalysis (Open Catalyst)
- `omat`: Inorganic materials
- `omol`: Molecules
- `odac`: Metal-organic frameworks
- `omc`: Molecular crystals

### Training Flow

`TrainEvalRunner` orchestrates training via TorchTNT's `fit()`. Core unit: `MLIPTrainEvalUnit` (handles forward pass, loss, metrics, EMA, gradient clipping). Checkpoints use DCP (Distributed Checkpoint Protocol) with `dcp_to_torch_save()` for inference export.

### Model Loading and Inference

```python
from fairchem.core import pretrained_mlip, FAIRChemCalculator

predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
calc = FAIRChemCalculator(predictor, task_name="oc20")
atoms.calc = calc
```

## Repository Structure

```
src/fairchem/core/
├── models/              # Backbones and heads (UMA, eSCN-MD, GemNet)
├── datasets/            # Data loading (LMDB, ASE), collaters, samplers
├── components/          # Runner components (train, evaluate, calculate)
├── units/               # TorchTNT train/eval/predict units
├── modules/             # Loss, schedulers, normalizers, evaluators
├── launchers/           # Local, Ray, SLURM job launchers
├── common/              # Registry, distributed utils, logging
├── graph/               # Graph generation, neighbor finding with PBC
└── _cli.py              # CLI entry point

tests/                   # All tests (mirrors src structure)
packages/                # Installable packages (fairchem-core, fairchem-data-*)
configs/                 # Hydra YAML configs (datasets, tasks, backbone, optimizer)
```

## Key Dependencies

- `torch~=2.13.0`, `e3nn>=0.5` - PyTorch + equivariant neural networks
- `ase>=3.26.0` - Atomic Simulation Environment
- `torchtnt` - PyTorch training framework (TrainUnit/EvalUnit)
- `hydra-core` + `omegaconf` - Configuration management
- `lmdb` - Dataset storage format
- `ray[serve]>=2.53.0` - Distributed computing

## Numerical Precision

- Model constructors must not mutate process-wide PyTorch precision settings
  such as `torch.set_float32_matmul_precision`. Precision is caller-owned;
  inference applies TF32 temporarily through `InferenceSettings.tf32` and
  restores the prior settings afterward.
- TF32 policy belongs to the training/evaluation unit config or
  `InferenceSettings.tf32`, never to a model config or model attribute.
  Execution callers scope and restore the policy outside compiled `forward`
  methods because precision getters cannot be traced by fullgraph.
- Training and evaluation units default TF32 to disabled. Configs should set
  `tf32` only when overriding that default. Hydra CLI overrides for configs
  that omit the key must use the add syntax, such as
  `+runner.train_eval_unit.tf32=true`.
- Keep one configurable TF32 context manager for scoped matmul precision and
  cuDNN state instead of introducing overlapping context managers.
- Use the `tf32_context_manager` name for that policy; it controls both matmul
  precision and cuDNN TF32, so `matmul_context` is too narrow.
- Training FLOPs profiling invokes the model from `on_train_start`; scoped
  execution settings must cover profiling as well as train/eval step methods.

## Cluster Validation Gotchas

- H100 compute nodes do not have PyPI egress. Provision Python environments on
  the submission host before launching validation jobs.
- Imports from home-backed virtual environments are extremely slow on H100
  nodes. Copy complete environments and large checkpoints to node-local scratch
  before running tests or benchmarks.
- Pretrained checkpoints are cached under `~/.cache/fairchem`. Set
  `HF_HUB_OFFLINE=1` in compute jobs to prevent blocked Hugging Face metadata
  requests when the required files are already cached.
- Separate Hugging Face downloads can populate different snapshots while
  `refs/main` points only to the latest one. Ensure the active snapshot contains
  every checkpoint needed by offline tests.
- Core test collection imports benchmark and calculation modules through the
  shared conftest. Validation environments need the `extras` dependencies,
  including `pandas`, `pyarrow`, and `pymatgen`, even for focused test subsets.
- Some GPU assertions are stochastic or tolerance-sensitive, and the complete
  GPU matrix is expensive. Reproduce failures with the exact test node (and
  repeat it when appropriate) before rerunning a full GPU shard.

## Graph-Parallel Comm Gotchas

- UMA is numerically nondeterministic run to run: edge messages scatter with
  `index_add_`/`scatter_add` atomics. At 30k atoms this is ~1.5e-1 eV total
  energy and ~8e-4 eV/A max force. Never judge a change by a single A-vs-B
  comparison; sample both paths several times and compare the cross-spread
  against each path's own self-spread. A single pair will show a ~4x "deviation"
  that is pure noise. Note the 8e-4 eV/A floor already exceeds the A2A PR's
  1e-4 eV/A tolerance, so that tolerance cannot be applied to max-abs force
  error on large systems.
- NCCL `all_to_all_single` badly underutilizes the link at UMA's halo sizes
  (few MB split across many peers). Measured on 4x H100 over PCIe/PHB: 3.6 GB/s
  versus 53.4 GB/s raw `cudaMemcpyPeer`. Direct peer writes through
  `torch.distributed._symmetric_memory` reach ~50% of peak, 14.8x faster on the
  isolated exchange and ~15-19% end to end. See
  `common/parallelism/symm_halo.py`, enabled with
  `job.graph_parallel.transport=symm_mem`.
- Comm-compute overlap by splitting `forward_chunk` was implemented and reverted
  on `origin/rgao_a2a_dev` (`dba967bc6`), 12-17% slower on H200. That cost is a
  **torch.compile graph-break artifact, not intrinsic**: replaying the real fused
  edgewise over 432k edges as two calls costs only 0.6-1.8% with compile off, and
  is flat across split ratios from 50/50 to 90/10. The GP forward already forces
  11 graph breaks / 12 graphs, almost all from `@torch.compiler.disable`d
  collectives; hand-splitting adds more. Fix the tracing before re-attempting
  overlap. The `local_edge_idx`/`remote_edge_idx` fields are leftovers from that
  attempt, not an unfinished intent.
- Peer puts genuinely overlap with SM compute: a 26 MB symmetric-memory transfer
  issued on side streams alongside a 16 ms GEMM block costs no extra wall time.
  Overlap is blocked by the compiler boundary, not by the hardware.
- Boundary-first (per-node) overlap cannot work at scale. The share of a rank's
  own atoms that some peer needs is 21% at P=4 but 82% at P=64 and 93% at P=128
  (N=100k, rho=0.085), so the interior left to hide comm behind vanishes exactly
  where comm matters. An edge split (local-source vs remote-source, 74-81% local
  at P=64) does retain cover and is the only viable split.
- Morton partitioning gives a rank only ~18-21 active peers regardless of P
  (geometric neighbour count saturates): 28% of ranks at P=64, 17% at P=128. Both
  NCCL all-to-all and `SymmetricMemory.barrier()` synchronize with all P, so sync
  is O(P) where it could be O(active peers).
- Per-peer `put_signal`/`wait_signal` is NOT unconditionally better: it costs one
  kernel launch per peer against the barrier's one total. Measured at P=4, where
  3 of 4 peers are active, signals cost 5.91 ms/step against the barrier's
  2.61 ms. `_exchange` therefore picks signals only when neighbours are sparse
  (`2 * n_readers < world_size`), which under Morton kicks in around P>=32. Do
  not "simplify" this back to always-signals without measuring at high P.
- Double buffering the symmetric allocation removes the pre-write barrier
  outright: a peer racing to exchange k+1 writes the other buffer, and it cannot
  reach k+2 without first consuming this rank's k+1 contribution. Two barriers
  per exchange became one, 2.61 -> 2.24 ms/step at P=4.
- A sync path that only triggers at high peer sparsity is easy to leave
  untested. `test_symm_halo.py` runs both a dense and a ring count matrix so the
  barrier and signal paths are each covered at world_size=3.
- The `x_full = torch.cat([x, x_received])` halo concatenation is 0.21 ms/step
  at 30k/GP=4, i.e. negligible. The large `CatArrayBatchedCopy` time in profiles
  belongs to other cats (graph gen, wigner, edge embedding). Do not "optimize"
  the halo cat without measuring it in isolation first.
- Under spatial partitioning the halo exceeds the local partition at high rank
  counts: at N=100k/P=64 it is 2326 halo nodes against 1562 owned. Comm volume
  per rank decays only as P^-1/3 while owned nodes decay as P^-1.
- Profiler `self_device_time_total` summed over all events double-counts,
  because `record_function` regions are credited alongside their child kernels.
  Total came out 2.8x wall time. Filter to leaf kernels or use wall clock.

## Environment Gotchas

- e3nn ships a `.pt` cache containing `slice`, which torch 2.14's default
  `weights_only=True` rejects at import time. Run tests with
  `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`, or call
  `torch.serialization.add_safe_globals([slice])` before importing fairchem.
- Several multi-process GPU tests fail on a 4-GPU dev box independently of any
  change: `test_a2a_vs_allgather_gpu[4-edges0]`, `test_a2a_backward_gpu`,
  `test_a2a_multi_rank_gpu[2,3]`, `test_a2a_spatial_partition_gpu`, and the
  `test_*_wigner_*_gradcheck` cases. Always confirm a failure against a clean
  tree (`git stash push -- src/ tests/`) before attributing it to your change.
- `pre-commit run` cannot fetch hook repos from this environment (proxy blocks
  github.com). Run `ruff check --config ruff.toml` and `ruff format --config
  ruff.toml` directly instead.
- Allgather GP mode OOMs at 30k atoms on 4x 95GB H100 without activation
  checkpointing; A2A mode fits. A2A at GP=4 OOMs by 60k atoms.

## Hessian Backend Gotchas

- PyTorch's generic `vmap` fallback cannot batch the mutable, output-argument
  Triton operators used by `umas_fast_gpu`. Backend validation rejects requested
  Hessians with `hessian_vmap=True`; set it to `False` until the backward
  operators have explicit batching rules. Automatic backend selection falls
  back to normal mode for this combination. This only changes Hessian
  construction: energy, force, and stress inference are unaffected. The
  fallback computes one vector-Jacobian product per Cartesian force component,
  so it can be slower for large systems while using less memory.
- Explicit `torch.library.register_vmap` rules are possible for mutable custom
  operators. A rule that loops over the mapped dimension would make the
  operator compatible but retain most kernel-launch overhead. Recovering the
  performance value of vectorized Hessians requires rules backed by genuinely
  batched Triton kernels, including every custom backward operator reached by
  the derivative graph.

## Dependency Compatibility

- `pymatgen` and `pymatgen-core` are independently versioned. Slab tests must
  not depend on enumeration order, seeded random coordinates, or atom counts
  unless those values are part of the public contract; prefer composition,
  Miller index, shift, placement, and cell invariants that survive upgrades.

Anytime we learn something that could be beneficial in future coding sessions, automatically add it to CLAUDE.md.

This includes:
- Gotchas that are not obvious
- Subtle bugs that manifest under specific conditions
- Repeat corrections I make to the output of coding agents
