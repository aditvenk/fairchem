# Fused wigner↔SO2-conv edgewise ops (umas_fast_gpu)

A memory-bandwidth optimization for UMA inference on the `umas_fast_gpu` backend.
It fuses the Wigner rotation / repacking "glue" around the SO2 convolutions into two
Triton custom ops so the large per-edge intermediates never round-trip through DRAM.

**Bit-exact fp32. ~1.36× faster and ~39% lower peak memory at 1000 atoms. cuBLAS GEMMs
and all other backends / training paths are unchanged.**

---

## 1. Why: UMA inference is HBM-bandwidth-bound

At natoms=1000, uma-s-1p2 fwd+bwd (compile + `umas_fast_gpu` + tf32), measured with
Nsight Compute:

- **82.9 GB DRAM traffic**, ~52 ms kernel time, ~1110 launches, **1.59 TB/s achieved
  (~47% of HBM peak)**.
- Matmul-FLOP compute Speed-of-Light is only **~3.4 ms (~6% of tf32 peak)**.

So it is memory/traffic-bound, not compute-bound — the lever is *bytes moved*, not FLOPs.

DRAM traffic by op family: Inductor elementwise **30.9 GB (37%)**, SO2 GEMMs **23 GB
(28%)**, the two hand-written Wigner Triton ops **15 GB (18%)**, gather+scatter **13 GB
(16%)**. Per-kernel SoL shows the elementwise and Wigner families already run at **77–93%
of the DRAM roofline** (individually maxed out); the GEMMs sit at ~55% SM / 22% occupancy,
which is intrinsic to tf32 tensor cores on these skinny shapes, not exploitable headroom.

## 2. The target: the M-major `x_message` round-trips at the op seams

`Edgewise.forward_chunk` (`escn_md_block.py`), per layer:

```
x_message = backend.node_to_edge_wigner_permute(x_full, edge_index, wigner)  # [E,9,2C]
x_message, gate = so2_conv_1(x_message, x_edge)   # radial-scale + pack + cuBLAS GEMMs
x_message = act(gate, x_message)
x_message = so2_conv_2(x_message)                 # cuBLAS GEMMs + cat repack
new_emb  = backend.permute_wigner_inv_edge_to_node(x_message, ...)  # M→L + inv-rot + scatter
```

`x_message` is `[E, 9, 2C]` ≈ **500 MB/layer**, written by the Wigner op and re-read by
conv1; conv1's radial-scale + M→GEMM packing is what Inductor materializes as ~289
elementwise kernels / 30.9 GB. The same pattern exists on the output side (conv2 → M→L
unpack → inverse rotation).

**Why the compiler can't fix it:** the Wigner kernels are registered via
`torch.library.triton_op`, which Inductor treats as fusion barriers; it also cannot fuse
into the extern cuBLAS GEMMs. Verified empirically:
`epilogue_fusion_user_defined_triton_kernel` is blocked by the multi-store (18-store)
Wigner kernels; `execution_mode=general` (letting Inductor fuse the pure-PyTorch path) is
**+15% worse**; max-autotune's Triton GEMM templates lose to cuBLAS. The only way to kill
the `x_message` round-trip is to fuse *into* the custom op by hand.

## 3. Computation graph: before and after

Legend:

```
[T] = hand-written Triton custom kernel (torch.library.triton_op / custom_op — opaque to Inductor)
[B] = cuBLAS extern GEMM (opaque to Inductor)
[I] = Inductor (torch.compile) codegen'd elementwise/gather kernels
══▶ HBM = big intermediate round-trips to DRAM (the bottleneck)
```

### BEFORE (umas_fast_gpu baseline)

```
 x_full[N,9,C]   edge_index   wigner[E,9,9]   radial[E,·]
      │
      ▼
 ┌──────────────────────────────────────┐
 │ node_to_edge_wigner_permute          │ [T]  gather + Wigner-rot + L→M
 └──────────────────────────────────────┘
      │══▶ HBM  x_message[E,9,2C] ~500MB   (+ x_edge stash ~500MB, for bwd)
      ▼
 ┌──────────────────────────────────────┐
 │ conv1: radial-scale  + M→GEMM pack   │ [I]  ~cat/mul/split/view  ← the 30.9 GB "glue"
 └──────────────────────────────────────┘
      │══▶ HBM  m0/m1/m2 packed buffers
      ▼
 ┌──────────────────────────────────────┐
 │ conv1: fc_m0 + block GEMMs (@Wᵀ)     │ [B]  cuBLAS
 └──────────────────────────────────────┘
      │══▶ HBM  (+ cat repack [I]) → x_message
      ▼
 ┌──────────────────────────────────────┐
 │ gate / activation                    │ [I]
 └──────────────────────────────────────┘
      │══▶ HBM  x_message
      ▼
 ┌──────────────────────────────────────┐
 │ conv2: M-split + block GEMMs (@Wᵀ)   │ [B]  cuBLAS
 └──────────────────────────────────────┘
      │══▶ HBM  (+ cat repack [I]) → x_message[E,9,C] M-major
      ▼
 ┌──────────────────────────────────────┐
 │ permute_wigner_inv (M→L + inv-rot)   │ [T]  (+ x_l stash, for bwd)
 └──────────────────────────────────────┘
      │
      ▼  index_add scatter [aten] ──► new_embedding[N,9,C]
```

Inductor cannot fuse the `[I]` glue into `[T]` (custom-op barrier) or into `[B]` (extern
cuBLAS), so every seam round-trips `x_message` (~500 MB) through HBM.

### AFTER (fused)

```
 x_full[N,9,C]   edge_index   wigner[E,9,9]   radial[E,·]
      │
      ▼
 ┌────────────────────────────────────────────────────┐
 │ wigner_conv1_fused_op                    [T] NEW    │
 │  gather + Wigner-rot + L→M + radial-scale + pack   │   ← absorbs the input-side [I] glue
 │  (x_message stays in registers; no x_edge stash)   │      + kills the x_message round-trip
 └────────────────────────────────────────────────────┘
      │══▶ HBM  m0/m1/m2 packed  (mandatory GEMM inputs — unavoidable)
      ▼
 ┌──────────────────────────────────────┐
 │ conv1: fc_m0 + block GEMMs (@Wᵀ)     │ [B]  cuBLAS   (UNCHANGED)
 └──────────────────────────────────────┘
      │
      ▼
 ┌──────────────────────────────────────┐
 │ gate / activation                    │ [I]  (small; only surviving glue)
 └──────────────────────────────────────┘
      │
      ▼
 ┌──────────────────────────────────────┐
 │ conv2: block GEMMs (@Wᵀ) → g0/g1/g2  │ [B]  cuBLAS   (UNCHANGED, no cat)
 └──────────────────────────────────────┘
      │  g0[E,3C] g1[E,4C] g2[E,2C]
      ▼
 ┌────────────────────────────────────────────────────┐
 │ wigner_inv_conv2_fused_op                [T] NEW    │
 │  M→L unpack + inv-Wigner-rot → x_rotated[E,9,C]     │   ← absorbs the output-side [I] glue
 │  (no x_l stash)                                    │      + kills the M-major round-trip
 └────────────────────────────────────────────────────┘
      │
      ▼  index_add scatter [aten] ──► new_embedding[N,9,C]
```

### What moved

| Work | Before | After |
|---|---|---|
| gather + Wigner-rot + L→M | `[T]` | fused into `[T]` producer |
| conv1 radial-scale + pack | `[I]` (round-trips `x_message`) | **absorbed into `[T]` producer** |
| conv1 / conv2 GEMMs | `[B]` cuBLAS | `[B]` cuBLAS (**unchanged**) |
| gate / act | `[I]` | `[I]` (unchanged) |
| conv2 cat-repack | `[I]` (round-trips `x_message`) | **absorbed into `[T]` consumer** |
| M→L + inv-Wigner-rot | `[T]` | fused into `[T]` consumer |
| scatter | aten | aten (**unchanged, stays outside the op**) |
| bwd activation stash | `x_edge` + `x_l` (~1 GB/layer) | **recomputed in bwd, not stashed** |

Net: the two `x_message` HBM round-trips (~500 MB each) and the input/output-side Inductor
glue are gone; the mandatory GEMM-input writes, the `[B]` GEMMs, and the scatter stay. The
three cuBLAS GEMMs and the scatter are deliberately left as fusion barriers — we only
collapsed the memory-movement glue at the seams into the two `[T]` kernels.

## 4. The two fused ops

**Producer — `wigner_conv1_fused_op`** (`triton/fused_wigner.py`; kernels
`wigner_conv1_fused_{fwd,bwd}_kernel`):

- Inputs: `x_full [N,9,C]`, `edge_index [2,E]`, `wigner [E,9,9]`, per-layer `radial`, `C`.
- Per edge, all in registers: gather src/tgt → block-diagonal Wigner rotate → L→M permute
  → per-m radial scale → write the three GEMM-ready packed buffers
  `m0 [E, 3·2C]`, `m1 [E, 4·2C]`, `m2 [E, 2·2C]`.
- `x_message` never materializes. cuBLAS then runs conv1's GEMMs on the packed buffers via
  `SO2_Conv1_WithRadialBlock.gemms_from_packed(...)` (mirrors `forward()` from `fc_m0`
  onward; the block `_w_block` GEMMs are unchanged).

**Consumer — `wigner_inv_conv2_fused_op`** (`triton/fused_wigner.py`):

- `SO2_Conv2_InternalBlock.gemms_to_buffers(x_message)` returns the raw block-GEMM outputs
  `g0 [E,3C]`, `g1 [E,4C]`, `g2 [E,2C]` *without* the `torch.cat` repack.
- The fused kernel absorbs the M→L unpack + inverse-Wigner rotation from `g0/g1/g2` →
  `x_rotated [E,9,C]`. The `index_add` scatter stays **outside** the op (visible to
  torch.compile), mirroring the existing `PermuteWignerInvEdgeToNode` design.

## 5. Autograd (conservative forces)

Forces are `-dE/dpos`, so gradients flow through both ops — they must be differentiable.
Each is a `torch.autograd.Function` that allocates outputs (visible to torch.compile) and
launches the fwd/bwd Triton kernels via `torch.library.triton_op` + `wrap_triton`
(`mutates_args`), backed by analytic backward Triton kernels:

- Producer bwd: grads w.r.t. node features (via the gather transpose), `wigner`, `radial`.
- Consumer bwd: grads w.r.t. `g0/g1/g2` and `wigner_inv`.
- **recompute-x_edge:** the original Wigner op stashes `x_edge [E,9,2C]` (~500 MB/layer)
  purely for the `grad_wigner` outer product. Because we are memory-bound, we do not stash
  it — the backward re-gathers from `x_full` (~9 MB) and re-rotates. Trading a cheap
  recompute for a large DRAM write+read is a net win here and cuts peak memory.

Validated with `torch.autograd.gradcheck` (float64, `fast_mode=True`) in
`tests/core/models/uma/uma_fast/test_fused_edgewise.py`.

## 6. Why cuBLAS is kept for the GEMMs

The SO2 conv GEMMs are small-K block-diagonal (K≈256–768, M=E). cuBLAS is the best
available: max-autotune's Triton templates fall back to `aten::mm`, and GEMM batching
(grouped_mm / padded bmm) is 2–3× slower (padding waste; the low occupancy is a tf32
characteristic, not fixable). Fusing the matmul into Triton would 10–30× regress it. So the
fusions only touch the Wigner/pack/unpack/rotate/scatter glue; the GEMMs are untouched.

## 7. Correctness

- **Equivariance preserved:** identical block-diagonal Wigner math (L0 1×1, L1 3×3, L2 5×5)
  and identical weights — only memory ops were reordered and the radial scale fused.
- **Bit-exact fp32:** `torch.equal` (0.0 diff, fp32 and fp64) on all fwd/bwd outputs of the
  four kernels. E2E energy diff 0.0, forces ~5e-7 (`index_add` ordering).
- **tf32 note:** e2e tf32 lands at the run-to-run nondeterminism floor (dE ~6.5e-8
  relative), because pulling conv2's GEMMs out of `SO2_Conv2_InternalBlock.forward` lets
  Inductor reselect a cuBLAS algo — not a logic change (the kernels are provably bit-exact).

## 8. Integration & gating

- `UMASFastGPUBackend.supports_fused_edgewise = True` (base `ExecutionBackend` = `False`);
  new backend methods `fused_node_to_edge_conv1_pack` / `fused_conv2_inv_edge_to_node`.
- `Edgewise.forward_chunk` dispatches to `_forward_chunk_fused` iff
  `getattr(self.backend, "supports_fused_edgewise", False)`; the `else` branch is
  byte-for-byte the original.
- `so2_layers.py` gains `gemms_from_packed` / `gemms_to_buffers` (additive; `forward()`
  untouched).
- **Only `umas_fast_gpu` inference is affected.** `general`, `umas_fast_pytorch`, and all
  training paths are unchanged. Constraints: lmax=mmax=2, C%128==0 (already asserted by
  `umas_fast_gpu`).

## 9. Results (uma-s-1p2, natoms=1000, fwd+bwd)

| | DRAM | wall | peak mem |
|---|---|---|---|
| baseline | 82.9 GB | 54.7 ms | 10.7 GB |
| producer only | 61.9 GB | 43.5 ms | — |
| **producer + inv** | **55.9 GB (1.48×)** | **40.2 ms (1.36×)** | **6.5 GB (−39%)** |

QPS by system size: **1000 → 1.32×, 4000 → 1.36×**; **2 / 100 atoms are flat** (launch-bound,
not memory-bound — fusion does not help there; candidate for a size-gate).

## 10. Kernel-implementation notes

Triton, `GRID_E_STRIDE=2048` grid-stride loop, `num_warps=1`, `BLOCK_C=128`. Block-diagonal
FMA is done explicitly (not `tl.dot`) to exploit the lmax=2 sparsity (35/81 nonzero).
Kernels are compacted with `tl.static_range` + small `@jit` helpers (`_wig_rot9`,
`_wig_rotT9`, `_wig_dw_store*`). Note Triton's `@jit` AST rejects list comprehensions,
`.append`, and tuple-indexing by a derived index.

## 11. Headroom left

- **bf16** is the next lever (~1.5–1.8× on top, toward the ~18–28 ms roofline): halves the
  byte-heavy families and moves the 23 GB of GEMMs onto bf16 tensor cores. Gated on an
  energy-drift + force-MAE conservation check; ship off by default.
- **Size-gate** the fused path (edge/atom threshold) so small systems keep the original
  path.
