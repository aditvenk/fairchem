"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import distributed as dist
from torch.distributed import _symmetric_memory as symm_mem
from torch.profiler import record_function

if TYPE_CHECKING:
    from fairchem.core.common.parallelism.graph_parallel_a2a import GPContext

# Peer copies are issued round-robin over these so several links are in
# flight at once; a single stream serialises them behind one another.
_MAX_STREAMS = 8
_streams: list[torch.cuda.Stream] = []


def _get_streams() -> list[torch.cuda.Stream]:
    global _streams
    if not _streams:
        _streams = [torch.cuda.Stream() for _ in range(_MAX_STREAMS)]
    return _streams


@dataclass
class SymmHaloPlan:
    """
    Peer write offsets and buffer capacities for a symmetric-memory exchange.

    Derived from the global count matrix, where entry ``(r, p)`` is the number
    of nodes rank r sends to rank p. Knowing the whole matrix lets a rank
    address its own slot inside every peer's buffer with no per-exchange
    metadata traffic.

    Attributes:
        world_size: Number of GP ranks.
        send_counts: Nodes this rank sends to each peer.
        recv_counts: Nodes this rank receives from each peer.
        fwd_write_offsets: Where this rank's payload lands in each peer's
            receive buffer, in nodes.
        bwd_write_offsets: Where this rank's gradient lands in each peer's
            send-gradient buffer, in nodes.
        total_send: Total nodes sent.
        total_recv: Total nodes received.
        fwd_capacity: Receive-buffer capacity, uniform across ranks.
        bwd_capacity: Send-gradient buffer capacity, uniform across ranks.
        plan_id: Registry key used to reach this plan from the custom ops.
    """

    world_size: int
    send_counts: list[int]
    recv_counts: list[int]
    fwd_write_offsets: list[int]
    bwd_write_offsets: list[int]
    total_send: int
    total_recv: int
    fwd_capacity: int
    bwd_capacity: int
    # Assigned by register_plan when the plan is first published to the ops.
    plan_id: int = -1


@torch.compiler.disable
def build_symm_plan(
    send_counts: torch.Tensor,
    group: dist.ProcessGroup,
) -> SymmHaloPlan:
    """
    Gather the global count matrix and derive every peer write offset.

    Costs one all_gather of ``world_size`` ints per rank. The plan depends only
    on the partition, so it can be reused for as long as ``GPContext`` is.

    Args:
        send_counts: Nodes this rank sends to each peer, shape (world_size,).
        group: GP process group.

    Returns:
        The exchange plan for this rank.
    """
    rank = dist.get_rank(group)
    world = dist.get_world_size(group)

    # Gloo requires a flat output buffer here, so gather flat and reshape.
    flat = torch.empty(world * world, dtype=torch.long, device=send_counts.device)
    dist.all_gather_into_tensor(flat, send_counts.contiguous().long(), group=group)
    m = flat.view(world, world).cpu()

    return SymmHaloPlan(
        world_size=world,
        send_counts=m[rank].tolist(),
        recv_counts=m[:, rank].tolist(),
        # A rank's receive buffer is ordered by source rank, matching the
        # layout all_to_all_single produces, so edge_index_local still applies.
        fwd_write_offsets=[int(m[:rank, p].sum()) for p in range(world)],
        # Peer p orders its send staging by destination rank.
        bwd_write_offsets=[int(m[p, :rank].sum()) for p in range(world)],
        total_send=int(m[rank].sum()),
        total_recv=int(m[:, rank].sum()),
        # Symmetric allocations must be identical on every rank.
        fwd_capacity=int(m.sum(0).max()),
        bwd_capacity=int(m.sum(1).max()),
    )


@dataclass
class _PoolSlot:
    """
    A double-buffered symmetric allocation and which half was last used.
    """

    pairs: list
    capacity: int
    index: int


class _BufferPool:
    """
    Caches double-buffered symmetric allocations keyed by feature shape/dtype.

    Allocation plus rendezvous is a collective, so buffers are grown to a
    high-water mark and reused across layers and MD steps rather than
    reallocated per exchange.

    Two buffers are kept and alternated. That removes the need to barrier
    *before* writing: a peer racing ahead to exchange k+1 writes the other
    buffer while this rank still reads k. It cannot reach k+2, which would
    reuse this buffer, without first consuming this rank's k+1 contribution,
    and a rank that never exchanges with this one never writes here at all.
    """

    def __init__(self) -> None:
        # key -> ([(buffer, handle), (buffer, handle)], index last handed out)
        self._slots: dict[tuple, _PoolSlot] = {}

    def get(self, capacity, feat, dtype, device, group):
        key = (tuple(feat), dtype, group)
        slot = self._slots.get(key)
        if slot is None or slot.capacity < capacity:
            cap = max(capacity, 1)
            if slot is not None:
                cap = max(cap, int(slot.capacity * 1.5))
            # Every rank must agree on the size or rendezvous deadlocks.
            cap_t = torch.tensor([cap], device=device)
            dist.all_reduce(cap_t, op=dist.ReduceOp.MAX, group=group)
            cap = int(cap_t.item())
            pairs = []
            for _ in range(2):
                buf = symm_mem.empty(cap, *feat, dtype=dtype, device=device)
                pairs.append((buf, symm_mem.rendezvous(buf, group)))
            slot = _PoolSlot(pairs=pairs, capacity=cap, index=0)
            self._slots[key] = slot
        slot.index ^= 1
        buf, hdl = slot.pairs[slot.index]
        return buf, hdl, slot.index


_POOL = _BufferPool()


def _exchange(src, offsets, write_counts, read_counts, capacity, group):
    """
    Bulk-copy each per-peer slice of ``src`` into that peer's buffer.

    ``src`` must already be packed in peer order. Completion is signalled per
    peer rather than with a group barrier: under spatial partitioning a rank
    exchanges with a bounded number of spatial neighbours (~18-21) regardless
    of world size, so waiting on all P ranks would make sync O(P) for no
    reason.

    Args:
        src: Payload packed in destination-rank order.
        offsets: Node offset of this rank's slot inside each peer's buffer.
        write_counts: Nodes written to each peer.
        read_counts: Nodes each peer writes here.
        capacity: Required buffer capacity in nodes.
        group: GP process group.

    Returns:
        This rank's buffer, holding every peer's contribution.
    """
    feat, dtype, device = tuple(src.shape[1:]), src.dtype, src.device
    buf, hdl, channel = _POOL.get(capacity, feat, dtype, device, group)

    stride = 1
    for d in feat:
        stride *= d

    # One barrier launch syncs all P; per-peer signals cost a launch each but
    # only wait on real neighbours. Signals win once neighbours are sparse,
    # which under Morton partitioning happens from roughly P >= 32 (peer count
    # saturates near 20 regardless of P). Below that a barrier is cheaper.
    n_readers = sum(1 for n in read_counts if n > 0)
    use_signals = 2 * n_readers < len(read_counts)

    streams = _get_streams()
    cur = torch.cuda.current_stream()
    src_off = 0
    for p, n in enumerate(write_counts):
        if n == 0:
            continue
        chunk = src[src_off : src_off + n]
        src_off += n
        s = streams[p % len(streams)]
        s.wait_stream(cur)
        with torch.cuda.stream(s):
            dst = hdl.get_buffer(p, (n, *feat), dtype, offsets[p] * stride)
            dst.copy_(chunk)
            chunk.record_stream(s)
            if use_signals:
                # Same stream as the copy, so the signal cannot outrun the data.
                hdl.put_signal(p, channel=channel)

    if use_signals:
        # Waiting on the current stream is safe: the side streams already
        # captured their dependency on it, so this rank's own puts still drain.
        for p, n in enumerate(read_counts):
            if n > 0:
                hdl.wait_signal(p, channel=channel)
        for s in streams:
            cur.wait_stream(s)
    else:
        for s in streams:
            cur.wait_stream(s)
        # Double buffering already removed the matching pre-barrier.
        hdl.barrier(channel=channel)
    return buf


# Custom ops may only take tensors and primitives, so the plan is reached by
# integer id -- the same device functional collectives use to pass a process
# group as ``group_name``. The id is a fixed slot rather than a running counter:
# Dynamo specialises on it, so a changing id would recompile every MD step.
# One slot suffices because predict() is synchronous, so a step's backward
# always runs against the plan its own forward registered.
_PLAN_SLOTS: dict[int, SymmHaloPlan] = {}
_DEFAULT_SLOT = 0


def register_plan(plan: SymmHaloPlan, slot: int = _DEFAULT_SLOT) -> int:
    """
    Publish a plan into a fixed slot and return that slot's id.
    """
    _PLAN_SLOTS[slot] = plan
    return slot


@torch.library.custom_op("fairchem::symm_halo_collect", mutates_args=())
def symm_halo_collect(
    x_local: torch.Tensor, send_indices: torch.Tensor, plan_id: int
) -> torch.Tensor:
    """
    Collect remote node embeddings, ordered by source rank.

    Opaque to Dynamo but traceable, so unlike a ``@torch.compiler.disable``d
    autograd.Function it does not split the surrounding compiled region.
    """
    from fairchem.core.common import gp_utils

    plan = _PLAN_SLOTS[int(plan_id)]
    if send_indices.numel() > 0:
        x_send = x_local[send_indices].contiguous()
    else:
        x_send = x_local.new_empty(0, *x_local.shape[1:])
    buf = _exchange(
        x_send,
        plan.fwd_write_offsets,
        plan.send_counts,
        plan.recv_counts,
        plan.fwd_capacity,
        gp_utils.get_gp_group(),
    )
    # The pooled buffer is reused two exchanges later, and returning an alias of
    # it would also violate the custom-op contract, so copy out.
    return buf[: plan.total_recv].clone()


@symm_halo_collect.register_fake
def _(x_local, send_indices, plan_id):
    # The halo size shifts as atoms move, so report it as unbacked rather than
    # baking in this step's value and guarding a recompile on it.
    n_recv = torch.library.get_ctx().new_dynamic_size()
    return x_local.new_empty(n_recv, *x_local.shape[1:])


@torch.library.custom_op("fairchem::symm_halo_collect_backward", mutates_args=())
def symm_halo_collect_backward(
    grad_recv: torch.Tensor,
    send_indices: torch.Tensor,
    plan_id: int,
    local_size: int,
) -> torch.Tensor:
    """
    Reverse the halo exchange and scatter gradients to their source nodes.
    """
    from fairchem.core.common import gp_utils

    plan = _PLAN_SLOTS[int(plan_id)]
    # Roles reverse: gradient flows back to whoever sent the embeddings.
    buf = _exchange(
        grad_recv.contiguous(),
        plan.bwd_write_offsets,
        plan.recv_counts,
        plan.send_counts,
        plan.bwd_capacity,
        gp_utils.get_gp_group(),
    )
    grad_local = torch.zeros(
        local_size, *grad_recv.shape[1:], device=grad_recv.device, dtype=grad_recv.dtype
    )
    if plan.total_send > 0:
        grad_local.index_add_(0, send_indices, buf[: plan.total_send])
    return grad_local


@symm_halo_collect_backward.register_fake
def _(grad_recv, send_indices, plan_id, local_size):
    return grad_recv.new_empty(local_size, *grad_recv.shape[1:])


def _collect_setup_context(ctx, inputs, output):
    x_local, send_indices, plan_id = inputs
    ctx.send_indices = send_indices
    ctx.plan_id = plan_id
    ctx.local_size = x_local.shape[0]


def _collect_backward(ctx, grad_recv):
    grad_local = torch.ops.fairchem.symm_halo_collect_backward(
        grad_recv, ctx.send_indices, ctx.plan_id, ctx.local_size
    )
    return grad_local, None, None


torch.library.register_autograd(
    "fairchem::symm_halo_collect",
    _collect_backward,
    setup_context=_collect_setup_context,
)


@torch.compiler.disable
def prepare_symm_plan(gp_ctx: GPContext) -> None:
    """
    Build and publish this context's exchange plan.

    Call once per GPContext, next to where the context itself is built. Doing
    it lazily inside the layer loop would break the compiled graph on every
    layer, which is the thing the custom op exists to avoid.
    """
    from fairchem.core.common import gp_utils

    with record_function("symm_build_plan"):
        plan = build_symm_plan(gp_ctx.send_counts, gp_utils.get_gp_group())
        plan.plan_id = register_plan(plan)
    gp_ctx.symm_plan = plan


def symm_all_to_all_collect(x_local: torch.Tensor, gp_ctx: GPContext):
    """
    Collect remote node embeddings via symmetric memory.

    Drop-in replacement for :func:`all_to_all_collect`; the plan is built once
    per ``GPContext`` and cached on it.

    Args:
        x_local: Local node embeddings, shape (local_atoms, *features).
        gp_ctx: Graph parallel context.

    Returns:
        Remote node embeddings, shape (total_recv, *features).
    """
    if gp_ctx.symm_plan is None:
        prepare_symm_plan(gp_ctx)
    return torch.ops.fairchem.symm_halo_collect(
        x_local, gp_ctx.send_indices, gp_ctx.symm_plan.plan_id
    )
