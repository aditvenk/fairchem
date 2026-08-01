"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from fairchem.core.common import gp_utils
from fairchem.core.common.parallelism.graph_parallel_a2a import AllToAllCollect
from fairchem.core.common.parallelism.symm_halo import (
    build_symm_plan,
    symm_all_to_all_collect,
)

SPH, CH = 9, 4


def _plan_only_worker(rank, world, port, out):
    """
    Offsets are pure arithmetic on the gathered count matrix, so this runs on
    gloo and needs no GPU.
    """
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)

    # counts[r][p] = nodes rank r sends to rank p; deterministic and uneven.
    counts = torch.tensor([[0, 3, 5], [2, 0, 1], [4, 6, 0]], dtype=torch.long)
    plan = build_symm_plan(counts[rank], dist.group.WORLD)

    out[rank] = {
        "send": plan.send_counts,
        "recv": plan.recv_counts,
        "fwd": plan.fwd_write_offsets,
        "bwd": plan.bwd_write_offsets,
        "total_send": plan.total_send,
        "total_recv": plan.total_recv,
        "fwd_cap": plan.fwd_capacity,
        "bwd_cap": plan.bwd_capacity,
    }
    dist.destroy_process_group()


def test_symm_plan_offsets_are_consistent():
    """
    Every rank's write offset into a peer must equal that peer's own receive
    offset for that source, or payloads land on top of one another.
    """
    world, port = 3, 29731
    mgr = mp.Manager()
    out = mgr.dict()
    mp.spawn(_plan_only_worker, args=(world, port, out), nprocs=world, join=True)

    counts = [[0, 3, 5], [2, 0, 1], [4, 6, 0]]
    for r in range(world):
        p = out[r]
        assert p["send"] == counts[r]
        assert p["recv"] == [counts[s][r] for s in range(world)]
        assert p["total_send"] == sum(counts[r])
        assert p["total_recv"] == sum(counts[s][r] for s in range(world))
        # Uniform capacity is required by symmetric allocation.
        assert p["fwd_cap"] == out[0]["fwd_cap"]
        assert p["bwd_cap"] == out[0]["bwd_cap"]
        assert p["fwd_cap"] >= p["total_recv"]
        assert p["bwd_cap"] >= p["total_send"]

    for src in range(world):
        for dst in range(world):
            # Where src writes into dst == where dst expects src's block.
            expected = sum(counts[s][dst] for s in range(src))
            assert out[src]["fwd"][dst] == expected
            # Backward mirrors it through dst's send ordering.
            assert out[src]["bwd"][dst] == sum(counts[dst][:src])


def _exchange_worker(rank, world, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    # symm_all_to_all_collect resolves the group through gp_utils.
    gp_utils.setup_graph_parallel_groups(world, "nccl")
    dev = torch.device("cuda", rank)
    group = gp_utils.get_gp_group()

    torch.manual_seed(rank)
    n_local = 32
    # Dense exchanges every peer, so _exchange syncs with a barrier; the ring
    # leaves one reader per rank, which is sparse enough to take the per-peer
    # signal path. Both must be covered.
    matrices = {
        "dense": [[0, 3, 5], [2, 0, 1], [4, 6, 0]],
        "ring": [[0, 4, 0], [0, 0, 4], [4, 0, 0]],
    }
    for label, rows in matrices.items():
        counts = torch.tensor(rows, dtype=torch.long)
        send_counts = counts[rank].to(dev)
        recv_counts = counts[:, rank].contiguous().to(dev)
        total_send, total_recv = int(send_counts.sum()), int(recv_counts.sum())

        x = torch.randn(n_local, SPH, CH, device=dev, requires_grad=True)
        send_idx = torch.arange(total_send, device=dev) % n_local
        gout = torch.randn(total_recv, SPH, CH, device=dev)

        class _Ctx:
            pass

        ctx = _Ctx()
        ctx.send_counts, ctx.send_indices, ctx.symm_plan = send_counts, send_idx, None

        ref = AllToAllCollect.apply(
            x,
            send_idx,
            send_counts,
            recv_counts,
            group,
            rank,
            world,
            send_counts.tolist(),
            recv_counts.tolist(),
            total_recv,
        )
        got = symm_all_to_all_collect(x, ctx)
        # Pure data movement: the forward must be bit-identical to NCCL's.
        assert torch.equal(ref, got), f"rank {rank} {label} forward mismatch"

        gref = torch.autograd.grad(ref, x, gout, retain_graph=True)[0]
        ggot = torch.autograd.grad(got, x, gout, retain_graph=True)[0]
        # send_idx has duplicates, so index_add_ accumulates with atomics and
        # bitwise equality is not guaranteed even for two identical runs.
        assert torch.allclose(
            gref, ggot, atol=1e-5, rtol=1e-5
        ), f"rank {rank} {label} backward mismatch"

    dist.destroy_process_group()


@pytest.mark.gpu()
@pytest.mark.skipif(torch.cuda.device_count() < 3, reason="needs 3 GPUs")
def test_symm_exchange_matches_nccl():
    mp.spawn(_exchange_worker, args=(3, 29732), nprocs=3, join=True)


def test_edge_perm_orders_local_sources_first():
    """
    The overlap split is a contiguous slice, so build_gp_context must place
    every local-source edge before every remote-source one and permute
    edge_index_local to match.
    """
    from fairchem.core.common.gp_utils import (
        GPMode,
        GPPartition,
        GPTransport,
        GraphParallelConfig,
        set_gp_config,
    )
    from fairchem.core.common.parallelism.graph_parallel_a2a import build_gp_context

    prev = gp_utils.get_gp_config()
    set_gp_config(
        GraphParallelConfig(
            group_size=2,
            mode=GPMode.ALL_TO_ALL,
            partition=GPPartition.SPATIAL,
            transport=GPTransport.SYMM_MEM,
            overlap=True,
        )
    )
    try:
        # Atoms 0-3 owned by rank 0, 4-7 by rank 1; mix local and remote sources.
        rank_assignments = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        edge_index = torch.tensor([[4, 0, 5, 1, 6], [0, 1, 2, 3, 0]])
        ctx = build_gp_context(
            edge_index=edge_index,
            rank_assignments=rank_assignments,
            rank=0,
            world_size=2,
            node_partition=torch.tensor([0, 1, 2, 3]),
        )

        assert ctx.edge_perm is not None
        n = ctx.num_local_edges
        srcs = ctx.edge_index_local[0]
        assert (srcs[:n] < ctx.total_local_atoms).all(), "local block has a remote src"
        assert (srcs[n:] >= ctx.total_local_atoms).all(), "remote block has a local src"
        # The permutation must be a bijection over the edges, losing none.
        assert sorted(ctx.edge_perm.tolist()) == list(range(edge_index.shape[1]))
    finally:
        set_gp_config(prev) if prev is not None else None
