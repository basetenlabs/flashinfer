# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest
import torch

from flashinfer.fp4_quantization import shuffle_matrix_a, shuffle_matrix_sf_a
from flashinfer.fused_moe import (
    trtllm_mxfp8_routed_gemm,
    trtllm_mxfp8_routed_gemm_configs,
    trtllm_mxfp8_routed_gemm_route,
)
from flashinfer.utils import get_compute_capability


@pytest.mark.parametrize("tokens", [1, 24, 257])
@torch.inference_mode()
def test_routed_gemm_quantized_reference_and_graph_updates(tokens):
    if not torch.cuda.is_available() or get_compute_capability(
        torch.device("cuda")
    ) not in ((10, 0), (10, 3)):
        pytest.skip("Requires Blackwell datacenter GPUs")
    torch.manual_seed(71)
    device = "cuda"
    experts, n, k, top_k, tile = 128, 256, 256, 16, 16
    x = (torch.randint(-8, 9, (tokens, k), device=device).float() / 8).to(
        torch.float8_e4m3fn
    )
    x_scale = torch.randint(
        125, 130, (tokens, k // 32), device=device, dtype=torch.uint8
    )
    raw = torch.randint(0, 256, (experts, n, k // 2), device=device, dtype=torch.uint8)
    raw_scale = torch.randint(
        125, 130, (experts, n, k // 32), device=device, dtype=torch.uint8
    )
    weight = shuffle_matrix_a(raw.reshape(experts * n, k // 2), 128).view_as(raw)
    weight_scale = shuffle_matrix_sf_a(
        raw_scale.reshape(experts * n, k // 32), 128
    ).view_as(raw_scale)
    ids = (
        torch.arange(top_k, device=device, dtype=torch.int32)
        .expand(tokens, -1)
        .contiguous()
    )
    weights = torch.ones((tokens, top_k), device=device, dtype=torch.bfloat16)
    configs = trtllm_mxfp8_routed_gemm_configs(
        x,
        num_tokens=tokens,
        num_experts=experts,
        top_k=top_k,
        n=n,
        k=k,
        tile_tokens=tile,
    )
    assert configs
    config = next((c for c in configs if c.tactic == 2827), configs[0])
    pairs = tokens * top_k
    max_ctas = min(experts, pairs) + max(0, pairs - experts) // tile

    def empty_int(count):
        return torch.empty(count, dtype=torch.int32, device=device)

    slot = SimpleNamespace(
        total_num_padded_tokens=empty_int(1),
        expanded_idx_to_permuted_idx=empty_int(pairs),
        inverse=empty_int(config.output_rows + 1),
        histogram=empty_int(max(2 * experts, 512)),
        expert_counts=empty_int(experts),
        cta_idx_xy_to_batch_idx=empty_int(max_ctas),
        cta_idx_xy_to_mn_limit=empty_int(max_ctas),
        num_non_exiting_ctas=empty_int(1),
    )
    q = torch.zeros((config.activation_rows, k), device=device, dtype=torch.uint8)
    sf = torch.zeros(
        (config.activation_rows * (k // 32),), device=device, dtype=torch.uint8
    )
    out = torch.empty((config.output_rows, n), device=device, dtype=torch.bfloat16)
    workspace = torch.empty(
        (max(1, config.workspace_bytes),), device=device, dtype=torch.uint8
    )
    args = [
        q,
        sf,
        weight,
        weight_scale,
        out,
        workspace,
        slot.total_num_padded_tokens,
        slot.cta_idx_xy_to_batch_idx,
        slot.cta_idx_xy_to_mn_limit,
        slot.num_non_exiting_ctas,
    ]
    options = dict(
        num_tokens=tokens, top_k=top_k, tile_tokens=tile, tactic=config.tactic
    )

    def prepare():
        trtllm_mxfp8_routed_gemm_route(
            ids,
            weights,
            slot.histogram,
            slot.total_num_padded_tokens,
            slot.expanded_idx_to_permuted_idx,
            slot.inverse,
            slot.expert_counts,
            slot.cta_idx_xy_to_batch_idx,
            slot.cta_idx_xy_to_mn_limit,
            slot.num_non_exiting_ctas,
            num_experts=experts,
            tile_tokens=tile,
        )
        rows = slot.expanded_idx_to_permuted_idx.long()
        q[rows] = x.view(torch.uint8).repeat_interleave(top_k, dim=0)
        row = rows[:, None]
        column = torch.arange(k // 32, device=device)[None, :]
        if config.scale_layout == 1:
            offset = (row // 8) * (k // 128) * 32 + (column // 4) * 32
            offset += (row % 8) * 4 + column % 4
        elif config.scale_layout == 3:
            offset = (row // 128) * (k // 128) * 512 + (column // 4) * 512
            offset += (row % 32) * 16 + ((row % 128) // 32) * 4 + column % 4
        else:
            offset = row * (k // 32) + column
        sf[offset] = x_scale.repeat_interleave(top_k, dim=0)

    def reference():
        lookup = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=device)
        nibble = torch.stack((raw & 15, raw >> 4), dim=-1).flatten(-2)
        dequant = lookup[(nibble & 7).long()] * torch.where(nibble < 8, 1, -1)
        dequant *= torch.exp2(raw_scale.float() - 127).repeat_interleave(32, dim=-1)
        inputs = x.float() * torch.exp2(x_scale.float() - 127).repeat_interleave(
            32, dim=-1
        )
        result = torch.empty((tokens, top_k, n), device=device, dtype=torch.bfloat16)
        for expert in ids.unique().tolist():
            token, route = torch.where(ids == expert)
            # Dyadic inputs keep the FP32 sum exact for these bounded shapes.
            result[token, route] = (
                inputs[token].double() @ dequant[expert].double().T
            ).bfloat16()
        return result.flatten(0, 1)

    prepare()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        trtllm_mxfp8_routed_gemm(*args, **options)
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(
        out[slot.expanded_idx_to_permuted_idx.long()], reference(), rtol=0, atol=0
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepare()
        trtllm_mxfp8_routed_gemm(*args, **options)
    raw.bitwise_xor_(9)
    weight.copy_(shuffle_matrix_a(raw.reshape(experts * n, k // 2), 128).view_as(raw))
    ids.copy_(
        (ids + torch.arange(tokens, device=device, dtype=torch.int32)[:, None])
        % experts
    )
    x_scale.add_(1)
    graph.replay()
    torch.testing.assert_close(
        out[slot.expanded_idx_to_permuted_idx.long()], reference(), rtol=0, atol=0
    )

    for index, bad in ((0, q[:-1]), (1, sf[:-1]), (4, out[:-1]), (4, out.float())):
        invalid = args.copy()
        invalid[index] = bad
        with pytest.raises(Exception, match="Check failed"):
            trtllm_mxfp8_routed_gemm(*invalid, **options)
    with pytest.raises(Exception, match="tactic >= 0"):
        trtllm_mxfp8_routed_gemm(*args, **(options | {"tactic": -1}))
