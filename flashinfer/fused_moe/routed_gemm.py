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

"""Routed MXFP4/MXFP8 GEMMs with caller-owned intermediates."""

import functools
from types import SimpleNamespace
from typing import NamedTuple

import torch

from ..api_logging import flashinfer_api
from ..jit import setup_cubin_loader
from ..jit.fused_moe import gen_trtllm_gen_fused_moe_sm100_module
from ..trace.templates.routed_gemm import trtllm_mxfp8_routed_gemm_trace
from ..utils import register_custom_op, register_fake_op


class TrtllmMxfp8RoutedGemmConfig(NamedTuple):
    """Native tactic and physical buffer requirements for one problem shape."""

    tactic: int
    scale_layout: int
    workspace_bytes: int
    activation_rows: int
    output_rows: int


@functools.cache
def _get_module():
    spec = gen_trtllm_gen_fused_moe_sm100_module()
    native = spec.try_load()
    if native is None or not hasattr(native, "trtllm_mxfp8_routed_gemm_route"):
        # An older AOT provider can lack the split-GEMM entry points.
        spec.build(need_lock=True)
        native = spec.load(spec.jit_library_path)
        setup_cubin_loader(str(spec.jit_library_path))
    else:
        for path in spec.get_library_paths():
            setup_cubin_loader(str(path))

    @register_custom_op(
        "flashinfer::trtllm_mxfp8_routed_gemm",
        mutates_args=["out", "workspace"],
    )
    def run(
        input: torch.Tensor,
        input_scale: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out: torch.Tensor,
        workspace: torch.Tensor,
        padded_count: torch.Tensor,
        cta_expert: torch.Tensor,
        cta_limit: torch.Tensor,
        num_ctas: torch.Tensor,
        num_tokens: int,
        top_k: int,
        tile_tokens: int,
        tactic: int,
    ) -> None:
        native.trtllm_mxfp8_routed_gemm(
            input,
            input_scale,
            weight,
            weight_scale,
            out,
            workspace,
            padded_count,
            cta_expert,
            cta_limit,
            num_ctas,
            num_tokens,
            top_k,
            tile_tokens,
            tactic,
        )

    @register_fake_op("flashinfer::trtllm_mxfp8_routed_gemm")
    def fake_run(
        input: torch.Tensor,
        input_scale: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out: torch.Tensor,
        workspace: torch.Tensor,
        padded_count: torch.Tensor,
        cta_expert: torch.Tensor,
        cta_limit: torch.Tensor,
        num_ctas: torch.Tensor,
        num_tokens: int,
        top_k: int,
        tile_tokens: int,
        tactic: int,
    ) -> None:
        pass

    @register_custom_op(
        "flashinfer::trtllm_mxfp8_routed_gemm_route",
        mutates_args=[
            "histogram",
            "padded_count",
            "indices",
            "inverse",
            "expert_counts",
            "cta_expert",
            "cta_limit",
            "num_ctas",
        ],
    )
    def route(
        ids: torch.Tensor,
        weights: torch.Tensor,
        histogram: torch.Tensor,
        padded_count: torch.Tensor,
        indices: torch.Tensor,
        inverse: torch.Tensor,
        expert_counts: torch.Tensor,
        cta_expert: torch.Tensor,
        cta_limit: torch.Tensor,
        num_ctas: torch.Tensor,
        num_experts: int,
        tile_tokens: int,
    ) -> None:
        native.trtllm_mxfp8_routed_gemm_route(
            ids,
            weights,
            histogram,
            padded_count,
            indices,
            inverse,
            expert_counts,
            cta_expert,
            cta_limit,
            num_ctas,
            num_experts,
            tile_tokens,
        )

    @register_fake_op("flashinfer::trtllm_mxfp8_routed_gemm_route")
    def fake_route(
        ids: torch.Tensor,
        weights: torch.Tensor,
        histogram: torch.Tensor,
        padded_count: torch.Tensor,
        indices: torch.Tensor,
        inverse: torch.Tensor,
        expert_counts: torch.Tensor,
        cta_expert: torch.Tensor,
        cta_limit: torch.Tensor,
        num_ctas: torch.Tensor,
        num_experts: int,
        tile_tokens: int,
    ) -> None:
        pass

    return SimpleNamespace(
        configs=native.trtllm_mxfp8_routed_gemm_configs, run=run, route=route
    )


def trtllm_mxfp8_routed_gemm_route(
    ids: torch.Tensor,
    weights: torch.Tensor,
    histogram: torch.Tensor,
    padded_count: torch.Tensor,
    indices: torch.Tensor,
    inverse: torch.Tensor,
    expert_counts: torch.Tensor,
    cta_expert: torch.Tensor,
    cta_limit: torch.Tensor,
    num_ctas: torch.Tensor,
    *,
    num_experts: int,
    tile_tokens: int,
) -> None:
    """Populate caller-owned routing buffers for the split GEMMs.

    ``ids`` is int32 [T, top_k] and ``weights`` is BF16 with the same shape.
    IDs must be in [-1, E); -1 denotes padding. Remaining tensors are contiguous int32:
    histogram [max(2*E, 512)], padded_count [1], indices [T*top_k], inverse
    [output_rows+1], expert_counts [E], cta_expert/cta_limit [max_ctas],
    num_ctas [1]. ``max_ctas = min(E, T*top_k) + max(0, T*top_k-E)//tile_tokens``.
    ``output_rows`` comes from :func:`trtllm_mxfp8_routed_gemm_configs`.
    All buffers share a CUDA device. Routing and GEMMs must use the same tile.
    """
    _get_module().route(
        ids,
        weights,
        histogram,
        padded_count,
        indices,
        inverse,
        expert_counts,
        cta_expert,
        cta_limit,
        num_ctas,
        num_experts,
        tile_tokens,
    )


def trtllm_mxfp8_routed_gemm_configs(
    device_anchor: torch.Tensor,
    *,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    n: int,
    k: int,
    tile_tokens: int,
) -> list[TrtllmMxfp8RoutedGemmConfig]:
    """Query valid tactics and allocation requirements before CUDA graph capture.

    ``device_anchor`` selects the CUDA device; its values are not read.
    ``num_tokens`` is the logical token count before routing, while returned
    row counts include routing padding and the native TMA storage minimum.
    ``scale_layout`` identifies the tactic's activation-scale layout; input
    scale bytes must be prepared in that layout. N and K must be divisible
    by 128, and ``tile_tokens`` must be 8, 16, 32 or 64.
    """
    return [
        TrtllmMxfp8RoutedGemmConfig(*(int(value) for value in row))
        for row in _get_module().configs(
            device_anchor, num_tokens, num_experts, top_k, n, k, tile_tokens
        )
    ]


@flashinfer_api(trace=trtllm_mxfp8_routed_gemm_trace)
def trtllm_mxfp8_routed_gemm(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
    workspace: torch.Tensor,
    padded_count: torch.Tensor,
    cta_expert: torch.Tensor,
    cta_limit: torch.Tensor,
    num_ctas: torch.Tensor,
    *,
    num_tokens: int,
    top_k: int,
    tile_tokens: int,
    tactic: int,
) -> None:
    """Write one routed MXFP4-weight, MXFP8-activation GEMM to BF16 storage.

    Routing, activation quantization, nonlinearities and output finalization
    remain with the caller. This permits inserting BF16 LoRA between GEMMs.

    ``input`` is contiguous uint8 MXFP8 storage [activation_rows, K].
    ``input_scale`` contains block-32 E8M0 bytes in the selected tactic's
    scale layout. ``weight`` is uint8 [E, N, K // 2] in TRT's shuffled MXFP4
    layout, with shuffled block-32 E8M0 ``weight_scale`` bytes. ``out`` is
    contiguous BF16 [output_rows, N]; ``workspace`` is contiguous uint8.
    Obtain the required storage sizes with ``trtllm_mxfp8_routed_gemm_configs``.

    The final four tensors are contiguous int32 routing metadata: padded row
    count, CTA-to-expert mapping, CTA row limit, and live CTA count. They must
    describe the same logical batch and token tile as the quantized inputs.
    Only live routed output rows are defined. The operation allocates no
    tensors and uses the caller's current CUDA stream.
    """
    _get_module().run(
        input,
        input_scale,
        weight,
        weight_scale,
        out,
        workspace,
        padded_count,
        cta_expert,
        cta_limit,
        num_ctas,
        num_tokens,
        top_k,
        tile_tokens,
        tactic,
    )
