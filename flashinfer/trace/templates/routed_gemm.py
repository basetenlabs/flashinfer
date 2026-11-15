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

from ..template import Const, Scalar, Tensor, TraceTemplate, Var

trtllm_mxfp8_routed_gemm_trace = TraceTemplate(
    op_type="moe_gemm",
    name_prefix="trtllm_mxfp8_routed_gemm",
    description="Routed MXFP4/MXFP8 expert GEMM with caller-owned BF16 output.",
    axes={
        "activation_rows": Var(),
        "output_rows": Var(),
        "num_experts": Const(abbrev="e"),
        "n": Const(abbrev="n"),
        "k": Const(abbrev="k"),
        "packed_k": Var(),
        "input_scale_bytes": Var(),
        "weight_scale_groups": Var(),
        "workspace_bytes": Var(),
        "max_ctas": Var(),
        "scalar_length": Const(abbrev=""),
    },
    inputs={
        "input": Tensor(["activation_rows", "k"]),
        "input_scale": Tensor(["input_scale_bytes"]),
        "weight": Tensor(["num_experts", "n", "packed_k"]),
        "weight_scale": Tensor(["num_experts", "n", "weight_scale_groups"]),
        "out": Tensor(["output_rows", "n"]),
        "workspace": Tensor(["workspace_bytes"]),
        "padded_count": Tensor(["scalar_length"]),
        "cta_expert": Tensor(["max_ctas"]),
        "cta_limit": Tensor(["max_ctas"]),
        "num_ctas": Tensor(["scalar_length"]),
        "num_tokens": Scalar("int32"),
        "top_k": Scalar("int32"),
        "tile_tokens": Scalar("int32"),
        "tactic": Scalar("int32"),
    },
    outputs={"out": Tensor(["output_rows", "n"], dtype_from="out")},
    constraints=[
        "k == packed_k * 2",
        "k == weight_scale_groups * 32",
        "scalar_length == 1",
    ],
    tags=["quantization:mxfp4", "quantization:mxfp8", "moe:trtllm"],
)
