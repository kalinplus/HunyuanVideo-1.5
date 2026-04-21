# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

import os
from typing import Optional
from dataclasses import dataclass, field

@dataclass
class InferState:
    enable_sageattn: bool = False  # whether to use SageAttention
    sage_blocks_range: Optional[range] = None  # block range to use SageAttention
    enable_torch_compile: bool = False  # whether to use torch compile

    enable_cache: bool = False  # whether to use cache
    cache_type: str = "deepcache" # cache type
    no_cache_block_id: Optional[range] = None # block ids to skip
    cache_start_step: int = 11 # start step to skip
    cache_end_step: int = 45 # end step to skip
    total_steps: int = 50 # total steps
    cache_step_interval: int = 4 # step interval to skip

    # taylorcache specific
    taylor_max_order: int = 2
    taylor_low_freqs_order: int = 2
    taylor_high_freqs_order: int = 2
    taylor_cutoff_ratio: float = 0.1

    use_fp8_gemm: bool = False  # whether to use fp8 gemm
    quant_type: str = "fp8-per-token-sgl"  # fp8 quantization type
    include_patterns: list = field(default_factory=lambda: ["double_blocks"])  # include patterns for fp8 gemm



__infer_state = None

def parse_range(value):
    if '-' in value:
        start, end = map(int, value.split('-'))
        return list(range(start, end + 1))
    else:
        return [int(x) for x in value.split(',')]

def initialize_infer_state(args):
    global __infer_state
    sage_blocks_range = parse_range(getattr(args, 'sage_blocks_range', '0-59'))
    no_cache_block_id = parse_range(getattr(args, 'no_cache_block_id', '53'))
    # Map CLI argument use_sageattn to internal enable_sageattn field
    use_sageattn = getattr(args, 'use_sageattn', False)


    
    # Parse include_patterns from args
    include_patterns = getattr(args, 'include_patterns', "double_blocks")
    if isinstance(include_patterns, str):
        # Split by comma and strip whitespace
        include_patterns = [p.strip() for p in include_patterns.split(',') if p.strip()]
    


    __infer_state = InferState(
        enable_sageattn = use_sageattn,
        sage_blocks_range = sage_blocks_range,
        enable_torch_compile = getattr(args, 'enable_torch_compile', False),

        # cache related
        enable_cache = getattr(args, 'enable_cache', False),
        cache_type = getattr(args, 'cache_type', 'deepcache'),
        no_cache_block_id = no_cache_block_id,
        cache_start_step = getattr(args, 'cache_start_step', 11),
        cache_end_step = getattr(args, 'cache_end_step', 45),
        total_steps = getattr(args, 'total_steps', 50),
        cache_step_interval = getattr(args, 'cache_step_interval', 4),

        # fp8 gemm related
        use_fp8_gemm = getattr(args, 'use_fp8_gemm', False),
        quant_type = getattr(args, 'quant_type', 'fp8-per-token-sgl'),
        include_patterns = include_patterns,

        # taylorcache specific (env var > CLI arg > default)
        taylor_max_order = int(os.getenv("TAYLOR_MAX_ORDER", str(getattr(args, 'taylor_max_order', 2)))),
        taylor_low_freqs_order = int(os.getenv("TAYLOR_LOW_FREQS_ORDER", str(getattr(args, 'taylor_low_freqs_order', 2)))),
        taylor_high_freqs_order = int(os.getenv("TAYLOR_HIGH_FREQS_ORDER", str(getattr(args, 'taylor_high_freqs_order', 2)))),
        taylor_cutoff_ratio = float(os.getenv("TAYLOR_CUTOFF_RATIO", str(getattr(args, 'taylor_cutoff_ratio', 0.1)))),
    )
    return __infer_state

def get_infer_state():
    return __infer_state
