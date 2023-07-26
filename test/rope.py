import os
import sys
from pathlib import Path

from accelerate import init_empty_weights

sys.path.append(str(Path(__file__).parent.parent / "src"))
from llm_mixed_q.utils import get_logger
from llm_mixed_q.models import get_model_cls, get_config_cls
from llm_mixed_q.models.quantize.quantized_layer_profiler import (
    profile_linear_layer,
    profile_matmul_layer,
)
from llm_mixed_q.models.llama_quantized.modeling_llama import LlamaRotaryEmbedding

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"


if __name__ == "__main__":
    logger = get_logger(__name__)
    import torch

    dim = 64

    arange = torch.arange(0, dim, 2).float()
    print("arange", arange)

    rope = LlamaRotaryEmbedding(dim)
    print("rope.cos_cached[.., 0, :]", rope.cos_cached[..., 0, :])
    breakpoint()
