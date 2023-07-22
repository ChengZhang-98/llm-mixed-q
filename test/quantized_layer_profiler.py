import os
import sys
from pathlib import Path

from accelerate import init_empty_weights

print(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent / "src"))
from llm_mixed_q.models import get_model_cls, get_config_cls
from llm_mixed_q.models.quantize.quantized_layer_profiler import profile_linear_layer, profile_matmul_layer

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"


def test_profiler():
    quant_config_int = {
        "linear": {
            "name": "integer",
            "weight_width": 8,
            "weight_frac_width": 7,
            "bias_width": 8,
            "bias_frac_width": 15,
            "data_in_width": 8,
            "data_in_frac_width": 4,
        },
        "matmul": {
            "name": "integer",
            "data_in_width": 8,
            "data_in_frac_width": 4,
            "weight_width": 8,
            "weight_frac_width": 7,
            "bias_width": 8,
            "bias_frac_width": 15,
        },
    }

    quant_config_block_fp = {
        "linear": {
            "name": "block_fp",
            "weight_width": 8,
            "weight_frac_width": 7,
            "weight_block_size": [1, 5],
            "weight_exponent_width": 4,
            "bias_width": 8,
            "bias_frac_width": 15,
            "bias_block_size": [5],
            "bias_exponent_width": 4,
            "data_in_width": 8,
            "data_in_frac_width": 4,
            "data_in_block_size": [1, 5],
            "data_in_exponent_width": 4,
        },
        "matmul": {
            "name": "block_fp",
            "weight_width": 8,
            "weight_frac_width": 7,
            "weight_block_size": [1, 5],
            "weight_exponent_width": 4,
            "data_in_width": 8,
            "data_in_frac_width": 4,
            "data_in_block_size": [1, 5],
            "data_in_exponent_width": 4,
        },
    }

    print(profile_linear_layer(quant_config_int["linear"], in_features=10, out_features=10, bias=True, batch_size=1))
    print(profile_matmul_layer(quant_config_int["matmul"], data_in_0_size=(1, 10), data_in_1_size=(10, 10)))
    print(profile_linear_layer(quant_config_block_fp["linear"], in_features=10, out_features=10, bias=True, batch_size=1))
    print(profile_matmul_layer(quant_config_block_fp["matmul"], data_in_0_size=(1, 10), data_in_1_size=(10, 10)))


if __name__ == "__main__":
    test_profiler()