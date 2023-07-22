import os
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from llm_mixed_q.models import get_model_cls, get_config_cls
from llm_mixed_q.models.llama_quantized import profile_llama_quantized
from llm_mixed_q.models.bert_quantized import profile_bert_quantized
from llm_mixed_q.tools import set_logging_verbosity

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"


def test_llama_profiler():
    llama_quant_config = {
        "default": {
            "bias_block_size": [1, 16],
            "bias_exponent_bias": None,
            "bias_exponent_width": 8,
            "bias_width": 5,
            "bypass": False,
            "data_in_block_size": [1, 16],
            "data_in_exponent_bias": None,
            "data_in_exponent_width": 8,
            "data_in_width": 6,
            "is_qat": False,
            "name": "block_fp",
            "weight_block_size": [1, 16],
            "weight_exponent_bias": None,
            "weight_exponent_width": 8,
            "weight_width": 6,
        }
    }
    config = get_config_cls("llama").from_pretrained(
        "Cheng98/llama-160m",
        # quant_config="./llama.toml",
        quant_config=llama_quant_config,
    )

    profile = profile_llama_quantized(config, seq_len=128)
    print(f"profile: {profile}")
    print(f"avg param bitwidth: {profile['param_bits'] / profile['num_params']}")
    print(f"avg act bitwidth: {profile['act_bits'] / profile['num_acts']}")


def test_bert_profiler():
    bert_quant_config = {
        "default": {
            "bias_block_size": [1, 16],
            "bias_exponent_bias": None,
            "bias_exponent_width": 8,
            "bias_width": 5,
            "bypass": False,
            "data_in_block_size": [1, 16],
            "data_in_exponent_bias": None,
            "data_in_exponent_width": 8,
            "data_in_width": 6,
            "is_qat": False,
            "name": "block_fp",
            "weight_block_size": [1, 16],
            "weight_exponent_bias": None,
            "weight_exponent_width": 8,
            "weight_width": 6,
        }
    }
    config = get_config_cls("bert").from_pretrained(
        "bert-base-uncased",
        # quant_config="./bert.toml",
        quant_config=bert_quant_config,
    )

    profile = profile_bert_quantized(config, seq_len=128)
    print(f"profile: {profile}")
    print(f"avg param bitwidth: {profile['param_bits'] / profile['num_params']}")
    print(f"avg act bitwidth: {profile['act_bits'] / profile['num_acts']}")


if __name__ == "__main__":
    set_logging_verbosity("debug")
    test_llama_profiler()
    test_bert_profiler()
