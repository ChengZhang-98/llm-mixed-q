import os
import sys
from pathlib import Path
from pprint import pprint as pp
import random

from accelerate import init_empty_weights

print(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent / "src"))
from llm_mixed_q.models import get_model_cls, get_config_cls
from llm_mixed_q.models.llama_quantized.quant_config_llama import (
    parse_llama_quantized_config,
)
from llm_mixed_q.models.quantize.quant_config_parser import parse_node_config

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"


def test_node_parser():
    default_config = {
        "default": {
            "name": ["integer"],
            "bypass": [False],
            "is_qat": [True],
            "data_in_width": [2, 4, 8],
            "data_in_frac_width": [1, 2, 3],
            "weight_width": [2, 4, 8],
            "weight_frac_width": [1, 2, 3],
            "bias_width": [2, 4, 8],
            "bias_frac_width": [1, 2, 3],
        }
    }

    pp(parse_llama_quantized_config(default_config, num_hidden_layers=2))


def test_ast():
    from llm_mixed_q.models import get_quant_config_parser
    import toml

    search_config = toml.load(
        # "../experiments/asplos/configs/search/bert_base_sst2.toml"
        "/data/zz7522/Projects/llm-mixed-q/experiments/asplos/configs/search/bert_base_sst2.toml"
    )

    # parser = get_quant_config_parser("bert")
    # q_config_seed = search_config["search_space"]["quant_config_seed"]

    q_config_seed = {
        "default": {
            "bias_block_size": ["!ast![1, 16]"],
            "bias_exponent_bias": ["!ast!None"],
            "bias_exponent_width": [8],
            "bias_width": [7, 6, 5],
            "bypass": ["!ast!False"],
            "data_in_block_size": ["!ast![1, 16]"],
            "data_in_exponent_bias": ["!ast!None"],
            "data_in_exponent_width": [8],
            "data_in_width": [7, 6, 5],
            "is_qat": ["!ast!False"],
            "name": ["block_fp"],
            "weight_block_size": ["!ast![1, 16]"],
            "weight_exponent_bias": ["!ast!None"],
            "weight_exponent_width": [8],
            "weight_width": [7, 6, 5],
        },
        "model_layer_0": {
            "self_attn": {
                "q_proj": {
                    "data_in_width": [1, 2, 3],
                },
                "k_proj": {
                    "data_in_width": [1, 2, 3],
                },
            }
        },
        "model_layer_1": {
            "self_attn": {
                "q_proj": {
                    "data_in_width": [7, 8, 9],
                },
                "k_proj": {
                    "data_in_width": [7, 8, 9],
                },
            }
        },
    }

    def sample(config, name):
        print(f"sample name = {name}, config = {config}, type(config) = {type(config)}")
        if isinstance(config, dict):
            for k, v in config.items():
                config[k] = sample(v, f"{name}:{k}")
            return config
        elif isinstance(config, list):
            sampled_value = random.choice(config)
            sampled_value = f"{name}:{sampled_value}"
            return sampled_value
        else:
            raise ValueError(f"Unknown type {type(config)}")

    sample(q_config_seed, "root")
    pp(q_config_seed)


def test_trail_to_quant_config():
    def parse_and_create_item(quant_config: dict, keys: list[str], value):
        for i, key in enumerate(keys):
            if key not in quant_config:
                quant_config[key] = {}
            if i == len(keys) - 1:
                quant_config[key] = value
            else:
                quant_config = quant_config[key]

    quant_config = {}

    keys = "model_layer_0:self_attn:q_proj:name"
    value = 111

    parse_and_create_item(quant_config, keys.split(":"), value)

    keys = "model_layer_0:self_attn:k_proj:name"
    value = 222
    parse_and_create_item(quant_config, keys.split(":"), value)

    keys = "model_layer_0:mlp:fc1:name"
    value = 333
    parse_and_create_item(quant_config, keys.split(":"), value)

    pp(quant_config)


if __name__ == "__main__":
    # test_node_parser()
    test_ast()
    # test_trail_to_quant_config()
