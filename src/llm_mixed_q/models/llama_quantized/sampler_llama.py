import ast
import logging
from copy import deepcopy

import optuna

from ..quantize import parse_node_config, sample_a_dict_of_list

logger = logging.getLogger(__name__)


def sample_a_layer_quant_config(
    trial: optuna.Trial,
    name: str,
    layer_qc: dict,
) -> dict:
    assert isinstance(layer_qc, dict), f"layer_qc must be a dict, got {layer_qc}"
    # fmt:off
    qc = {
        "self_attn": {
            "q_proj": sample_a_dict_of_list(trial, f"{name}:self_attn:q_proj", layer_qc["self_attn"]["q_proj"]),
            "k_proj": sample_a_dict_of_list(trial, f"{name}:self_attn:k_proj", layer_qc["self_attn"]["k_proj"]),
            "v_proj": sample_a_dict_of_list(trial, f"{name}:self_attn:v_proj", layer_qc["self_attn"]["v_proj"]),
            "o_proj": sample_a_dict_of_list(trial, f"{name}:self_attn:o_proj", layer_qc["self_attn"]["o_proj"]),
            "rotary_positional_encoding": sample_a_dict_of_list(trial, f"{name}:self_attn:rotary_positional_encoding", layer_qc["self_attn"]["rotary_positional_encoding"]),
            "matmul_0": sample_a_dict_of_list(trial, f"{name}:self_attn:matmul_0", layer_qc["self_attn"]["matmul_0"]),
            "matmul_1": sample_a_dict_of_list(trial, f"{name}:self_attn:matmul_1", layer_qc["self_attn"]["matmul_1"]),
        },
        "mlp": {
            "gate_proj": sample_a_dict_of_list(trial, f"{name}:mlp:gate_proj", layer_qc["mlp"]["gate_proj"]),
            "down_proj": sample_a_dict_of_list(trial, f"{name}:mlp:down_proj", layer_qc["mlp"]["down_proj"]),
            "up_proj": sample_a_dict_of_list(trial, f"{name}:mlp:up_proj", layer_qc["mlp"]["up_proj"])
        },
    }
    # fmt:on
    return qc


def sample_llama_quant_config(
    trial: optuna.Trial,
    name: str,
    config_seed: dict,
):
    sampled_config = {}

    for k, v in config_seed.items():
        if k == "default":
            sampled_config[k] = sample_a_dict_of_list(trial, f"{name}:{k}", v)
        elif k == "model_layer":
            sampled_config[k] = sample_a_layer_quant_config(trial, f"{name}:{k}", v)
        elif "model_layer_" in k:
            sampled_config[k] = sample_a_layer_quant_config(trial, f"{name}:{k}", v)
        elif k == "rotary_positional_encoding":
            sampled_config[k] = sample_a_dict_of_list(trial, f"{name}:{k}", v)
        else:
            logger.warning(f"Unknown key: {k}, ignored")
    return sampled_config
