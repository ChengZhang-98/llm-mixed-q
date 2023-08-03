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
):
    assert isinstance(layer_qc, dict), f"layer_qc must be a dict, got {layer_qc}"
    # fmt: off
    qc = {
        "self_attn": {
            "q_proj": sample_a_dict_of_list(trial, f"{name}:self_attn:q_proj", layer_qc["self_attn"]["q_proj"]),
            "k_proj": sample_a_dict_of_list(trial, f"{name}:self_attn:k_proj", layer_qc["self_attn"]["k_proj"]),
            "v_proj": sample_a_dict_of_list(trial, f"{name}:self_attn:v_proj", layer_qc["self_attn"]["v_proj"]),
            "out_proj": sample_a_dict_of_list(trial, f"{name}:self_attn:out_proj", layer_qc["self_attn"]["out_proj"]),
            "bmm_0": sample_a_dict_of_list(trial, f"{name}:self_attn:bmm_0", layer_qc["self_attn"]["bmm_0"]),
            "bmm_1": sample_a_dict_of_list(trial, f"{name}:self_attn:bmm_1", layer_qc["self_attn"]["bmm_1"]),
        },
        "fc1": sample_a_dict_of_list(trial, f"{name}:fc1", layer_qc["fc1"]),
        "fc2": sample_a_dict_of_list(trial, f"{name}:fc2", layer_qc["fc2"]),
    }
    # fmt: on

    return qc


def sample_opt_quant_config(
    trial: optuna.Trial,
    name: str,
    config_seed: dict,
):
    sampled_config = {}

    for k, v in config_seed.items():
        if k == "by":
            sampled_config[k] = v
        elif k == "default":
            sampled_config[k] = sample_a_dict_of_list(trial, f"{name}:{k}", v)
        elif k == "model_layer":
            sampled_config[k] = sample_a_layer_quant_config(trial, f"{name}:{k}", v)
        elif "model_layer_" in k:
            sampled_config[k] = sample_a_layer_quant_config(trial, f"{name}:{k}", v)
        else:
            logger.warning(f"Unknown key: {k}, ignored")
    return sampled_config
