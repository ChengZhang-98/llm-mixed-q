import logging
import ast
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
        "attention": {
            "query": sample_a_dict_of_list(trial, f"{name}:attention:query", layer_qc["attention"]["query"]),
            "key": sample_a_dict_of_list(trial, f"{name}:attention:key", layer_qc["attention"]["key"]),
            "value": sample_a_dict_of_list(trial, f"{name}:attention:value", layer_qc["attention"]["value"]),
            "matmul_0": sample_a_dict_of_list(trial, f"{name}:attention:matmul_0", layer_qc["attention"]["matmul_0"]),
            "matmul_1": sample_a_dict_of_list(trial, f"{name}:attention:matmul_1", layer_qc["attention"]["matmul_1"]),
            "output": {
                "dense": sample_a_dict_of_list(trial, f"{name}:attention:output:dense", layer_qc["attention"]["output"]["dense"]),
            },
        },
        "intermediate": {
            "dense": sample_a_dict_of_list(trial, f"{name}:intermediate:dense", layer_qc["intermediate"]["dense"]),
        },
        "output": {
            "dense": sample_a_dict_of_list(trial, f"{name}:output:dense", layer_qc["output"]["dense"]),
        }
    }
    # fmt:on
    return qc


def sample_bert_quant_config(
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
