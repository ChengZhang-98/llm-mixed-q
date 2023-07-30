import re
from copy import deepcopy
import logging

import toml
from ...utils.config_load import convert_str_na_to_none
from ..quantize.quant_config_parser import parse_node_config

logger = logging.getLogger(__name__)

"""
An example of quant_config for opt

{
    "model_layer": {
        "self_attn": {
            "q_proj": {},
            "k_proj": {},
            "v_proj": {},
            "out_proj": {},
            "bmm_0": {},
            "bmm_1": {},
        },
        "fc1": {},
        "fc2": {},
    }
    "linear_default": {},
    "matmul_default": {},
}
"""


def create_a_layer_config(
    linear_qc: dict = None,
    bmm_qc: dict = None,
    layer_qc=None,
) -> dict:
    if (layer_qc is None and bmm_qc is None) and layer_qc is None:
        raise ValueError("Must provide either (linear_qc & bmm_qc ) or layer_qc")
    if layer_qc is None:
        layer_qc = {}
    # fmt: off
    qc = {
        "self_attn": {
            "q_proj": deepcopy(parse_node_config(layer_qc.get("self_attn", {}).get("q_proj", linear_qc), "linear")),
            "k_proj": deepcopy(parse_node_config(layer_qc.get("self_attn", {}).get("k_proj", linear_qc), "linear")),
            "v_proj": deepcopy(parse_node_config(layer_qc.get("self_attn", {}).get("v_proj", linear_qc), "linear")),
            "out_proj": deepcopy(parse_node_config(layer_qc.get("self_attn", {}).get("out_proj", linear_qc), "linear")),
            "bmm_0": deepcopy(parse_node_config(layer_qc.get("self_attn", {}).get("bmm_0", bmm_qc), "matmul")),
            "bmm_1": deepcopy(parse_node_config(layer_qc.get("self_attn", {}).get("bmm_1", bmm_qc), "matmul")),
        },
        "fc1": deepcopy(parse_node_config(layer_qc.get("fc1", linear_qc), "linear")),
        "fc2": deepcopy(parse_node_config(layer_qc.get("fc2", linear_qc), "linear")),
    }
    # fmt: on
    return qc


# def by_type_parser(config: dict, num_hidden_layers: int) -> dict:
#     assert "default" in config, "Must provide default config for by_class_parser"
#     default_qc: dict = config["default"]
#     linear_qc: dict = parse_node_config(
#         config.get("linear", default_qc), mase_op="linear"
#     )
#     bmm_qc: dict = parse_node_config(config.get("bmm", default_qc), mase_op="bmm")
#     layer_qc: dict = config.get("model_layer", None)

#     # parsed config
#     p_config = {}
#     for i in range(num_hidden_layers):
#         layer_entry = f"model_layer_{i}"
#         p_config[layer_entry] = create_a_layer_config(linear_qc, bmm_qc, layer_qc)
#     p_config["default"] = default_qc
#     return p_config


def _parse_and_complete_config(config: dict, num_hidden_layers: int) -> dict:
    assert "default" in config, "Must provide default config for by_name_parser"
    default_qc: dict = config["default"]
    linear_qc: dict = parse_node_config(
        config.get("linear", default_qc), mase_op="linear"
    )
    bmm_qc: dict = parse_node_config(config.get("bmm", default_qc), mase_op="matmul")
    general_layer_qc: dict = config.get("model_layer", None)

    # parsed config
    p_config = {}
    for i in range(num_hidden_layers):
        layer_entry = f"model_layer_{i}"
        layer_qc = config.get(layer_entry, general_layer_qc)
        p_config[layer_entry] = create_a_layer_config(linear_qc, bmm_qc, layer_qc)
    p_config["default"] = default_qc
    return p_config


def parse_opt_quantized_config(
    config: str | dict | None, num_hidden_layers: int
) -> dict:
    # logger.debug(f"Parsing opt quant config: {config}")
    assert isinstance(
        config, (str, dict, type(None))
    ), "Must provide either a path, None or a dict"
    if config is None:
        return None
    if isinstance(config, str):
        config = toml.load(config)
    config = convert_str_na_to_none(config)
    parsed_config = _parse_and_complete_config(config, num_hidden_layers)
    return parsed_config
