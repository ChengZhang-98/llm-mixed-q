import re
from copy import deepcopy

import toml
from ...utils.config_load import convert_str_na_to_none
from ..quantize.quant_config_parser import parse_node_config

"""
An example of quant_config for llama

{
    "model_layer": {
        "self_attn": {
            "q_proj": {},
            "k_proj": {},
            "v_proj": {},
            "o_proj": {},
            "rotary_positional_encoding": {},
            "matmul_0": {},
            "matmul_1": {},
        },
        "mlp": {
            "gate_proj": {},
            "down_proj": {},
            "up_proj": {},
        },
    }
    "linear_default": {},
    "matmul_default": {},
}
"""


def cp_multi_values(src: dict, dst: dict, src_keys: tuple, dst_keys: tuple = None):
    """Copy multiple values from src dict to dst dict."""
    if dst_keys is None:
        for key in src_keys:
            dst[key] = deepcopy(src[key])
    else:
        for src_key, dst_key in zip(src_keys, dst_keys):
            dst[dst_key] = deepcopy(src[src_key])


def has_multi_keys(src: dict, keys: tuple):
    """Check if src dict has multiple keys."""
    for key in keys:
        if key not in src:
            return False
    return True


def match_a_pattern(name: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        match = re.fullmatch(pattern, name)
        if match:
            return pattern
    return None


def create_a_layer_config(
    linear_qc: dict = None,
    matmul_qc: dict = None,
    rotary_positional_encoding_qc: dict = None,
    layer_qc=None,
) -> dict:
    if (layer_qc is None and matmul_qc is None) and layer_qc is None:
        raise ValueError("Must provide either (linear_qc & matmul_qc) or layer_qc")
    if layer_qc is None:
        layer_qc = {}
    # fmt: off
    qc = {
        "self_attn": {
            "q_proj": deepcopy(parse_node_config(layer_qc.get("self_attn", {}).get("q_proj", linear_qc), "linear")),
            "k_proj": deepcopy(parse_node_config(layer_qc.get("self_attn", {}).get("k_proj", linear_qc), "linear")),
            "v_proj": deepcopy(parse_node_config(layer_qc.get("self_attn", {}).get("v_proj", linear_qc), "linear")),
            "o_proj": deepcopy(parse_node_config(layer_qc.get("self_attn", {}).get("o_proj", linear_qc), "linear")),
            "rotary_positional_encoding": deepcopy(parse_node_config(layer_qc.get("self_attn", {}).get("rotary_positional_encoding", rotary_positional_encoding_qc), "rotary_positional_encoding")),
            "matmul_0": deepcopy(parse_node_config(layer_qc.get("self_attn", {}).get("matmul_0", matmul_qc), "matmul")),
            "matmul_1": deepcopy(parse_node_config(layer_qc.get("self_attn", {}).get("matmul_1", matmul_qc), "matmul")),
        },
        "mlp": {
            "gate_proj": deepcopy(parse_node_config(layer_qc.get("mlp", {}).get("gate_proj", linear_qc), "linear")),
            "down_proj": deepcopy(parse_node_config(layer_qc.get("mlp", {}).get("down_proj", linear_qc), "linear")),
            "up_proj": deepcopy(parse_node_config(layer_qc.get("mlp", {}).get("up_proj", linear_qc), "linear"))
        },
    }
    # fmt: on
    return qc


# def by_type_parser(config: dict, num_hidden_layers: int) -> dict:
#     assert "default" in config, "Must provide default config for by_class_parser"
#     default_qc: dict = config["default"]
#     linear_qc: dict = parse_node_config(
#         config.get("linear", default_qc), mase_op="linear"
#     )
#     rotary_positional_encoding_qc: dict = parse_node_config(
#         config.get("rotary_positional_encoding", default_qc),
#         mase_op="rotary_positional_encoding",
#     )
#     matmul_qc: dict = parse_node_config(
#         config.get("matmul", default_qc), mase_op="matmul"
#     )
#     layer_qc: dict = config.get("model_layer", None)

#     # parsed config
#     p_config = {}
#     for i in range(num_hidden_layers):
#         layer_entry = f"model_layer_{i}"
#         p_config[layer_entry] = create_a_layer_config(
#             linear_qc, matmul_qc, rotary_positional_encoding_qc, layer_qc
#         )
#     p_config["default"] = default_qc
#     return p_config


def _parse_and_complete_config(config: dict, num_hidden_layers: int) -> dict:
    assert "default" in config, "Must provide default config for by_name_parser"
    default_qc: dict = config["default"]
    linear_qc: dict = parse_node_config(
        config.get("linear", default_qc), mase_op="linear"
    )
    rotary_positional_encoding_qc: dict = parse_node_config(
        config.get("rotary_positional_encoding", default_qc),
        mase_op="rotary_positional_encoding",
    )
    matmul_qc: dict = parse_node_config(
        config.get("matmul", default_qc), mase_op="matmul"
    )
    general_layer_qc: dict = config.get("model_layer", None)

    # parsed config
    p_config = {}
    for i in range(num_hidden_layers):
        layer_entry = f"model_layer_{i}"
        layer_qc = config.get(layer_entry, general_layer_qc)
        p_config[layer_entry] = create_a_layer_config(
            linear_qc, matmul_qc, rotary_positional_encoding_qc, layer_qc
        )
    p_config["default"] = default_qc
    return p_config


def parse_llama_quantized_config(config: str | dict | None, num_hidden_layers: int):
    assert isinstance(
        config, (str, dict, type(None))
    ), "config must be a str path to config toml, None or dict"

    if config is None:
        return None

    if isinstance(config, str):
        config = toml.load(config)

    config = convert_str_na_to_none(config)
    parsed_config = _parse_and_complete_config(config, num_hidden_layers)
    return parsed_config
