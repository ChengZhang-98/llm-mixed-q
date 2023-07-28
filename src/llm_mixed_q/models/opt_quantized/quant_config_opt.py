import re
from copy import deepcopy

import toml
from ...utils.config_load import convert_str_na_to_none
from ..quantize.quant_config_parser import parse_node_config

"""
An example of quant_config for opt

{
    "model_layer": {
        "self_attn": {
            "q_proj": {},
            "k_proj": {},
            "v_proj": {},
            "out_proj": {},
            "matmul_0": {},
            "matmul_1": {},
        },
        "fc1": {},
        "fc2": {},
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
            "matmul_0": deepcopy(parse_node_config(layer_qc.get("self_attn", {}).get("matmul_0", matmul_qc), "matmul")),
            "matmul_1": deepcopy(parse_node_config(layer_qc.get("self_attn", {}).get("matmul_1", matmul_qc), "matmul")),
        },
        "fc1": deepcopy(parse_node_config(layer_qc.get("fc1", linear_qc), "linear")),
        "fc2": deepcopy(parse_node_config(layer_qc.get("fc2", linear_qc), "linear")),
    }
    # fmt: on
    return qc


def by_type_parser(config: dict, num_hidden_layers: int) -> dict:
    assert "default" in config, "Must provide default config for by_class_parser"
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
    layer_qc: dict = config.get("model_layer", None)

    # parsed config
    p_config = {}
    for i in range(num_hidden_layers):
        layer_entry = f"model_layer_{i}"
        p_config[layer_entry] = create_a_layer_config(
            linear_qc, matmul_qc, rotary_positional_encoding_qc, layer_qc
        )
    p_config["default"] = default_qc
    return p_config


def by_name_parser(config: dict, num_hidden_layers: int) -> dict:
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

    # parsed config
    p_config = {}
    for i in range(num_hidden_layers):
        layer_entry = f"model_layer_{i}"
        layer_qc = config.get(layer_entry, None)
        p_config[layer_entry] = create_a_layer_config(
            linear_qc, matmul_qc, rotary_positional_encoding_qc, layer_qc
        )
    p_config["default"] = default_qc
    return p_config
