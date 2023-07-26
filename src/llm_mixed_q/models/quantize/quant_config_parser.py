from functools import partial
from copy import deepcopy


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


QUANT_ARITH_ENTRIES = {
    "integer": {
        "weight_entries": ("weight_width", "weight_frac_width"),
        "data_in_entries": ("data_in_width", "data_in_frac_width"),
        "bias_entries": ("bias_width", "bias_frac_width"),
    },
    "block_fp": {
        "weight_entries": (
            "weight_width",
            "weight_exponent_width",
            "weight_exponent_bias",
            "weight_block_size",
        ),
        "data_in_entries": (
            "data_in_width",
            "data_in_exponent_width",
            "data_in_exponent_bias",
            "data_in_block_size",
        ),
        "bias_entries": (
            "bias_width",
            "bias_exponent_width",
            "bias_exponent_bias",
            "bias_block_size",
        ),
    },
}


def cp_name(config: dict, p_config: dict, entries=None):
    cp_multi_values(config, p_config, ("name",))


def cp_is_qat(config: dict, p_config: dict, entries=None):
    cp_multi_values(config, p_config, ("is_ptq",))


def cp_weight_entries(config: dict, p_config: dict, entries: dict):
    cp_multi_values(config, p_config, entries["weight_entries"])


def cp_data_in_entries(config: dict, p_config: dict, entries: dict):
    cp_multi_values(config, p_config, entries["data_in_entries"])


def cp_bias_entries(config: dict, p_config: dict, entries: dict):
    cp_multi_values(config, p_config, entries["bias_entries"])


def cp_weight_entries_to_bias(config: dict, p_config: dict, entries: dict):
    if has_multi_keys(config, entries["bias_entries"]):
        cp_multi_values(config, p_config, entries["bias_entries"])
    else:
        cp_multi_values(
            config, p_config, entries["weight_entries"], entries["bias_entries"]
        )


# ====================
# quant_arith_cp_fn_map is a map from quant_arith to a dict of functions
# ====================
# <quant_arith>: {
#    "name": cp_name_function_<quant_arith>,
#    "weight_entries": cp_weight_entries_function_<quant_arith>,
#    "data_in_entries": cp_data_in_entries_function_<quant_arith>,
#    "bias_entries": cp_bias_entries_function_<quant_arith>,
#    "weight_entries_to_bias": cp_weight_entries_to_bias_function_<quant_arith>
# }
QUANT_ARITH_TO_CP_FN = {}


for quant_arith, entries in QUANT_ARITH_ENTRIES.items():
    QUANT_ARITH_TO_CP_FN[quant_arith] = {
        "name": partial(cp_name, entries=entries),
        "is_ptq": partial(cp_is_qat, entries=entries),
        "weight_entries": partial(cp_weight_entries, entries=entries),
        "data_in_entries": partial(cp_data_in_entries, entries=entries),
        "bias_entries": partial(cp_bias_entries, entries=entries),
        "weight_entries_to_bias": partial(cp_weight_entries_to_bias, entries=entries),
    }

MASE_OP_TO_ENTRIES = {
    "add": ("name", "data_in_entries"),
    "bmm": ("name", "data_in_entries", "weight_entries"),
    "conv1d": ("name", "is_ptq", "data_in_entries", "weight_entries", "bias_entries"),
    "conv2d": ("name", "is_ptq", "data_in_entries", "weight_entries", "bias_entries"),
    "matmul": ("name", "data_in_entries", "weight_entries"),
    "mul": ("name", "data_in_entries"),
    "linear": ("name", "is_ptq", "data_in_entries", "weight_entries", "bias_entries"),
    "relu": ("name", "data_in_entries"),
    "sub": ("name", "data_in_entries"),
}


def parse_node_config(config: dict, mase_op: str) -> dict:
    assert mase_op in MASE_OP_TO_ENTRIES, f"Unknown mase op: {mase_op}"
    if config.get("bypass", False):
        return config
    op_entries = MASE_OP_TO_ENTRIES[mase_op]
    p_config = {}
    for entry in op_entries:
        entry_cp_fn = QUANT_ARITH_TO_CP_FN[config["name"]][entry]
        entry_cp_fn(config, p_config)
    return p_config
