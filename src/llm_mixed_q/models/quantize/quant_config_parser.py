from functools import partial
from copy import deepcopy
import logging


logger = logging.getLogger(__name__)


def cp_multi_values(
    src: dict, dst: dict, src_keys: tuple, dst_keys: tuple = None, strict: bool = True
):
    """Copy multiple values from src dict to dst dict."""
    if dst_keys is None:
        for key in src_keys:
            if not strict and key not in src:
                continue
            dst[key] = deepcopy(src[key])
    else:
        for src_key, dst_key in zip(src_keys, dst_keys):
            if not strict and src_key not in src:
                continue
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
        "data_out_entries": ("data_out_width", "data_out_frac_width"),
    },
    "minifloat_ieee": {
        "weight_entries": (
            "weight_width",
            "weight_exponent_width",
            "weight_exponent_bias",
        ),
        "data_in_entries": (
            "data_in_width",
            "data_in_exponent_width",
            "data_in_exponent_bias",
        ),
        "bias_entries": ("bias_width", "bias_exponent_width", "bias_exponent_bias"),
        "data_out_entries": (
            "data_out_width",
            "data_out_exponent_width",
            "data_out_exponent_bias",
        ),
    },
    "minifloat_denorm": {
        "weight_entries": (
            "weight_width",
            "weight_exponent_width",
            "weight_exponent_bias",
        ),
        "data_in_entries": (
            "data_in_width",
            "data_in_exponent_width",
            "data_in_exponent_bias",
        ),
        "bias_entries": ("bias_width", "bias_exponent_width", "bias_exponent_bias"),
        "data_out_entries": (
            "data_out_width",
            "data_out_exponent_width",
            "data_out_exponent_bias",
        ),
    },
    "log": {
        "weight_entries": ("weight_width", "weight_exponent_bias"),
        "data_in_entries": ("data_in_width", "data_in_exponent_bias"),
        "bias_entries": ("bias_width", "bias_exponent_bias"),
        "data_out_entries": ("data_out_width", "data_out_exponent_bias"),
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
        "data_out_entries": (
            "data_out_width",
            "data_out_exponent_width",
            "data_out_exponent_bias",
            "data_out_block_size",
        ),
    },
    "block_minifloat": {
        "weight_entries": (
            "weight_width",
            "weight_exponent_width",
            "weight_exponent_bias_width",
            "weight_block_size",
        ),
        "data_in_entries": (
            "data_in_width",
            "data_in_exponent_width",
            "data_in_exponent_bias_width",
            "data_in_block_size",
        ),
        "bias_entries": (
            "bias_width",
            "bias_exponent_width",
            "bias_exponent_bias_width",
            "bias_block_size",
        ),
        "data_out_entries": (
            "data_out_width",
            "data_out_exponent_width",
            "data_out_exponent_bias_width",
            "data_out_block_size",
        ),
    },
    "block_log": {
        "weight_entries": (
            "weight_width",
            "weight_exponent_bias_width",
            "weight_block_size",
        ),
        "data_in_entries": (
            "data_in_width",
            "data_in_exponent_bias_width",
            "data_in_block_size",
        ),
        "bias_entries": (
            "bias_width",
            "bias_exponent_bias_width",
            "bias_block_size",
        ),
        "data_out_entries": (
            "data_out_width",
            "data_out_exponent_bias_width",
            "data_out_block_size",
        ),
    },
}


def cp_name(config: dict, p_config: dict, entries=None, strict: bool = True):
    cp_multi_values(config, p_config, ("name",), strict=strict)


def cp_bypass(config: dict, p_config: dict, entries=None, strict: bool = True):
    cp_multi_values(config, p_config, ("bypass",), strict=strict)


def cp_is_qat(config: dict, p_config: dict, entries=None, strict: bool = True):
    cp_multi_values(config, p_config, ("is_ptq",), strict=strict)


def cp_weight_entries(config: dict, p_config: dict, entries: dict, strict: bool = True):
    cp_multi_values(config, p_config, entries["weight_entries"], strict=strict)


def cp_data_in_entries(
    config: dict, p_config: dict, entries: dict, strict: bool = True
):
    cp_multi_values(config, p_config, entries["data_in_entries"], strict=strict)


def cp_bias_entries(config: dict, p_config: dict, entries: dict, strict: bool = True):
    cp_multi_values(config, p_config, entries["bias_entries"], strict=strict)


def cp_weight_entries_to_bias(
    config: dict, p_config: dict, entries: dict, strict: bool = True
):
    if has_multi_keys(config, entries["bias_entries"]):
        cp_multi_values(
            config,
            p_config,
            entries["bias_entries"],
            entries["bias_entries"],
            strict=strict,
        )
    else:
        cp_multi_values(
            config,
            p_config,
            entries["weight_entries"],
            entries["bias_entries"],
            strict=strict,
        )


def cp_data_out_entries(
    config: dict, p_config: dict, entries: dict, strict: bool = True
):
    cp_multi_values(config, p_config, entries["data_out_entries"], strict=strict)


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
        "bypass": partial(cp_bypass, entries=entries),
        "is_ptq": partial(cp_is_qat, entries=entries),
        "weight_entries": partial(cp_weight_entries, entries=entries),
        "data_in_entries": partial(cp_data_in_entries, entries=entries),
        "bias_entries": partial(cp_bias_entries, entries=entries),
        "weight_entries_to_bias": partial(cp_weight_entries_to_bias, entries=entries),
        "data_out_entries": partial(cp_data_out_entries, entries=entries),
    }

MASE_OP_TO_ENTRIES = {
    # <op_name> : (<entries>, <optional_entries>)
    "add": (("name", "data_in_entries"), ("bypass",)),
    "bmm": (("name", "data_in_entries", "weight_entries"), ("bypass",)),
    "conv1d": (
        ("name", "is_ptq", "data_in_entries", "weight_entries"),
        (
            "bias_entries",
            "bypass",
        ),
    ),
    "conv2d": (
        ("name", "is_ptq", "data_in_entries", "weight_entries"),
        (
            "bias_entries",
            "bypass",
        ),
    ),
    "matmul": (("name", "data_in_entries", "weight_entries"), ("bypass",)),
    "mul": (("name", "data_in_entries"), ("bypass",)),
    "linear": (
        ("name", "is_ptq", "data_in_entries", "weight_entries"),
        (
            "bias_entries",
            "data_out_entries",
            "bypass",
        ),
    ),
    "relu": (("name", "data_in_entries"), ("bypass",)),
    "rotary_positional_encoding": (("name", "data_in_entries"), ("bypass",)),
    "sub": (("name", "data_in_entries"), ("bypass",)),
}


def optional_entry_exists(config: dict, entry_name: str) -> bool:
    entry_name = entry_name.removesuffix("_entries")
    for key in config.keys():
        if key.startswith(entry_name):
            return True
    return False


def parse_node_config(config: dict, mase_op: str, strict: bool = True) -> dict:
    """
    Parse a node config from a MASE op config.

    Args:
        - `strict` (bool) allows missing node config entries if False, e.g.,
        a missing `bias_frac_width` in linear node config
    """
    assert mase_op in MASE_OP_TO_ENTRIES, f"Unknown mase op: {mase_op}"
    if config.get("bypass", False):
        return config
    op_entries, op_entries_optional = MASE_OP_TO_ENTRIES[mase_op]
    assert isinstance(
        op_entries, tuple
    ), f"op_entries must be a tuple, check MASE_OP_TO_ENTRIES[{mase_op}]"
    assert isinstance(
        op_entries_optional, tuple
    ), f"op_entries_optional must be a tuple, check MASE_OP_TO_ENTRIES[{mase_op}]"
    p_config = {}
    for entry in op_entries:
        entry_cp_fn = QUANT_ARITH_TO_CP_FN[config["name"]][entry]
        entry_cp_fn(config, p_config, strict=strict)
    for entry in op_entries_optional:
        if optional_entry_exists(config, entry):
            entry_cp_fn = QUANT_ARITH_TO_CP_FN[config["name"]][entry]
            entry_cp_fn(config, p_config, strict=strict)
    return p_config
