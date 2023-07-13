from .quantized_functions import QUANTIZED_FUNC_MAP
from .quantized_modules import QUANTIZED_MODULE_MAP
from .quant_config_parser import parse_node_config


def get_quantized_cls(op: str, config: dict):
    return QUANTIZED_MODULE_MAP[op][config["name"]]


def get_quantized_func(op: str, config: dict):
    return QUANTIZED_FUNC_MAP[op][config["name"]]
