from .quantized_functions import QUANTIZED_FUNC_MAP
from .quantized_modules import QUANTIZED_MODULE_MAP
from .quantizers import QUANTIZER_MAP
from .quant_config_parser import parse_node_config
from .quantized_layer_profiler import (
    profile_linear_layer,
    profile_matmul_layer,
    update_profile,
)
from .quant_config_sampler import sample_a_dict_of_list
from .stat_profile_to_quant_config import transform_stat_profile_to_int_quant_config


def get_quantized_cls(op: str, config: dict):
    return QUANTIZED_MODULE_MAP[op][config["name"]]


def get_quantized_func(op: str, config: dict):
    return QUANTIZED_FUNC_MAP[op][config["name"]]


def get_quantizer(op: str, config: dict):
    return QUANTIZER_MAP[config["name"]]
