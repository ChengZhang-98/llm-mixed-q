from .block_fp import block_fp_quantizer
from .block_log import block_log_quantizer
from .block_minifloat import block_minifloat_quantizer
from .integer import integer_quantizer
from .log import log_quantizer
from .minifloat import minifloat_denorm_quantizer, minifloat_ieee_quantizer

QUANTIZER_MAP = {
    "block_fp": block_fp_quantizer,
    "block_log": block_log_quantizer,
    "block_minifloat": block_minifloat_quantizer,
    "integer": integer_quantizer,
    "log": log_quantizer,
    "minifloat_denorm": minifloat_denorm_quantizer,
    "minifloat_ieee": minifloat_ieee_quantizer,
}
