from .linear import (LinearBlockFP, LinearBlockLog, LinearBlockMinifloat,
                     LinearInteger, LinearLog, LinearMinifloatDenorm,
                     LinearMinifloatIEEE)

QUANTIZED_MODULE_MAP = {
    "linear": {
        "block_fp": LinearBlockFP,
        "integer": LinearInteger,
        "minifloat_ieee": LinearMinifloatIEEE,
        "minifloat_denorm": LinearMinifloatDenorm,
        "block_log": LinearBlockLog,
        "log": LinearLog,
        "block_minifloat": LinearBlockMinifloat,
    },
}
