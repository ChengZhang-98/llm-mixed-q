from .linear import LinearBlockFP, LinearInteger

QUANTIZED_MODULE_MAP = {
    "linear": {
        "block_fp": LinearBlockFP,
        "integer": LinearInteger,
    },
}
