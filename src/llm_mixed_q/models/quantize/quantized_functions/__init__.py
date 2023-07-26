from .matmul import matmul_block_fp, matmul_integer, bmm_block_fp, bmm_integer

QUANTIZED_FUNC_MAP = {
    "matmul": {
        "block_fp": matmul_block_fp,
        "integer": matmul_integer,
    },
    "bmm": {
        "block_fp": bmm_block_fp,
        "integer": bmm_integer,
    },
}
