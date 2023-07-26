from .matmul import (
    matmul_block_fp,
    matmul_block_log,
    matmul_block_minifloat,
    matmul_integer,
    matmul_minifloat_denorm,
    matmul_minifloat_ieee,
    bmm_block_fp,
    bmm_block_log,
    bmm_block_minifloat,
    bmm_integer,
    bmm_minifloat_denorm,
    bmm_minifloat_ieee,
)
from .rotary_positional_encoding import (
    apply_rotary_pos_emb_block_fp,
    apply_rotary_pos_emb_block_log,
    apply_rotary_pos_emb_block_minifloat,
    apply_rotary_pos_emb_integer,
    apply_rotary_pos_emb_log,
    apply_rotary_pos_emb_minifloat_denorm,
    apply_rotary_pos_emb_minifloat_ieee,
)

QUANTIZED_FUNC_MAP = {
    "matmul": {
        "block_fp": matmul_block_fp,
        "block_log": matmul_block_log,
        "block_minifloat": matmul_block_minifloat,
        "integer": matmul_integer,
        "log": matmul_block_log,
        "minifloat_denorm": matmul_minifloat_denorm,
        "minifloat_ieee": matmul_minifloat_ieee,
    },
    "bmm": {
        "block_fp": bmm_block_fp,
        "block_log": bmm_block_log,
        "block_minifloat": bmm_block_minifloat,
        "integer": bmm_integer,
        "log": bmm_block_log,
        "minifloat_denorm": bmm_minifloat_denorm,
        "minifloat_ieee": bmm_minifloat_ieee,
    },
    "rotary_positional_encoding": {
        "block_fp": apply_rotary_pos_emb_block_fp,
        "block_log": apply_rotary_pos_emb_block_log,
        "block_minifloat": apply_rotary_pos_emb_block_minifloat,
        "integer": apply_rotary_pos_emb_integer,
        "log": apply_rotary_pos_emb_log,
        "minifloat_denorm": apply_rotary_pos_emb_minifloat_denorm,
        "minifloat_ieee": apply_rotary_pos_emb_minifloat_ieee,
    },
}
