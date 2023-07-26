from functools import partial

import torch
from torch import Tensor
from ..quantizers import (
    block_fp_quantizer,
    block_log_quantizer,
    block_minifloat_quantizer,
    integer_quantizer,
    log_quantizer,
    minifloat_denorm_quantizer,
    minifloat_ieee_quantizer,
)


def _rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]  # [bs, nh, t, hd/2]
    x2 = x[..., x.shape[-1] // 2 :]  # [bs, nh, t, hd/2]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, position_ids: list[int] | Tensor
):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_block_fp(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_ids: list[int] | Tensor,
    config: dict,
):
    freq_quantizer = partial(
        block_fp_quantizer,
        width=config["data_in_width"],
        exponent_width=config["data_in_exponent_width"],
        exponent_bias=config["data_in_exponent_bias"],
        block_size=config["data_in_block_size"],
        skip_first_dim=False,
    )
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = freq_quantizer(cos.squeeze(1).squeeze(0))  # [seq_len, dim]
    sin = freq_quantizer(sin.squeeze(1).squeeze(0))  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_block_log(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_ids: list[int] | Tensor,
    config: dict,
):
    freq_quantizer = partial(
        block_log_quantizer,
        width=config["data_in_width"],
        exponent_bias_width=config["data_in_exponent_bias_width"],
        block_size=config["data_in_block_size"],
        skip_first_dim=False,
    )
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = freq_quantizer(cos.squeeze(1).squeeze(0))  # [seq_len, dim]
    sin = freq_quantizer(sin.squeeze(1).squeeze(0))  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_block_minifloat(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_ids: list[int] | Tensor,
    config: dict,
):
    freq_quantizer = partial(
        block_minifloat_quantizer,
        width=config["data_in_width"],
        exponent_width=config["data_in_exponent_width"],
        exponent_bias_width=config["data_in_exponent_bias_width"],
        block_size=config["data_in_block_size"],
        skip_first_dim=False,
    )
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = freq_quantizer(cos.squeeze(1).squeeze(0))  # [seq_len, dim]
    sin = freq_quantizer(sin.squeeze(1).squeeze(0))  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_integer(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_ids: list[int] | Tensor,
    config: dict,
):
    freq_quantizer = partial(
        integer_quantizer,
        width=config["data_in_width"],
        frac_width=config["data_in_frac_width"],
    )
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = freq_quantizer(cos.squeeze(1).squeeze(0))  # [seq_len, dim]
    sin = freq_quantizer(sin.squeeze(1).squeeze(0))  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_log(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_ids: list[int] | Tensor,
    config: dict,
):
    freq_quantizer = partial(
        log_quantizer,
        width=config["data_in_width"],
        exponent_bias=config["data_in_exponent_bias"],
    )
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = freq_quantizer(cos.squeeze(1).squeeze(0))  # [seq_len, dim]
    sin = freq_quantizer(sin.squeeze(1).squeeze(0))  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_minifloat_denorm(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_ids: list[int] | Tensor,
    config: dict,
):
    freq_quantizer = partial(
        minifloat_denorm_quantizer,
        width=config["data_in_width"],
        exponent_width=config["data_in_exponent_width"],
        exponent_bias=config["data_in_exponent_bias"],
    )
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = freq_quantizer(cos.squeeze(1).squeeze(0))  # [seq_len, dim]
    sin = freq_quantizer(sin.squeeze(1).squeeze(0))  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_minifloat_ieee(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_ids: list[int] | Tensor,
    config: dict,
):
    freq_quantizer = partial(
        minifloat_ieee_quantizer,
        width=config["data_in_width"],
        exponent_width=config["data_in_exponent_width"],
        exponent_bias=config["data_in_exponent_bias"],
    )
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = freq_quantizer(cos.squeeze(1).squeeze(0))  # [seq_len, dim]
    sin = freq_quantizer(sin.squeeze(1).squeeze(0))  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed
