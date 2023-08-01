from ..quantize.quantized_layer_profiler import (
    profile_linear_layer,
    profile_matmul_layer,
    update_profile,
)
import logging

logger = logging.getLogger(__name__)


def log_avg_bitwidth(profile, tag=None):
    logger.debug(f"Profiler tag: {tag}")
    logger.debug(
        f"{tag} avg param bitwidth: {profile['param_bits'] / profile['num_params']}"
    )
    logger.debug(f"{tag} avg act bitwidth: {profile['act_bits'] / profile['num_acts']}")
    return profile


def _profile_bitwidth_bert_attention_layer(
    attn_quant_config: dict,
    hidden_size: int,
    num_attention_heads: int,
    seq_len: int,
):
    profile = {
        "num_params": 0,
        "num_acts": 0,
        "param_bits": 0,
        "act_bits": 0,
    }

    delta_list = []
    delta_list.append(
        profile_linear_layer(
            attn_quant_config["query"],
            in_features=hidden_size,
            out_features=hidden_size,
            bias=True,
            batch_size=seq_len,
        ),
    )
    delta_list.append(
        profile_linear_layer(
            attn_quant_config["key"],
            in_features=hidden_size,
            out_features=hidden_size,
            bias=True,
            batch_size=seq_len,
        )
    )
    delta_list.append(
        profile_linear_layer(
            attn_quant_config["value"],
            in_features=hidden_size,
            out_features=hidden_size,
            bias=True,
            batch_size=seq_len,
        ),
    )
    for i in range(num_attention_heads):
        delta_list.append(
            profile_matmul_layer(
                attn_quant_config["matmul_0"],
                data_in_0_size=(seq_len, hidden_size // num_attention_heads),
                data_in_1_size=(hidden_size // num_attention_heads, seq_len),
            )
        )
        delta_list.append(
            profile_matmul_layer(
                attn_quant_config["matmul_1"],
                data_in_0_size=(seq_len, seq_len),
                data_in_1_size=(seq_len, hidden_size // num_attention_heads),
            )
        )
    delta_list.append(
        profile_linear_layer(
            attn_quant_config["output"]["dense"],
            in_features=hidden_size,
            out_features=hidden_size,
            bias=True,
            batch_size=seq_len,
        ),
    )
    for delta in delta_list:
        update_profile(profile, delta)
    return profile


def _profile_bitwidth_bert_layer(
    layer_quant_config: dict,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    seq_len: int,
):
    profile = {
        "num_params": 0,
        "num_acts": 0,
        "param_bits": 0,
        "act_bits": 0,
    }

    delta_list = []
    delta_list.append(
        _profile_bitwidth_bert_attention_layer(
            layer_quant_config["attention"],
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            seq_len=seq_len,
        ),
    )
    # TODO: does not support crossattention
    delta_list.append(
        profile_linear_layer(
            layer_quant_config["intermediate"]["dense"],
            in_features=hidden_size,
            out_features=intermediate_size,
            bias=True,
            batch_size=seq_len,
        )
    )
    delta_list.append(
        profile_linear_layer(
            layer_quant_config["output"]["dense"],
            in_features=intermediate_size,
            out_features=hidden_size,
            bias=True,
            batch_size=seq_len,
        )
    )

    for delta in delta_list:
        update_profile(profile, delta)

    return profile


def profile_bitwidth_bert_quantized(config, seq_len: int):
    """
    Profile a quantized bert model

    Args:
        config (BertQuantizedConfig): bert quantized config
        seq_len (int): sequence length
    """
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    num_hidden_layers = config.num_hidden_layers

    profile = {
        "num_params": 0,
        "num_acts": 0,
        "param_bits": 0,
        "act_bits": 0,
    }

    for i in range(num_hidden_layers):
        layer_quant_config = config.quant_config[f"model_layer_{i}"]
        update_profile(
            profile,
            _profile_bitwidth_bert_layer(
                layer_quant_config,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=config.num_attention_heads,
                seq_len=seq_len,
            ),
        )

    return profile
