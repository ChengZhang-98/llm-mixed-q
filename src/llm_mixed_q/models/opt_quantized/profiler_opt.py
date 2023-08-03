import torch

from ..quantize.quantized_layer_profiler import (profile_linear_layer,
                                                 profile_matmul_layer,
                                                 register_a_stat_hook,
                                                 update_profile)
from .modeling_opt import (OPTQuantizedDecoderLayer,
                           OPTQuantizedForSequenceClassification)


def _profile_bitwidth_opt_layer(
    layer_quant_config: dict,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    seq_len: int,
    bias: bool,
):
    """
    K = X W_k + b
    Q = X W_q + b
    V = X W_v + b

    A = Q K^T
    A = A V

    O = A W_o + b
    Y = O W_1 + b
    Y = Y W_2 + b

    """

    profile = {
        "num_params": 0,
        "num_acts": 0,
        "param_bits": 0,
        "act_bits": 0,
    }
    delta_list = []
    delta_list.append(
        profile_linear_layer(
            layer_quant_config["self_attn"]["q_proj"],
            in_features=hidden_size,
            out_features=hidden_size,
            bias=bias,
            batch_size=seq_len,
        )
    )
    delta_list.append(
        profile_linear_layer(
            layer_quant_config["self_attn"]["k_proj"],
            in_features=hidden_size,
            out_features=hidden_size,
            bias=bias,
            batch_size=seq_len,
        )
    )
    delta_list.append(
        profile_linear_layer(
            layer_quant_config["self_attn"]["v_proj"],
            in_features=hidden_size,
            out_features=hidden_size,
            bias=bias,
            batch_size=seq_len,
        )
    )
    for i in range(num_attention_heads):
        delta_list.append(
            profile_matmul_layer(
                layer_quant_config["self_attn"]["bmm_0"],
                data_in_0_size=(seq_len, hidden_size // num_attention_heads),
                data_in_1_size=(hidden_size // num_attention_heads, seq_len),
            )
        )
        delta_list.append(
            profile_matmul_layer(
                layer_quant_config["self_attn"]["bmm_1"],
                data_in_0_size=(seq_len, seq_len),
                data_in_1_size=(seq_len, hidden_size // num_attention_heads),
            )
        )
    delta_list.append(
        profile_linear_layer(
            layer_quant_config["self_attn"]["out_proj"],
            in_features=hidden_size,
            out_features=hidden_size,
            bias=bias,
            batch_size=seq_len,
        )
    )
    delta_list.append(
        profile_linear_layer(
            layer_quant_config["fc1"],
            in_features=hidden_size,
            out_features=intermediate_size,
            bias=bias,
            batch_size=seq_len,
        )
    )
    delta_list.append(
        profile_linear_layer(
            layer_quant_config["fc2"],
            in_features=intermediate_size,
            out_features=hidden_size,
            bias=bias,
            batch_size=seq_len,
        )
    )

    for delta in delta_list:
        update_profile(profile, delta)
    return profile


def profile_bitwidth_opt_quantized(config, seq_len: int):
    """
    Profile opt quantized model

    Args:
        config (OPTQuantizedConfig): opt quantized config
        seq_len (int): sequence length
    """
    hidden_size = config.hidden_size
    intermediate_size = config.ffn_dim
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
            profile=profile,
            delta=_profile_bitwidth_opt_layer(
                layer_quant_config,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=config.num_attention_heads,
                seq_len=seq_len,
                bias=config.enable_bias,
            ),
        )
    return profile


# ================================================================================
# Register statistic profiler hooks
# ================================================================================


def _register_stat_hook_opt_layer(
    stat_manager, decoder_layer: OPTQuantizedDecoderLayer, name: str
):
    hooks_to_register = {
        "self_attn": {
            "q_proj": ["data_in", "weight", "bias", "data_out"],
            "k_proj": ["data_in", "weight", "bias", "data_out"],
            "v_proj": ["data_in", "weight", "bias", "data_out"],
            "out_proj": ["data_in", "weight", "bias"],
        },
        "fc1": ["data_in", "weight", "bias"],
        "fc2": ["data_in", "weight", "bias"],
    }
    # fmt: off

    # k, q, v, o_proj
    self_attn_name = f"{name}:self_attn"
    for layer, entries in hooks_to_register["self_attn"].items():
        for entry in entries:
            entry_name = f"{self_attn_name}:{layer}:{entry}"
            register_a_stat_hook(stat_manager, name=entry_name, module=getattr(decoder_layer.self_attn, layer), entry=entry)

    # fc1, fc2
    for layer in ["fc1", "fc2"]:
        for entry in hooks_to_register[layer]:
            entry_name = f"{name}:{layer}:{entry}"
            register_a_stat_hook(stat_manager, name=entry_name, module=getattr(decoder_layer, layer), entry=entry)


def register_stat_hooks_opt_quantized(
    stat_manager,
    name: str,
    model: OPTQuantizedForSequenceClassification,
    num_hidden_layers: int,
):
    """
    Register statistic profiler hooks for opt quantized model

    Args:
        decoder (OPTQuantizedDecoder): opt quantized decoder
        stat_manager (StatManager): statistic profiler
    """
    for i in range(num_hidden_layers):
        model_layer = model.model.decoder.layers[i]
        layer_name = f"{name}:model_layer_{i}"
        _register_stat_hook_opt_layer(
            decoder_layer=model_layer,
            stat_manager=stat_manager,
            name=layer_name,
        )
