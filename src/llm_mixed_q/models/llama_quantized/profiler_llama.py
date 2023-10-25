from ..quantize.quantized_layer_profiler import (profile_linear_layer,
                                                 profile_matmul_layer,
                                                 register_a_stat_hook,
                                                 update_profile)
from .modeling_llama import (LlamaQuantizedDecoderLayer,
                             LlamaQuantizedForCausalLM)


def _profile_llama_layer(
    layer_quant_config: dict,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    seq_len: int,
):
    """
    K = X W_k
    Q = X W_q
    V = X W_v

    A = Q K^T
    A = A V

    O = A W_o
    Y_1 = O W_1
    Y_2 = O W_2
    Y = (Y_1 .* Y_2) W_3
    """

    profile = {
        "num_params": 0,
        "num_acts": 0,
        "param_bits": 0,
        "act_bits": 0,
        "flops": 0,
    }
    delta_list = []
    delta_list.append(
        profile_linear_layer(
            layer_quant_config["self_attn"]["q_proj"],
            in_features=hidden_size,
            out_features=hidden_size,
            bias=False,
            batch_size=seq_len,
        )
    )
    delta_list.append(
        profile_linear_layer(
            layer_quant_config["self_attn"]["k_proj"],
            in_features=hidden_size,
            out_features=hidden_size,
            bias=False,
            batch_size=seq_len,
        )
    )
    delta_list.append(
        profile_linear_layer(
            layer_quant_config["self_attn"]["v_proj"],
            in_features=hidden_size,
            out_features=hidden_size,
            bias=False,
            batch_size=seq_len,
        )
    )
    for i in range(num_attention_heads):
        delta_list.append(
            profile_matmul_layer(
                layer_quant_config["self_attn"]["matmul_0"],
                data_in_0_size=(seq_len, hidden_size // num_attention_heads),
                data_in_1_size=(hidden_size // num_attention_heads, seq_len),
            )
        )
        delta_list.append(
            profile_matmul_layer(
                layer_quant_config["self_attn"]["matmul_1"],
                data_in_0_size=(seq_len, seq_len),
                data_in_1_size=(seq_len, hidden_size // num_attention_heads),
            )
        )
    delta_list.append(
        profile_linear_layer(
            layer_quant_config["self_attn"]["o_proj"],
            in_features=hidden_size,
            out_features=hidden_size,
            bias=False,
            batch_size=seq_len,
        )
    )
    delta_list.append(
        profile_linear_layer(
            layer_quant_config["mlp"]["gate_proj"],
            in_features=hidden_size,
            out_features=intermediate_size,
            bias=False,
            batch_size=seq_len,
        )
    )
    delta_list.append(
        profile_linear_layer(
            layer_quant_config["mlp"]["down_proj"],
            in_features=intermediate_size,
            out_features=hidden_size,
            bias=False,
            batch_size=seq_len,
        )
    )
    delta_list.append(
        profile_linear_layer(
            layer_quant_config["mlp"]["up_proj"],
            in_features=hidden_size,
            out_features=intermediate_size,
            bias=False,
            batch_size=seq_len,
        )
    )
    for delta in delta_list:
        update_profile(profile, delta)
    return profile


def profile_llama_quantized(config, seq_len: int):
    """
    Profile llama quantized model

    Args:
        config (LlamaQuantizedConfig): llama quantized config
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
        "flops": 0,
    }

    for i in range(num_hidden_layers):
        layer_quant_config = config.quant_config[f"model_layer_{i}"]
        update_profile(
            profile=profile,
            delta=_profile_llama_layer(
                layer_quant_config,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=config.num_attention_heads,
                seq_len=seq_len,
            ),
        )
    return profile


def _register_stat_hook_opt_layer(
    stat_manager, decoder_layer: LlamaQuantizedDecoderLayer, name: str
):
    hooks_to_register = {
        "self_attn": {
            "q_proj": ["data_in", "weight", "data_out"],
            "k_proj": ["data_in", "weight", "data_out"],
            "v_proj": ["data_in", "weight", "data_out"],
            "o_proj": ["data_in", "weight"],
        },
        "mlp": {
            "gate_proj": ["data_in", "weight"],
            "down_proj": ["data_in", "weight"],
            "up_proj": ["data_in", "weight"],
        },
    }

    self_attn_name = f"{name}:self_attn"
    for layer, entries in hooks_to_register["self_attn"].items():
        for entry in entries:
            entry_name = f"{self_attn_name}:{layer}:{entry}"
            register_a_stat_hook(
                stat_manager=stat_manager,
                name=entry_name,
                module=getattr(decoder_layer.self_attn, layer),
                entry=entry,
            )
    for layer in ["gate_proj", "down_proj", "up_proj"]:
        for entry in hooks_to_register["mlp"][layer]:
            entry_name = f"{name}:mlp:{layer}:{entry}"
            register_a_stat_hook(
                stat_manager,
                name=entry_name,
                module=getattr(decoder_layer.mlp, layer),
                entry=entry,
            )


def register_stat_hooks_llama_quantized(
    stat_manager,
    name: str,
    model: LlamaQuantizedForCausalLM,
    num_hidden_layers: int,
):
    for i in range(num_hidden_layers):
        model_layer = model.model.layers[i]
        layer_name = f"{name}:model_layer_{i}"
        _register_stat_hook_opt_layer(
            stat_manager=stat_manager,
            decoder_layer=model_layer,
            name=layer_name,
        )
