import logging

from ..quantize.quantized_layer_profiler import (
    profile_linear_layer,
    profile_matmul_layer,
    register_a_stat_hook,
    update_profile,
)
from .modeling_bert import BertQuantizedForSequenceClassification, BertQuantizedLayer

logger = logging.getLogger(__name__)


def log_avg_bitwidth(profile, tag=None):
    logger.debug(f"Profiler tag: {tag}")
    logger.debug(
        f"{tag} avg param bitwidth: {profile['param_bits'] / profile['num_params']}"
    )
    logger.debug(f"{tag} avg act bitwidth: {profile['act_bits'] / profile['num_acts']}")
    return profile


def _profile_bert_attention_layer(
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
        "flops": 0,
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


def _profile_bert_layer(
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
        "flops": 0,
    }

    delta_list = []
    delta_list.append(
        _profile_bert_attention_layer(
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


def profile_bert_quantized(config, seq_len: int):
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
        "flops": 0,
    }

    for i in range(num_hidden_layers):
        layer_quant_config = config.quant_config[f"model_layer_{i}"]
        update_profile(
            profile,
            _profile_bert_layer(
                layer_quant_config,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=config.num_attention_heads,
                seq_len=seq_len,
            ),
        )

    return profile


# ================================================================
# Register statistic profiler hooks
# ================================================================


def _register_stat_hook_bert_layer(
    stat_manager, decoder_layer: BertQuantizedLayer, name: str
):
    hooks_to_register = {
        "attention": {
            "query": ["data_in", "weight", "bias", "data_out"],
            "key": ["data_in", "weight", "bias", "data_out"],
            "value": ["data_in", "weight", "bias", "data_out"],
            "output": {"dense": ["data_in", "weight", "bias"]},
        },
        "intermediate": {"dense": ["data_in", "weight", "bias"]},
        "output": {"dense": ["data_in", "weight", "bias"]},
    }

    attn_name = f"{name}:attention"
    for layer, entries in hooks_to_register["attention"].items():
        if layer != "output":
            for entry in entries:
                entry_name = f"{attn_name}:{layer}:{entry}"
                register_a_stat_hook(
                    stat_manager,
                    name=entry_name,
                    module=getattr(decoder_layer.attention.self, layer),
                    entry=entry,
                )
        else:
            for entry in hooks_to_register["attention"]["output"]["dense"]:
                entry_name = f"{attn_name}:output:dense:{entry}"
                register_a_stat_hook(
                    stat_manager,
                    name=entry_name,
                    module=decoder_layer.attention.output.dense,
                    entry=entry,
                )

    # ffn.fc1
    for entry in hooks_to_register["intermediate"]["dense"]:
        entry_name = f"{name}:intermediate:dense:{entry}"
        register_a_stat_hook(
            stat_manager,
            name=entry_name,
            module=decoder_layer.intermediate.dense,
            entry=entry,
        )
    # ffn.fc2
    for entry in hooks_to_register["output"]["dense"]:
        entry_name = f"{name}:output:dense:{entry}"
        register_a_stat_hook(
            stat_manager,
            name=entry_name,
            module=decoder_layer.output.dense,
            entry=entry,
        )


def register_stat_hooks_bert_quantized(
    stat_manager,
    name: str,
    model: BertQuantizedForSequenceClassification,
    num_hidden_layers: int,
):
    for i in range(num_hidden_layers):
        model_layer = model.bert.encoder.layer[i]
        layer_name = f"{name}:model_layer_{i}"
        _register_stat_hook_bert_layer(
            stat_manager=stat_manager,
            decoder_layer=model_layer,
            name=layer_name,
        )
