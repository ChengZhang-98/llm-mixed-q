from transformers import AutoTokenizer
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.llama.tokenization_llama import LlamaTokenizer

from .bert_quantized import (BertQuantizedConfig,
                             BertQuantizedForSequenceClassification,
                             format_stat_profiled_int_config_bert_quantized,
                             parse_bert_quantized_config,
                             profile_bitwidth_bert_quantized,
                             register_stat_hooks_bert_quantized,
                             sample_bert_quant_config)
from .llama_quantized import (LlamaQuantizedConfig, LlamaQuantizedForCausalLM,
                              LlamaQuantizedForSequenceClassification,
                              format_stat_profiled_int_config_llama_quantized,
                              parse_llama_quantized_config,
                              profile_bitwidth_llama_quantized,
                              register_stat_hooks_llama_quantized,
                              sample_llama_quant_config)
from .opt_quantized import (OPTQuantizedConfig, OPTQuantizedForCausalLM,
                            OPTQuantizedForSequenceClassification,
                            format_stat_profiled_int_config_opt_quantized,
                            parse_opt_quantized_config,
                            profile_bitwidth_opt_quantized,
                            register_stat_hooks_opt_quantized,
                            sample_opt_quant_config)

MODEL_MAP = {
    "bert": {
        "cls": BertQuantizedForSequenceClassification,
    },
    "llama": {
        "cls": LlamaQuantizedForSequenceClassification,
        "lm": LlamaQuantizedForCausalLM,
    },
    "opt": {
        "cls": OPTQuantizedForSequenceClassification,
        "lm": OPTQuantizedForCausalLM,
    },
}

CONFIG_MAP = {
    "bert": BertQuantizedConfig,
    "llama": LlamaQuantizedConfig,
    "opt": OPTQuantizedConfig,
}

TOKENIZER_MAP = {
    "bert": BertTokenizer,
    "llama": LlamaTokenizer,
    "opt": AutoTokenizer,
}

BITWIDTH_PROFILER_MAP = {
    "bert": profile_bitwidth_bert_quantized,
    "llama": profile_bitwidth_llama_quantized,
    "opt": profile_bitwidth_opt_quantized,
}

QUANT_CONFIG_PARSER_MAP = {
    "bert": parse_bert_quantized_config,
    "llama": parse_llama_quantized_config,
    "opt": parse_opt_quantized_config,
}

QUANT_CONFIG_SAMPLER_MAP = {
    "bert": sample_bert_quant_config,
    "llama": sample_llama_quant_config,
    "opt": sample_opt_quant_config,
}

STAT_PROFILER_HOOK_MAP = {
    "bert": register_stat_hooks_bert_quantized,
    "llama": register_stat_hooks_llama_quantized,
    "opt": register_stat_hooks_opt_quantized,
}

STAT_CONFIG_FORMATTER_MAP = {
    "bert": format_stat_profiled_int_config_bert_quantized,
    "llama": format_stat_profiled_int_config_llama_quantized,
    "opt": format_stat_profiled_int_config_opt_quantized,
}


def get_model_cls(arch: str, task: str):
    assert arch in MODEL_MAP, f"arch {arch} not supported"
    assert task in MODEL_MAP[arch], f"task {task} not supported for arch {arch}"
    return MODEL_MAP[arch][task]


def get_config_cls(arch: str):
    assert arch in CONFIG_MAP, f"arch {arch} not supported"
    return CONFIG_MAP[arch]


def get_tokenizer_cls(arch: str):
    assert arch in TOKENIZER_MAP, f"arch {arch} not supported"
    return TOKENIZER_MAP[arch]


def get_bitwidth_profiler(arch: str):
    assert arch in BITWIDTH_PROFILER_MAP, f"arch {arch} not supported"
    return BITWIDTH_PROFILER_MAP[arch]


def get_quant_config_parser(arch: str):
    assert arch in QUANT_CONFIG_PARSER_MAP, f"arch {arch} not supported"
    return QUANT_CONFIG_PARSER_MAP[arch]


def get_quant_config_sampler(arch: str):
    assert arch in QUANT_CONFIG_SAMPLER_MAP, f"arch {arch} not supported"
    return QUANT_CONFIG_SAMPLER_MAP[arch]


def get_stat_profiler_hook(arch: str):
    assert arch in STAT_PROFILER_HOOK_MAP, f"arch {arch} not supported"
    return STAT_PROFILER_HOOK_MAP[arch]


def get_stat_config_formatter(arch: str):
    assert arch in STAT_CONFIG_FORMATTER_MAP, f"arch {arch} not supported"
    return STAT_CONFIG_FORMATTER_MAP[arch]
