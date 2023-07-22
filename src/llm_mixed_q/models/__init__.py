from .bert_quantized import (
    BertQuantizedConfig,
    BertQuantizedForSequenceClassification,
    profile_bert_quantized,
    parse_bert_quantized_config,
)
from .llama_quantized import (
    LlamaQuantizedConfig,
    LlamaQuantizedForCausalLM,
    LlamaQuantizedForSequenceClassification,
    profile_llama_quantized,
    parse_llama_quantized_config,
)
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.llama.tokenization_llama import LlamaTokenizer

MODEL_MAP = {
    "bert": {
        "cls": BertQuantizedForSequenceClassification,
    },
    "llama": {
        "cls": LlamaQuantizedForSequenceClassification,
        "lm": LlamaQuantizedForCausalLM,
    },
}

CONFIG_MAP = {
    "bert": BertQuantizedConfig,
    "llama": LlamaQuantizedConfig,
}

TOKENIZER_MAP = {
    "bert": BertTokenizer,
    "llama": LlamaTokenizer,
}

PROFILER_MAP = {
    "bert": profile_bert_quantized,
    "llama": profile_llama_quantized,
}

QUANT_CONFIG_PARSER_MAP = {
    "bert": parse_bert_quantized_config,
    "llama": parse_llama_quantized_config,
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


def get_q_profiler(arch: str):
    assert arch in PROFILER_MAP, f"arch {arch} not supported"
    return PROFILER_MAP[arch]


def get_quant_config_parser(arch: str):
    assert arch in QUANT_CONFIG_PARSER_MAP, f"arch {arch} not supported"
    return QUANT_CONFIG_PARSER_MAP[arch]
