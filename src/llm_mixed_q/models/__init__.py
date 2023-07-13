from .bert_quantized import BertQuantizedConfig, BertQuantizedForSequenceClassification
from .llama_quantized import (
    LlamaQuantizedConfig,
    LlamaQuantizedForCausalLM,
    LlamaQuantizedForSequenceClassification,
)

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


def get_model_cls(arch: str, task: str):
    assert arch in MODEL_MAP, f"arch {arch} not supported"
    assert task in MODEL_MAP[arch], f"task {task} not supported for arch {arch}"
    return MODEL_MAP[arch][task]


def get_config_cls(arch: str):
    assert arch in CONFIG_MAP, f"arch {arch} not supported"
    return CONFIG_MAP[arch]
