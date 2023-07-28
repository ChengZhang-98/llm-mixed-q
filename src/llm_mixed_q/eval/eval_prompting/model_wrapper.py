from typing import List, Optional, Union

import torch
import transformers
from transformers import AutoModel

from ...lm_eval.models.huggingface import AutoCausalLM, HuggingFaceAutoLM, BaseLM
from ...lm_eval.models import MODEL_REGISTRY
from ...lm_eval.models.huggingface import _DeviceMapping, dtype


class QuantizedCausalLM(AutoCausalLM):
    """LLM-Mixed-Q's Quantized causal language modeling."""

    AUTO_CONFIG_CLASS = None
    AUTO_TOKENIZER_CLASS = None
    AUTO_MODEL_CLASS = None
    AUTO_PEFT_CLASS = None

    def __init__(
        self,
        model_arch: str,
        model_name: str,
        # pretrained: str,
        # quantized: bool | str | None = False,
        # tokenizer: str | None = None,
        # subfolder: str | None = None,
        # revision: str | None = "main",
        batch_size: int | str | None = 1,
        max_batch_size: int | None = 512,
        max_gen_toks: int | None = 256,
        max_length: int | None = None,
        add_special_tokens: bool | None = None,
        use_accelerate: bool | None = False,
        device_map_option: str | None = "auto",
        max_memory_per_gpu: int | str | None = None,
        max_cpu_memory: int | str | None = None,
        offload_folder: str | None = "./offload",
        dtype: str | dtype | None = None,
        device: int | str | None = "cuda",
        peft: str = None,
        # load_in_8bit: bool | None = False,
        # load_in_4bit: bool | None = False,
        trust_remote_code: bool | None = False,
        # gptq_use_triton: bool | None = False,
        # bnb_4bit_quant_type: str | None = None,
        # bnb_4bit_compute_dtype: str | dtype | None = None,
    ):
        BaseLM.__init__(self)

        super().__init__(
            pretrained,
            quantized,
            tokenizer,
            subfolder,
            revision,
            batch_size,
            max_batch_size,
            max_gen_toks,
            max_length,
            add_special_tokens,
            use_accelerate,
            device_map_option,
            max_memory_per_gpu,
            max_cpu_memory,
            offload_folder,
            dtype,
            device,
            peft,
            load_in_8bit,
            load_in_4bit,
            trust_remote_code,
            gptq_use_triton,
            bnb_4bit_quant_type,
            bnb_4bit_compute_dtype,
        )
