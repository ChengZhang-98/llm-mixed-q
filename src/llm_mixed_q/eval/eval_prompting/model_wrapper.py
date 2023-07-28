from typing import List, Optional, Union

import torch
import transformers
from transformers import AutoModel

from ...lm_eval.models.huggingface import AutoCausalLM
from ...lm_eval.models import MODEL_REGISTRY
from ...lm_eval.models.huggingface import _DeviceMapping, dtype


class QuantizedCausalLM(AutoCausalLM):
    """LLM-Mixed-Q's Quantized causal language modeling."""

    def _create_auto_model(
        self,
        *,
        pretrained: str,
        quantized: bool | str | None = False,
        revision: str,
        subfolder: str,
        device_map: str | _DeviceMapping | None = None,
        max_memory: dict | None = None,
        offload_folder: str | None = None,
        load_in_8bit: bool | None = False,
        load_in_4bit: bool | None = False,
        trust_remote_code: bool | None = False,
        torch_dtype: str | torch.dtype | None = None,
        gptq_use_triton: bool | None = False,
        bnb_4bit_quant_type: str | None = None,
        bnb_4bit_compute_dtype: str | torch.dtype | None = None
    ) -> AutoModel:
        return super()._create_auto_model(
            pretrained=pretrained,
            quantized=quantized,
            revision=revision,
            subfolder=subfolder,
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=offload_folder,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            gptq_use_triton=gptq_use_triton,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        )
