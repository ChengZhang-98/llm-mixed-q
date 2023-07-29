from typing import List, Optional, Union
import logging
from pprint import pformat

import torch
import transformers
from transformers import AutoModel

from lm_eval.models.huggingface import (
    AutoCausalLM,
    BaseLM,
    _DeviceMapping,
    _get_accelerate_args,
)
from lm_eval.models import MODEL_REGISTRY
from ...models import (
    get_config_cls,
    get_quant_config_parser,
    get_model_cls,
    get_tokenizer_cls,
)

logger = logging.getLogger(__name__)


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
        quant_config: str | dict | None = None,
        # pretrained: str,
        # quantized: bool | str | None = False,
        tokenizer: str | None = None,
        subfolder: str | None = None,
        revision: str | None = "main",
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
        # dtype: str | dtype | None = None,
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

        self.AUTO_CONFIG_CLASS = get_config_cls(model_arch)
        self.AUTO_TOKENIZER_CLASS = get_tokenizer_cls(model_arch)
        self.AUTO_MODEL_CLASS = get_model_cls(model_arch, "lm")

        # setup for automatic batch size detection
        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self._batch_size = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self._batch_size = int(batch_size)
        self.max_batch_size = max_batch_size

        self._max_gen_toks = max_gen_toks
        self._max_length = max_length
        self._config = self.AUTO_CONFIG_CLASS.from_pretrained(
            model_name,
            quant_config=quant_config,
            trust_remote_code=trust_remote_code,
        )
        self.model_arch = model_arch
        self.model_name = model_name

        self._add_special_tokens = add_special_tokens
        self.tokenizer = self._create_auto_tokenizer(
            pretrained=model_name,
            revision=revision,
            subfolder=subfolder,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer.model_max_length = self.max_length

        model_kwargs = {}

        if use_accelerate:
            model_kwargs = _get_accelerate_args(
                device_map_option=device_map_option,
                max_memory_per_gpu=max_memory_per_gpu,
                max_cpu_memory=max_cpu_memory,
                offload_folder=offload_folder,
            )
        self.model = self._create_auto_model(
            revision=revision,
            subfolder=subfolder,
            **model_kwargs,
        )
        self.model.eval()
        torch.set_grad_enabled(False)

        self._device = device
        if use_accelerate and "lm_head" in self.model.hf_device_map:
            # `accelerate` can place `lm_head` weights on a different device than
            # the user specified one so we force `self._device` to be the same as
            # `lm_head`'s.
            self._device = self.model.hf_device_map["lm_head"]
        if not use_accelerate:
            try:
                self.model.to(self._device)
            except:
                print(
                    "Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes`. If the desired GPU is being used, this message is safe to ignore."
                )

    def _create_auto_model(
        self,
        *,
        # quantized: bool | str | None = False,
        revision: str,
        subfolder: str,
        device_map: str | _DeviceMapping | None = None,
        max_memory: dict | None = None,
        offload_folder: str | None = None,
        # load_in_8bit: bool | None = False,
        # load_in_4bit: bool | None = False,
        trust_remote_code: bool | None = False,
        # torch_dtype: str | dtype | None = None,
        # gptq_use_triton: bool | None = False,
        # bnb_4bit_quant_type: str | None = None,
        # bnb_4bit_compute_dtype: str | dtype | None = None
    ) -> AutoModel:
        model = self.AUTO_MODEL_CLASS.from_pretrained(
            self.model_name,
            config=self._config,
            revision=revision + ("/" + subfolder if subfolder is not None else ""),
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=offload_folder,
            trust_remote_code=trust_remote_code,
        )
        # logger.debug(
        #     f"============= Model.config.quant_config (layer 0.k_proj) ============="
        # )
        # logger.debug(
        #     "\n"
        #     + pformat(
        #         model.config.quant_config["model_layer_0"]["self_attn"].get(
        #             "k_proj", "failed to get k_proj"
        #         )
        #     )
        # )
        # logger.debug(str(model.model.decoder.layers[0]))
        return model

    def _create_auto_tokenizer(
        self,
        *,
        pretrained: str,
        revision: str,
        subfolder: str,
        tokenizer: Optional[str] = None,
        trust_remote_code: Optional[bool] = False,
    ) -> transformers.PreTrainedTokenizer:
        """Returns a pre-trained tokenizer from a pre-trained tokenizer configuration."""
        tokenizer = self.AUTO_TOKENIZER_CLASS.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            revision=revision + ("/" + subfolder if subfolder is not None else ""),
            trust_remote_code=trust_remote_code,
            legacy=False,
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @property
    def add_special_tokens(self) -> bool:
        """Whether to include special tokens in encoded text. This should be
        determined by whether or not the model was trained with special tokens.
        TODO: Remove these conditionals once HuggingFace supports a way to
        check whether or not an arbitrary model was trained with special tokens.
        """
        if self._add_special_tokens is not None:
            return self._add_special_tokens
        elif "ForCausalLM" in self.AUTO_MODEL_CLASS.__name__:
            return False
        # elif self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM:
        elif "ForSeq2SeqLM" in self.AUTO_MODEL_CLASS.__name__:
            return True
        else:
            raise ValueError(
                "Could not determine `add_special_tokens` value from the model "
                "class. Set to `True` or `False` depending on whether the model "
                "was pre-trained with special tokens."
            )
