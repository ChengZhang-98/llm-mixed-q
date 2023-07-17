from ..models import get_config_cls, get_model_cls
from ..tools import set_logging_verbosity, load_config, save_config
import logging

import math
import os
from functools import partial
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from tqdm.auto import tqdm
from transformers import get_scheduler

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


def checkpoint_model(
    accelerator: Accelerator, model: torch.nn.Module, save_dir: str | os.PathLike
):
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.num_processes > 1:
        # gather the checkpoint to rank 0's CPU memory to avoid GPU OOM
        # this requires enough CPU memory
        full_state_dict_config = FullStateDictConfig(
            offload_to_cpu=True, rank0_only=True
        )
        with FSDP.state_dict_type(
            unwrapped_model, StateDictType.FULL_STATE_DICT, full_state_dict_config
        ):
            state = accelerator.get_state_dict(unwrapped_model)
            # unwrapped_model is HuggingFace AutoPretrainedModel, so we can use save_pretrained() to save the checkpoint
            unwrapped_model.save_pretrained(
                save_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=state,
            )
    else:
        unwrapped_model.save_pretrained(
            save_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )
