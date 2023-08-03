import os
import sys
from pathlib import Path
import logging

import torch
from accelerate import init_empty_weights
from transformers import default_data_collator
from transformers import DataCollatorForLanguageModeling
from transformers.utils.logging import set_verbosity_error as set_hf_verbosity_error
from torch.utils.data import DataLoader
import datasets as hf_datasets

sys.path.append(str(Path(__file__).parent.parent / "src"))
from llm_mixed_q.models import get_model_cls, get_config_cls, get_tokenizer_cls
from llm_mixed_q.eval import eval_lm_wikitext2
from llm_mixed_q.datasets import get_raw_dataset_dict, preprocess_dataset_dict
from llm_mixed_q.utils import set_logging_verbosity


os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
set_hf_verbosity_error()
logger = logging.getLogger(__name__)


def test_perplexity():
    """
    ==============
    OPT-125M, Wikitext2
    FP32: {'loss': 3.249823993649976, 'perplexity': 25.785801053044537, 'num_samples': 116, 'seq_len': 2048, 'batch_size': 2}
    8-bit BFP: {'loss': 3.2512285709381104, 'perplexity': 25.82204465106874, 'num_samples': 116, 'seq_len': 2048, 'batch_size': 2}
    """
    hf_datasets.disable_caching()

    task = "wikitext2"
    max_length = 2048
    batch_size = 2
    model_arch = "opt"
    model_name = "facebook/opt-125m"
    # quant_config = "./opt.toml"
    quant_config = None
    device = "cuda:0"

    tokenizer = get_tokenizer_cls(model_arch).from_pretrained(model_name)
    config = get_config_cls(model_arch).from_pretrained(
        model_name, quant_config=quant_config
    )
    model = (
        get_model_cls(model_arch, task="lm")
        .from_pretrained(model_name, config=config)
        .to(device)
    )
    logger.info("Model:\n", model)

    wikitext2 = get_raw_dataset_dict(task)
    wikitext2 = preprocess_dataset_dict(
        wikitext2,
        task=task,
        tokenizer=tokenizer,
        padding="max_length",
        max_length=max_length,
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    eval_dataloader = DataLoader(
        wikitext2["validation"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=os.cpu_count(),
    )

    results = eval_lm_wikitext2(
        model,
        eval_dataloader,
        num_samples=None,
        progress_bar=True,
        input_device="cuda:0",
    )
    print(results)


if __name__ == "__main__":
    test_perplexity()
