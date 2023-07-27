import os
import sys
from pathlib import Path

import torch
from accelerate import init_empty_weights
import datasets as hf_datasets

sys.path.append(str(Path(__file__).parent.parent / "src"))
from llm_mixed_q.models import get_model_cls, get_config_cls, get_tokenizer_cls
from llm_mixed_q.datasets import get_raw_dataset_dict, preprocess_dataset_dict

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"


def test_wikitext2():
    hf_datasets.disable_caching()

    arch = "llama"
    name = "Cheng98/llama-160m"

    tokenizer = get_tokenizer_cls(arch).from_pretrained(name)

    wikitext2 = get_raw_dataset_dict("wikitext2")
    breakpoint()

    wikitext2 = preprocess_dataset_dict(
        wikitext2,
        task="wikitext2",
        tokenizer=tokenizer,
        padding="max_length",
        max_length=2048,
    )

    breakpoint()


if __name__ == "__main__":
    test_wikitext2()
