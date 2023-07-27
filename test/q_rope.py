import os
import sys
from pathlib import Path

import os
from pathlib import Path

from transformers import default_data_collator
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent / "src"))

from llm_mixed_q.search.estimator.software_metrics import evaluate_cls_task
from llm_mixed_q.utils import set_logging_verbosity

from llm_mixed_q.models import (
    get_model_cls,
    get_config_cls,
    get_tokenizer_cls,
)
from llm_mixed_q.datasets import (
    get_num_labels,
    get_raw_dataset_dict,
    preprocess_dataset_dict,
    is_regression_task,
)

import torch


os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"


def test_bert():
    arch = "bert"
    task = "cls"
    name = "bert-base-uncased"
    quant_config = "./bert.toml"

    model_cls = get_model_cls(arch, task)
    config_cls = get_config_cls(arch)

    config = config_cls.from_pretrained(name, quant_config=quant_config)
    model = model_cls.from_pretrained(name, config=config, device_map=None)

    x = torch.randint(0, 1000, (16, 128))
    y = model(x)
    print(model.bert.encoder.layer[0].attention.self.query.weight[0, :10])


def test_llama():
    arch = "llama"
    task = "sst2"
    name = str(
        Path(__file__).parent.parent
        / "checkpoints"
        / "asplos"
        / "fine_tune"
        / "llama_160m_sst2"
    )
    quant_config = "./llama_q_rope.toml"

    num_labels = get_num_labels(task)

    model_cls = get_model_cls(arch, "cls")
    config_cls = get_config_cls(arch)

    config = config_cls.from_pretrained(
        name, quant_config=quant_config, num_labels=num_labels
    )
    model = model_cls.from_pretrained(name, config=config, device_map=None)
    tokenizer = get_tokenizer_cls(arch).from_pretrained(name)

    raw_dataset_dict = get_raw_dataset_dict(task)
    preprocessed_dataset_dict = preprocess_dataset_dict(
        raw_dataset_dict,
        task=task,
        tokenizer=tokenizer,
        padding="max_length",
        max_length=196,
    )
    is_regression = is_regression_task(task)
    eval_dataloader = DataLoader(
        preprocessed_dataset_dict["validation"],
        batch_size=16,
        collate_fn=default_data_collator,
        num_workers=os.cpu_count(),
        shuffle=False,
    )

    model = model.to("cuda")
    results = evaluate_cls_task(
        model,
        task=task,
        eval_dataloader=eval_dataloader,
        is_regression=is_regression,
        progress_bar=True,
    )
    print(results)


"""
SST2
====================
FP32 baseline:
{'accuracy': 0.8979357798165137}

====================
Quantize RoPE's sin and cos only
[rotary_positional_encoding]
bypass = false
name = "integer"
data_in_width = x
data_in_frac_width = y

Integer (x, y)
--------------------
*: This result looks weird.
Integer (8, 7): {'accuracy': 0.8979357798165137}
Integer (7, 6): {'accuracy': 0.8990825688073395}
Integer (6, 5): {'accuracy': 0.9002293577981652}
Integer (5, 4): {'accuracy': 0.9002293577981652}
Integer (4, 3): {'accuracy': 0.9036697247706422}
Integer (3, 2): {'accuracy': 0.8669724770642202}
"""


if __name__ == "__main__":
    # test_bert()
    set_logging_verbosity("debug")
    test_llama()
