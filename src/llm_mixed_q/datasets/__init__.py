import logging

import datasets

from .glue import GLUE_TASKS
from .glue import get_num_labels as get_num_labels_glue
from .glue import get_raw_dataset_dict as get_raw_dataset_dict_glue
from .glue import preprocess_dataset_dict as preprocess_dataset_dict_glue
from .wikitext2 import get_raw_dataset_dict as get_raw_dataset_dict_wikitext2
from .wikitext2 import \
    preprocess_dataset_dict as preprocess_dataset_dict_wikitext2

logger = logging.getLogger(__name__)


def get_num_labels(task: str):
    if task in GLUE_TASKS:
        return get_num_labels_glue(task)
    elif task == "wikitext2":
        logger.warning(
            "returning None for num_labels for language modeling dataset wikitext2"
        )
        return None
    else:
        raise ValueError(f"task {task} not supported")


def get_raw_dataset_dict(task: str) -> datasets.DatasetDict:
    if task in GLUE_TASKS:
        return get_raw_dataset_dict_glue(task)
    elif task == "wikitext2":
        return get_raw_dataset_dict_wikitext2()
    else:
        raise ValueError(f"task {task} not supported")


def preprocess_dataset_dict(
    raw_dataset_dict, task: str, tokenizer, padding, max_length
) -> datasets.DatasetDict:
    if task in GLUE_TASKS:
        return preprocess_dataset_dict_glue(
            raw_dataset_dict,
            task=task,
            tokenizer=tokenizer,
            padding=padding,
            max_length=max_length,
        )
    elif task == "wikitext2":
        return preprocess_dataset_dict_wikitext2(
            raw_dataset_dict,
            tokenizer=tokenizer,
            max_length=max_length,
        )
    else:
        raise ValueError(f"task {task} not supported")


def is_regression_task(task: str) -> bool:
    if task in GLUE_TASKS:
        return task == "stsb"
    elif task == "wikitext2":
        return False
    else:
        raise ValueError(f"task {task} not supported")
