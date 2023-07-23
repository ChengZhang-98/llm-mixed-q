import datasets
from .glue import (
    get_num_labels as get_num_labels_glue,
    get_raw_dataset as get_raw_dataset_dict_glue,
    preprocess_datasets as preprocess_dataset_dict_glue,
    GLUE_TASKS,
)


def get_num_labels(task: str):
    if task in GLUE_TASKS:
        return get_num_labels_glue(task)
    else:
        raise ValueError(f"task {task} not supported")


def get_raw_dataset_dict(task: str) -> datasets.DatasetDict:
    if task in GLUE_TASKS:
        return get_raw_dataset_dict_glue(task)
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
    else:
        raise ValueError(f"task {task} not supported")


def is_regression_task(task: str) -> bool:
    if task in GLUE_TASKS:
        return task == "stsb"
    else:
        raise ValueError(f"task {task} not supported")
