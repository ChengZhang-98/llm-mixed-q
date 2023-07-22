import datasets

TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

GLUE_TASKS = list(TASK_TO_KEYS.keys())


def get_num_labels(task: str):
    assert task in TASK_TO_KEYS, f"task {task} not supported"

    raw_datasets = datasets.load_dataset("glue", task)
    is_regression = task == "stsb"

    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    return num_labels


def get_raw_dataset(task: str) -> datasets.DatasetDict:
    assert task in TASK_TO_KEYS, f"task {task} not supported"
    raw_datasets = datasets.load_dataset("glue", task)
    return raw_datasets


def preprocess_datasets(
    raw_dataset_dict, task: str, split: str, tokenizer, padding, max_length
) -> datasets.DatasetDict:
    assert task in TASK_TO_KEYS, f"task {task} not supported"
    sentence1_key, sentence2_key = TASK_TO_KEYS[task]

    def preprocess_fn(examples):
        texts = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *texts,
            padding=padding,
            max_length=max_length,
            truncation=True,
        )
        result["labels"] = examples["label"]
        return result

    processed_dataset = raw_dataset_dict.map(
        preprocess_fn,
        batched=True,
        remove_columns=raw_dataset_dict["train"].column_names,
        desc="Running tokenizer on dataset",
    )
    train_dataset = processed_dataset["train"]
    val_dataset = processed_dataset[
        "validation_matched" if task == "mnli" else "validation"
    ]
    test_dataset = processed_dataset["test_matched" if task == "mnli" else "test"]
    return datasets.DatasetDict(
        train=train_dataset,
        validation=val_dataset,
        test=test_dataset,
    )
