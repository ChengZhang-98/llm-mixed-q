import os

import datasets as hf_datasets


def get_raw_dataset_dict() -> hf_datasets.DatasetDict:
    dataset_dict = hf_datasets.load_dataset("wikitext", "wikitext-2-raw-v1")
    return dataset_dict


def preprocess_dataset_dict(
    raw_dataset_dict,
    tokenizer,
    max_length,
) -> hf_datasets.DatasetDict:
    if tokenizer.pad_token in ["<unk>", None]:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(["  ".join(examples["text"])])

    encodings = raw_dataset_dict.map(
        tokenize_function,
        batched=True,
        remove_columns=raw_dataset_dict["train"].column_names,
        desc="Running tokenizer on dataset",
        num_proc=os.cpu_count() // 2,
    )

    def group_texts(examples):
        # Concatenate all texts.
        # >>> sum([[1,2,3],[4,5,6]],[])
        # [1, 2, 3, 4, 5, 6]
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= max_length:
            total_length = (total_length // max_length) * max_length
        # Split by chunks of block_size.
        result = {
            k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    preprocessed = encodings.map(
        group_texts,
        batched=True,
        num_proc=os.cpu_count() // 2,
        desc="Grouping texts",
    )

    return preprocessed
