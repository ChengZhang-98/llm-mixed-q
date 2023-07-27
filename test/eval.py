import os
import sys
from pathlib import Path

import torch
from accelerate import init_empty_weights
from transformers import default_data_collator
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
import datasets as hf_datasets

sys.path.append(str(Path(__file__).parent.parent / "src"))
from llm_mixed_q.models import get_model_cls, get_config_cls
from llm_mixed_q.eval import evaluate_lm_task_wikitext2
from llm_mixed_q.datasets import get_raw_dataset_dict, preprocess_dataset_dict


os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"


def test_perplexity():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_datasets.disable_caching()

    task = "wikitext2"
    max_length = 2048
    batch_size = 4
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m").to("cuda")

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

    results = evaluate_lm_task_wikitext2(
        model,
        eval_dataloader,
        num_samples=None,
        progress_bar=True,
        input_device="cuda:0",
    )
    print(results)


if __name__ == "__main__":
    test_perplexity()
