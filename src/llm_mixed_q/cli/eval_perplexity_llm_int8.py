import ast
import json
import logging
import os
from argparse import ArgumentParser
from pathlib import Path

from torch.utils.data import DataLoader
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling)

from ..datasets import get_raw_dataset_dict, preprocess_dataset_dict
from ..eval import eval_lm_wikitext2

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


def cli_eval_lm_wikitext2_llm_int8():
    logger.info("Evaluation started")

    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--task", type=str, choices=["wikitext2"], required=True)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--load_in_n_bit", type=str, choices=["8", "4"], required=True)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--model_parallelism", action="store_true")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
    )
    args = parser.parse_args()
    if args.load_in_n_bit == "8":
        args.load_in_8bit = True
        args.load_in_4bit = False
    elif args.load_in_n_bit == "4":
        args.load_in_8bit = False
        args.load_in_4bit = True
    else:
        raise ValueError(f"Invalid value for args.load_in_n_bit: {args.load_in_n_bit}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, legacy=False)

    if not args.model_parallelism:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
        )
    else:
        if args.device_map == "auto":
            device_map = "auto"
        else:
            device_map = ast.literal_eval(args.device_map)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
            device_map=device_map,
        )

    raw_dataset = get_raw_dataset_dict(args.task)
    preprocessed_dataset_dict = preprocess_dataset_dict(
        raw_dataset,
        task=args.task,
        tokenizer=tokenizer,
        padding="max_length",
        max_length=args.max_length,
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    eval_dataloader = DataLoader(
        preprocessed_dataset_dict[args.dataset_split],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=os.cpu_count(),
    )

    results = eval_lm_wikitext2(
        model,
        eval_dataloader=eval_dataloader,
        num_samples=None,
        progress_bar=True,
        input_device="cuda:0",
    )

    logger.info(results)

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "results.json", "w") as f:
            json.dump(results, f, indent=4)

    logger.info("Evaluation finished")


# if __name__ == "__main__":

#     set_logging_verbosity("info")
#     eval_perplexity_wikitext_runner()
