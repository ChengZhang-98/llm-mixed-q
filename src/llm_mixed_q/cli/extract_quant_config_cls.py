import json
import os
from argparse import ArgumentParser
from pathlib import Path
from pprint import pformat

from torch.utils.data import DataLoader
from transformers import default_data_collator

from ..datasets import (get_raw_dataset_dict, is_regression_task,
                        preprocess_dataset_dict)
from ..eval import evaluate_cls_glue
from ..models import get_config_cls, get_model_cls, get_tokenizer_cls
from ..utils import extract_quant_config
from ..utils.logger import get_logger

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def cls_extract_quant_config_and_eval_cls_glue():
    logger = get_logger(__name__)

    parser = ArgumentParser()
    parser.add_argument("--study", type=str, required=True, help="Path to study.pkl")
    parser.add_argument("--trial", type=int, required=True, help="Target trial index")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--model_arch", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)

    parser.add_argument("--max_length", type=int, default=196)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    quant_config_path = None
    results_path = None
    if args.save_dir is not None:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        quant_config_path = save_dir / "quant_config.toml"
        results_path = save_dir / "results.json"

    args = parser.parse_args()
    logger.info("============== Extracting quant config ==============")
    logger.info(pformat(vars(args)))

    quant_config = extract_quant_config(args.study, args.trial, quant_config_path)
    if quant_config_path is not None:
        logger.info(f"Saved quant config to {quant_config_path}")

    logger.info("============== Evaluating on GLUE ==============")

    tokenizer = get_tokenizer_cls(args.model_arch).from_pretrained(
        args.model_name, legacy=False
    )
    config = get_config_cls(args.model_arch).from_pretrained(
        args.model_name, quant_config=quant_config
    )
    model = (
        get_model_cls(args.model_arch, task="cls")
        .from_pretrained(args.model_name, config=config)
        .to("cuda")
    )
    raw_dataset_dict = get_raw_dataset_dict(args.task)
    preprocessed_dataset_dict = preprocess_dataset_dict(
        raw_dataset_dict,
        task=args.task,
        tokenizer=tokenizer,
        padding="max_length",
        max_length=args.max_length,
    )

    eval_dataloader = DataLoader(
        preprocessed_dataset_dict["validation"],
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        shuffle=False,
    )
    results = evaluate_cls_glue(
        model,
        task=args.task,
        eval_dataloader=eval_dataloader,
        is_regression=is_regression_task(args.task),
        num_samples=None,
        progress_bar=True,
    )

    if results_path is not None:
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Saved results to {results_path}")

    logger.info(pformat(results))
    logger.info("============== Done ==============")


if __name__ == "__main__":
    cls_extract_quant_config_and_eval_cls_glue()
