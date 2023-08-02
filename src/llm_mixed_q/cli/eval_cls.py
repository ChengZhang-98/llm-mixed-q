from pathlib import Path
import os
from pprint import pformat
from argparse import ArgumentParser
import json
import logging
from torch.utils.data import DataLoader
from transformers import default_data_collator


from ..models import (
    get_model_cls,
    get_config_cls,
    get_tokenizer_cls,
)
from ..datasets import (
    get_raw_dataset_dict,
    preprocess_dataset_dict,
    is_regression_task,
)
from ..eval import evaluate_cls_glue_fn

logger = logging.getLogger(__name__)


def eval_cls_runner():
    parser = ArgumentParser()

    parser.add_argument(
        "--model_arch",
        help="model architecture",
        choices=["bert", "opt", "llama"],
    )
    parser.add_argument("--model_name", required=True, help="model name")
    parser.add_argument(
        "--quant_config", required=True, help="path to quant config file"
    )
    parser.add_argument(
        "--task", required=True, choices=["sst2"], help="task to evaluate on"
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=196)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument(
        "--dataset_split", default="validation", choices=["validation", "test", "train"]
    )
    args = parser.parse_args()

    logger.info(f"========== Running eval_cls ==========")
    logger.info(pformat(vars(args)))

    config = get_config_cls(args.model_arch).from_pretrained(
        args.model_name, quant_config=args.quant_config
    )
    model = (
        get_model_cls(args.model_arch, "cls")
        .from_pretrained(args.model_name, config=config)
        .to("cuda")
    )
    logger.debug(model)
    tokenizer = get_tokenizer_cls(args.model_arch).from_pretrained(
        args.model_name, legacy=False
    )

    raw_dataset_dict = get_raw_dataset_dict(args.task)
    preprocessed_dataset_dict = preprocess_dataset_dict(
        raw_dataset_dict,
        args.task,
        tokenizer=tokenizer,
        padding="max_length",
        max_length=args.max_length,
    )
    is_regression = is_regression_task(args.task)
    search_dataloader = DataLoader(
        preprocessed_dataset_dict[args.dataset_split],
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        num_workers=os.cpu_count(),
        shuffle=False,
    )

    results = evaluate_cls_glue_fn(
        model=model,
        task=args.task,
        eval_dataloader=search_dataloader,
        is_regression=is_regression,
        progress_bar=True,
    )

    dumped = json.dumps(results, indent=2)
    logger.info(dumped)

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "results.json"
        with open(save_path, "w") as f:
            f.write(dumped)
