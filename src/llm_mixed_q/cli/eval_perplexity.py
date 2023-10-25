import json
import logging
import os
from argparse import ArgumentParser
from pathlib import Path
import ast

import torch
import psutil
from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

from ..datasets import get_raw_dataset_dict, preprocess_dataset_dict
from ..eval import eval_lm_wikitext2
from ..models import get_config_cls, get_model_cls, get_tokenizer_cls

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


def cli_eval_lm_wikitext2():
    logger.info("Evaluation started")

    parser = ArgumentParser()
    parser.add_argument("--model_arch", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--task", type=str, choices=["wikitext2"], required=True)
    parser.add_argument("--quant_config", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)

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
    if args.quant_config is None:
        args.quant_config = {
            "default": {
                "name": "integer",
                "bypass": True,
            }
        }

    model_cls = get_model_cls(args.model_arch, "lm")
    config_cls = get_config_cls(args.model_arch)

    config = config_cls.from_pretrained(args.model_name, quant_config=args.quant_config)
    tokenizer = get_tokenizer_cls(args.model_arch).from_pretrained(
        args.model_name, legacy=False
    )

    if not args.model_parallelism:
        model = model_cls.from_pretrained(args.model_name, config=config).to("cuda")
    else:
        model = model_cls.from_pretrained(args.model_name, config=config)
        if args.device_map == "auto":
            device_map = infer_auto_device_map(
                model,
                no_split_module_classes=model._no_split_modules,
                dtype=torch.float16,
            )
        else:
            device_map = ast.literal_eval(args.device_map)
        model = dispatch_model(model, device_map=device_map)

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
