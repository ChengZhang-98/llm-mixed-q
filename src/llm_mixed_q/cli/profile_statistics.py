import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from pprint import pformat

import toml
from accelerate import dispatch_model, infer_auto_device_map
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, default_data_collator

from ..datasets import (get_raw_dataset_dict, is_regression_task,
                        preprocess_dataset_dict)
from ..models import (get_config_cls, get_model_cls, get_stat_profiler_hook,
                      get_tokenizer_cls)
from ..statstic_profiler import (profile_statistics_cls_glue,
                                 profile_statistics_lm_fn)

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


def cli_profile_statistics_cls_glue():
    parser = ArgumentParser()
    parser.add_argument("--model_arch", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--quant_config", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--max_length", type=int, required=True)

    parser.add_argument("--act_stats", type=str, nargs="+", required=True)
    parser.add_argument("--weight_stats", type=str, nargs="+", required=True)

    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--model_parallelism", action="store_true")

    args = parser.parse_args()

    logging.info("==================== Profile Statistics ====================")

    model_cls = get_model_cls(args.model_arch, "cls")
    config_cls = get_config_cls(args.model_arch)

    config = config_cls.from_pretrained(args.model_name, quant_config=args.quant_config)
    tokenizer = get_tokenizer_cls(args.model_arch).from_pretrained(
        args.model_name, legacy=False
    )
    if not args.model_parallelism:
        model = model_cls.from_pretrained(args.model_name, config=config).to("cuda")
    else:
        model = model_cls.from_pretrained(args.model_name, config=config)
        device_map = infer_auto_device_map(
            model, no_split_module_classes=model._no_split_modules
        )
        model = dispatch_model(model, device_map=device_map)

    raw_dataset_dict = get_raw_dataset_dict(args.task)
    preprocessed_dataset_dict = preprocess_dataset_dict(
        raw_dataset_dict,
        task=args.task,
        tokenizer=tokenizer,
        padding="max_length",
        max_length=args.max_length,
    )

    eval_dataloader = DataLoader(
        preprocessed_dataset_dict["train"],
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        num_workers=os.cpu_count(),
        shuffle=False,
    )

    results = profile_statistics_cls_glue(
        act_stats=args.act_stats,
        weight_stats=args.weight_stats,
        hook_registration_fn=get_stat_profiler_hook(args.model_arch),
        model=model,
        task=args.task,
        eval_dataloader=eval_dataloader,
        is_regression=is_regression_task(args.task),
        num_samples=args.num_samples,
        show_progress_bar=True,
    )

    logger.info(
        "The statistics of first profiled nodes:\n{}".format(
            pformat(results[list(results.keys())[0]])
        )
    )

    if args.save_dir is not None:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "statistic_profile.toml"
        with open(save_path, "w") as f:
            toml.dump(results, f)
        logger.info(f"Profile statistics saved to {save_path}")

    logger.info("==================== Profile Statistics Ends ====================")


def profile_statistics_lm_runner():
    parser = ArgumentParser()
    parser.add_argument("--model_arch", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--quant_config", type=str, required=True)
    parser.add_argument("--task", type=str, choices=["wikitext2"], required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--max_length", type=int, default=2048)

    parser.add_argument("--act_stats", type=str, nargs="+", default=None)
    parser.add_argument("--weight_stats", type=str, nargs="+", default=None)
    parser.add_argument("--profile_config", type=str, default=None)

    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--model_parallelism", action="store_true")

    args = parser.parse_args()

    if (args.act_stats is None and args.weight_stats is None) and (
        args.profile_config is None
    ):
        raise ValueError(
            "Either (act_stats, weight_stats) or profile_config should be provided."
        )
    elif args.profile_config is not None:
        if args.act_stats is not None:
            logger.warning(
                "act_stats is provided, but will be overwritten by profile_config."
            )
        if args.weight_stats is not None:
            logger.warning(
                "weight_stats is provided, but will be overwritten by profile_config."
            )
        profile_config = toml.load(args.profile_config)
        args.act_stats = profile_config["act_stats"]
        args.weight_stats = profile_config["weight_stats"]

    logging.info("==================== Profile Statistics ====================")

    model_cls = get_model_cls(args.model_arch, "lm")
    config_cls = get_config_cls(args.model_arch)

    config = config_cls.from_pretrained(args.model_name, quant_config=args.quant_config)
    tokenizer = get_tokenizer_cls(args.model_arch).from_pretrained(
        args.model_name, legacy=False
    )
    if tokenizer.pad_token in ["<unk>", None]:
        tokenizer.pad_token = tokenizer.eos_token
    if not args.model_parallelism:
        model = model_cls.from_pretrained(args.model_name, config=config).to("cuda")
    else:
        model = model_cls.from_pretrained(args.model_name, config=config)
        device_map = infer_auto_device_map(
            model, no_split_module_classes=model._no_split_modules
        )
        model = dispatch_model(model, device_map=device_map)

    raw_dataset_dict = get_raw_dataset_dict(args.task)
    preprocessed_dataset_dict = preprocess_dataset_dict(
        raw_dataset_dict,
        task=args.task,
        tokenizer=tokenizer,
        padding="max_length",
        max_length=args.max_length,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    eval_dataloader = DataLoader(
        preprocessed_dataset_dict["train"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=os.cpu_count(),
    )

    results = profile_statistics_lm_fn(
        act_stats=args.act_stats,
        weight_stats=args.weight_stats,
        hook_registration_fn=get_stat_profiler_hook(args.model_arch),
        model=model,
        eval_dataloader=eval_dataloader,
        num_samples=args.num_samples,
        input_device="cuda:0",
        root_name="root",
        show_progress_bar=True,
    )

    logger.info(
        "The statistics of first profiled nodes:\n{}".format(
            pformat(results[list(results.keys())[0]])
        )
    )

    if args.save_dir is not None:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "statistic_profile.toml"
        with open(save_path, "w") as f:
            toml.dump(results, f)
        logger.info(f"Profile statistics saved to {save_path}")

    logger.info("==================== Profile Statistics Ends ====================")
