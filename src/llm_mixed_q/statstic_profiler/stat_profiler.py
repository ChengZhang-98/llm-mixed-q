import torch
import os
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from accelerate import infer_auto_device_map, dispatch_model
from transformers import default_data_collator, DataCollatorForLanguageModeling
from ..eval import evaluate_cls_glue_fn, evaluate_lm_wikitext2_fn
from .stat_manager import StatManager
from ..models import get_config_cls, get_model_cls, get_tokenizer_cls
from ..datasets import (
    get_num_labels,
    get_raw_dataset_dict,
    get_raw_dataset_dict_wikitext2,
    preprocess_dataset_dict,
    is_regression_task,
)


def profile_statistics_cls_glue_fn(
    act_stats: tuple[str],
    weight_stats: tuple[str],
    hook_registration_fn: callable,
    model,
    task: str,
    eval_dataloader,
    is_regression: bool,
    num_samples: int,
):
    """
    This function is used to profile the statistics of the activations and weights of the model.
    The statistics are collected by the hooks registered by the hook_registration_fn.

    Args:
        act_stats (tuple[str]): A tuple of strings, each of which is the name of an activation statistic.
        weight_stats (tuple[str]): A tuple of strings, each of which is the name of a weight statistic.
        hook_registration_fn (callable): A function that registers hooks to the model.

    ----
    """

    stat_manager = StatManager(act_stats, weight_stats)
    hook_registration_fn(model, stat_manager)
    evaluate_cls_glue_fn(
        model=model,
        task=task,
        eval_dataloader=eval_dataloader,
        is_regression=is_regression,
        num_samples=num_samples,
        progress_bar=True,
    )
    stat_profile = stat_manager.finalize()
    return stat_profile


def profile_statistics_lm_wikitext2_fn(
    act_stats: tuple[str],
    weight_stats: tuple[str],
    hook_registration_fn: callable,
    model,
    eval_dataloader,
    num_samples: int,
    input_device: str,
):
    stat_manager = StatManager(act_stats, weight_stats)
    hook_registration_fn(model, stat_manager)

    evaluate_lm_wikitext2_fn(
        model=model,
        eval_dataloader=eval_dataloader,
        num_samples=num_samples,
        progress_bar=True,
        input_device=input_device,
    )
    stat_profile = stat_manager.finalize()
    return stat_profile


def profile_statistics_cls_glue_runner():
    parser = ArgumentParser()
    parser.add_argument("--model_arch", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--max_length", type=int, required=True)

    parser.add_argument("--act_stats", type=str, nargs="+", required=True)
    parser.add_argument("--weight_stats", type=str, nargs="+", required=True)

    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--model_parallelism", action="store_true")

    args = parser.parse_args()

    model_cls = get_model_cls(args.model_arch, "cls")
    config_cls = get_config_cls(args.model_arch)

    config = config_cls.from_pretrained(args.model_name)
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

    results = profile_statistics_cls_glue_fn(
        act_stats=args.act_stats,
        weight_stats=args.weight_stats,
        hook_registration_fn="",
        model=model,
        task=args.task,
        eval_dataloader=eval_dataloader,
        is_regression=is_regression_task(args.task),
        num_samples=args.num_samples,
    )
