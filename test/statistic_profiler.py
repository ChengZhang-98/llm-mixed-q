import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model
from transformers import default_data_collator, DataCollatorForLanguageModeling

sys.path.append(str(Path(__file__).parent.parent / "src"))
from llm_mixed_q.models import (
    get_model_cls,
    get_config_cls,
    get_tokenizer_cls,
    get_model_profiler,
    get_stat_profiler_hook,
)
from llm_mixed_q.statstic_profiler.stat_manager import StatManager
from llm_mixed_q.datasets import (
    get_raw_dataset_dict,
    preprocess_dataset_dict,
    is_regression_task,
)
from llm_mixed_q.statstic_profiler.stat_profiler import profile_statistics_cls_glue

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"


def test_stat_profiler_cls():
    class Args:
        model_arch = "opt"
        model_name = "facebook/opt-125m"
        task = "sst2"
        num_samples = 4
        batch_size = 1
        max_length = 128
        act_stats = ("range_min_max",)
        weight_stats = ("range_min_max",)
        model_parallelism = False

    args = Args()

    model_cls = get_model_cls(args.model_arch, "cls")
    config_cls = get_config_cls(args.model_arch)

    config = config_cls.from_pretrained(args.model_name, quant_config="./bypass.toml")
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
    )

    print(results)


if __name__ == "__main__":
    test_stat_profiler_cls()
