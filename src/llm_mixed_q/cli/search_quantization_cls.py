import logging
import os
from argparse import ArgumentParser
from pprint import pformat

from torch.utils.data import DataLoader
from transformers import default_data_collator, set_seed

from ..datasets import (get_num_labels, get_raw_dataset_dict,
                        is_regression_task, preprocess_dataset_dict)
from ..search import (SearchQuantisationForClassification,
                      SearchQuantisationForPromptingCLS)

logger = logging.getLogger(__name__)

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def cli_search_quant_on_cls_glue():
    parser = ArgumentParser()
    parser.add_argument("--model_arch", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--task", type=str, choices=["sst2"], required=True)
    parser.add_argument("--search_config", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--padding", type=str, default="max_length")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument(
        "--search_dataset_split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
    )
    parser.add_argument(
        "--eval_dataset_split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
    )
    parser.add_argument("--accelerator", type=str, default="cuda:0")
    parser.add_argument("--model_parallel", action="store_true")
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    logger.info("==================== Search Config ====================")
    logger.info(pformat(vars(args)))
    logger.info("==================== Search Starts ====================")

    if args.seed is not None:
        set_seed(args.seed)

    search_obj = SearchQuantisationForClassification(
        model_arch=args.model_arch,
        model_name=args.model_name,
        search_config=args.search_config,
        save_dir=args.save_dir,
        num_labels=get_num_labels(args.task),
        device=args.accelerator,
        model_parallel=args.model_parallel,
    )

    raw_dataset_dict = get_raw_dataset_dict(args.task)
    preprocessed_dataset_dict = preprocess_dataset_dict(
        raw_dataset_dict,
        args.task,
        tokenizer=search_obj.tokenizer,
        padding=args.padding,
        max_length=args.max_length,
    )
    is_regression = is_regression_task(args.task)
    search_dataloader = DataLoader(
        preprocessed_dataset_dict[args.search_dataset_split],
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        num_workers=os.cpu_count(),
        shuffle=False,
    )
    eval_dataloader = DataLoader(
        preprocessed_dataset_dict[args.eval_dataset_split],
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        num_workers=os.cpu_count(),
        shuffle=False,
    )

    study = search_obj.search(
        eval_dataloader=search_dataloader,
        task=args.task,
        is_regression=is_regression,
        seq_len=args.max_length,
    )

    search_obj.evaluate_best_trials(
        study,
        eval_dataloader=eval_dataloader,
        task=args.task,
        is_regression=is_regression,
    )

    logger.info("==================== Search Ends ====================")
