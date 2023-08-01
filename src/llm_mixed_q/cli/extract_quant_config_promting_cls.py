import sys
from pathlib import Path
from argparse import ArgumentParser
from pprint import pformat
import json
import logging

from torch.utils.data import DataLoader
from transformers import default_data_collator


from ..utils import set_logging_verbosity, extract_quant_config_fn
from ..utils.logger import get_logger, set_logging_verbosity
from ..datasets import (
    get_num_labels,
    get_raw_dataset_dict,
    preprocess_dataset_dict,
    is_regression_task,
)
from ..models import get_model_cls, get_config_cls, get_tokenizer_cls
from ..eval import evaluate_cls_glue_fn
from ..eval import evaluate_prompting_fn

logger = logging.getLogger(__name__)


def extract_quant_config_and_eval_prompting_runner():
    parser = ArgumentParser()
    parser.add_argument("--study", type=str, required=True, help="Path to study.pkl")
    parser.add_argument("--trial", type=int, required=True, help="Target trial index")
    parser.add_argument("--model_arch", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--tasks", type=str, nargs="+", required=True)
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=None)
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        help="Maximal batch size to try with --batch_size auto",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit the number of examples per task. "
        "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument("--data_sampling", type=float, default=None)
    # parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)
    args = parser.parse_args()

    quant_config_path = None
    results_path = None
    if args.save_dir is not None:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        quant_config_path = save_dir / "quant_config.toml"
        results_path = save_dir / "results.json"

    logger.info("============== Extracting quant config ==============")
    logger.info(pformat(vars(args)))

    quant_config = extract_quant_config_fn(args.study, args.trial, quant_config_path)
    if quant_config_path is not None:
        logger.info(f"Saved quant config to {quant_config_path}")

    logger.info("============== Evaluating in Prompting style ==============")

    results = evaluate_prompting_fn(
        model_wrapper="llm-mixed-q",
        model_arch=args.model_arch,
        model_name=args.model_name,
        quant_config=quant_config,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        limit=args.limit,
        no_cache=not args.use_cache,
    )

    dumped = json.dumps(results, indent=2)
    logger.info(dumped)

    if args.save_dir:
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

    logger.info("============== Done ==============")
