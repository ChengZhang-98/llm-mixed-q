import logging
import os
from argparse import ArgumentParser
from pprint import pformat

import toml

from ..search import SearchIntQuantisationForPromptingCLS

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


def cli_conditional_search_quantization_on_prompting_cls_tasks():
    parser = ArgumentParser()
    parser.add_argument("--model_arch", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--search_config", type=str, required=True)
    parser.add_argument("--stat_profile", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--tasks", type=str, nargs="+", required=True)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--limit",
        "--num_samples_per_trial",
        dest="num_samples_per_trial",
        type=float,
        default=None,
    )
    parser.add_argument("--profiler_seq_len", type=int, default=256)
    parser.add_argument("--range_entry", type=str, default="range_min_max")

    args = parser.parse_args()

    logger.info("==================== Search Config ====================")
    logger.info(pformat(vars(args)))
    logger.info("==================== Search Starts ====================")

    stat_profile = toml.load(args.stat_profile)

    search_obj = SearchIntQuantisationForPromptingCLS(
        model_arch=args.model_arch,
        model_name=args.model_name,
        search_config=args.search_config,
        save_dir=args.save_dir,
    )

    study = search_obj.search(
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        num_samples_per_trial=args.num_samples_per_trial,
        stat_profile=stat_profile,
        range_entry=args.range_entry,
        profiler_seq_len=args.profiler_seq_len,
    )

    search_obj.evaluate_best_trials(
        study,
        tasks=args.tasks,
        stat_profile=stat_profile,
        range_entry=args.range_entry,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
    )

    logger.info("==================== Search Ends ====================")
