import logging
import os
from argparse import ArgumentParser
from pprint import pformat

from ..search import SearchQuantisationForPromptingCLS

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


def cli_search_quantisation_on_prompting_cls_tasks():
    parser = ArgumentParser()
    parser.add_argument("--model_arch", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--search_config", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--tasks", type=str, nargs="+", required=True)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--profiler_seq_len", type=int, default=256)

    args = parser.parse_args()

    logger.info("==================== Search Config ====================")
    logger.info(pformat(vars(args)))
    logger.info("==================== Search Starts ====================")

    search_obj = SearchQuantisationForPromptingCLS(
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
        limit=args.limit,
        profiler_seq_len=args.profiler_seq_len,
    )

    search_obj.evaluate_best_trials(
        study,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        limit=args.limit,
    )

    logger.info("==================== Search Ends ====================")
