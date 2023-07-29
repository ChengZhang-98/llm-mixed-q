from argparse import ArgumentParser
import json
import logging
import os
from pprint import pformat
from lm_eval.models import MODEL_REGISTRY
from lm_eval import utils as lm_eval_utils
from lm_eval import tasks as lm_eval_tasks
from lm_eval import evaluator as lm_eval_evaluator
from .evaluator import simple_evaluate_llm_mixed_q
from .model_wrapper import QuantizedCausalLM

logger = logging.getLogger(__name__)

MODEL_REGISTRY["llm-mixed-q"] = QuantizedCausalLM


def evaluate_prompting_fn(
    model_wrapper: str,
    model_arch: str,
    model_name: str,
    quant_config: str | dict,
    tasks: list[str],
    num_fewshot: int = 0,
    batch_size: str = None,
    max_batch_size: int = None,
    device: str = None,
    limit: float = None,
    no_cache: bool = False,
):
    task_names = lm_eval_utils.pattern_match(tasks, lm_eval_tasks.ALL_TASKS)
    description_dict = {}

    model_args = f"model_arch={model_arch},model_name={model_name}"
    # results = lm_eval_evaluator.simple_evaluate(
    #     model=model_wrapper,
    #     model_args=model_args,
    #     tasks=task_names,
    #     num_fewshot=num_fewshot,
    #     batch_size=batch_size,
    #     max_batch_size=max_batch_size,
    #     device=device,
    #     no_cache=no_cache,
    #     limit=limit,
    #     description_dict=description_dict,
    #     decontamination_ngrams_path=None,
    #     check_integrity=False,
    #     write_out=False,
    #     output_base_path=None,
    # )
    results = simple_evaluate_llm_mixed_q(
        model=model_wrapper,
        model_args=model_args,
        quant_config=quant_config,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        device=device,
        no_cache=no_cache,
        limit=limit,
        description_dict=description_dict,
        decontamination_ngrams_path=None,
        check_integrity=False,
        write_out=False,
        output_base_path=None,
    )
    return results


def evaluate_prompting_runner():
    parser = ArgumentParser()

    parser.add_argument(
        "--model",
        help="lm-eval model wrapper name",
        choices=list(MODEL_REGISTRY.keys()),
        default="llm-mixed-q",
    )
    parser.add_argument(
        "--model_args", required=True, help="model arguments separated by commas"
    )
    parser.add_argument(
        "--tasks", required=True, help="comma-separated list of tasks to evaluate on"
    )
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
    parser.add_argument("--output_path", default=None)
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit the number of examples per task. "
        "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)

    args = parser.parse_args()

    logger.info(f"========== Running eval_prompting ==========")
    logger.info(pformat(vars(args)))

    task_names = lm_eval_utils.pattern_match(
        args.tasks.split(","), lm_eval_tasks.ALL_TASKS
    )

    logger.info(f"Selected Tasks: {task_names}")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    results = lm_eval_evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        no_cache=args.no_cache,
        limit=args.limit,
        description_dict=description_dict,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        output_base_path=args.output_base_path,
    )

    dumped = json.dumps(results, indent=2)
    print(dumped)

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w") as f:
            f.write(dumped)

    batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))
    print(
        f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
        f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
    )
    print(lm_eval_evaluator.make_table(results))
