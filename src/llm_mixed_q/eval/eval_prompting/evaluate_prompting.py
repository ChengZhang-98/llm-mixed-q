import logging

from lm_eval import tasks as lm_eval_tasks
from lm_eval import utils as lm_eval_utils
from lm_eval.models import MODEL_REGISTRY

from ...utils import load_config
from .evaluator import simple_evaluate_llm_mixed_q
from .model_wrapper import QuantizedCausalLMWrapper

logger = logging.getLogger(__name__)

MODEL_REGISTRY["llm-mixed-q"] = QuantizedCausalLMWrapper


def eval_prompting_tasks(
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

    if isinstance(quant_config, str):
        quant_config = load_config(quant_config)

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
