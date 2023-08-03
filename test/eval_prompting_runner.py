import os
import sys
from pathlib import Path

import torch
from accelerate import init_empty_weights

sys.path.append(str(Path(__file__).parent.parent / "src"))
from llm_mixed_q.models import (
    get_model_cls,
    get_config_cls,
    get_tokenizer_cls,
    get_bitwidth_profiler,
)
from llm_mixed_q.eval import evaluate_prompting_runner
from llm_mixed_q.eval import eval_prompting_tasks

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"


def test_eval_prompting_runner():
    evaluate_prompting_runner()


def test_eval_prompting_fn():
    model_arch = "opt"
    model_name = "facebook/opt-350m"
    quant_config = "/home/zz7522/Projects/llm-mixed-q/experiments/asplos/configs/quantize/bypass.toml"
    eval_prompting_tasks(
        model_wrapper="llm-mixed-q",
        model_arch=model_arch,
        model_name=model_name,
        quant_config=quant_config,
        tasks=["sst"],
        num_fewshot=0,
        no_cache=True,
    )


if __name__ == "__main__":
    test_eval_prompting_runner()
