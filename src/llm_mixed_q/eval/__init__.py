from .eval_cls import eval_cls_glue
from .eval_lm import eval_lm_wikitext2
from .eval_prompting import eval_prompting_tasks


# from mase_dse import eval_dse_results
def eval_dse_results(*args, **kwargs):
    """
    This is a dummy function for DSE evaluation.

    DSE stands for Design Space Exploration,
    which is not a part of EMNLP 2023 paper: Revisiting Block-based Quantisation: What is Important for Sub-8-bit LLM Inference?

    The DSE repo is not open-sourced yet.
    """
    dummy_dse_results = {
        "best_fps": 0.0,
        "resource": 1.0,
    }
    return dummy_dse_results
