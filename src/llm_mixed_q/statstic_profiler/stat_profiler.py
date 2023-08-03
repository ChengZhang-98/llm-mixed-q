import logging
from ..eval import evaluate_cls_glue, eval_lm_wikitext2
from .stat_manager import StatManager

import logging

logger = logging.getLogger(__name__)


def profile_statistics_cls_glue_fn(
    act_stats: tuple[str],
    weight_stats: tuple[str],
    hook_registration_fn: callable,
    model,
    task: str,
    eval_dataloader,
    is_regression: bool,
    num_samples: int,
    root_name: str = "root",
    show_progress_bar: bool = True,
):
    """
    This function is used to profile the statistics of the activations and weights of the model.
    The statistics are collected by the hooks registered by the hook_registration_fn.

    Args:
        act_stats (tuple[str]): A tuple of strings, each of which is the name of an activation statistic.
        weight_stats (tuple[str]): A tuple of strings, each of which is the name of a weight statistic.
        hook_registration_fn (callable): A function that registers hooks to the model.

    ----
    hook_registration_fn should have the following signature:
        def hook_registration_fn(stat_manager: StatManager, model, root_name: str, num_hidden_layers: int)->None:
    """

    stat_manager = StatManager(act_stats, weight_stats)
    hook_registration_fn(
        stat_manager=stat_manager,
        model=model,
        name=root_name,
        num_hidden_layers=model.config.num_hidden_layers,
    )
    evaluate_cls_glue(
        model=model,
        task=task,
        eval_dataloader=eval_dataloader,
        is_regression=is_regression,
        num_samples=num_samples,
        progress_bar=True,
    )
    stat_profile = stat_manager.finalize(show_progress_bar=show_progress_bar)
    return stat_profile


def profile_statistics_lm_fn(
    act_stats: tuple[str],
    weight_stats: tuple[str],
    hook_registration_fn: callable,
    model,
    eval_dataloader,
    num_samples: int,
    input_device: str,
    root_name: str = "root",
    show_progress_bar: bool = True,
):
    stat_manager = StatManager(act_stats, weight_stats)
    hook_registration_fn(
        stat_manager=stat_manager,
        model=model,
        name=root_name,
        num_hidden_layers=model.config.num_hidden_layers,
    )

    eval_lm_wikitext2(
        model=model,
        eval_dataloader=eval_dataloader,
        num_samples=num_samples,
        progress_bar=True,
        input_device=input_device,
    )
    stat_profile = stat_manager.finalize(show_progress_bar=show_progress_bar)
    return stat_profile
