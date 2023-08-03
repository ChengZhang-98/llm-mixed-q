import ast
import logging
from copy import deepcopy

import optuna

logger = logging.getLogger(__name__)


def sample_a_list(trial: optuna.Trial, name: str, choices: list):
    assert isinstance(choices, list), f"choices must be a list, got {choices}"
    sampled = trial.suggest_categorical(name, deepcopy(choices))
    if isinstance(sampled, str) and sampled.startswith("!ast!"):
        sampled = ast.literal_eval(sampled.removeprefix("!ast!"))
    # logger.debug(f"sampled {name} = {sampled}")
    return sampled


def sample_a_dict_of_list(
    trial: optuna.Trial, name: str, config: dict[list[str | int | float]]
):
    assert isinstance(config, dict), f"config must be a dict, got {config}"
    sampled_dict = {}
    for k, v in config.items():
        sampled_dict[k] = sample_a_list(trial, f"{name}:{k}", v)
    return sampled_dict
