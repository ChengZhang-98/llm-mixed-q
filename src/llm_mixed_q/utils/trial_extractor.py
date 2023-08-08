import ast
import logging
from pprint import pformat

import joblib
import optuna

from .config_load import save_config

logger = logging.getLogger(__name__)


def save_trial_to_quant_config(trial: optuna.trial.FrozenTrial, save_path: str = None):
    def parse_and_create_item(quant_config: dict, keys: list[str], value):
        for i, key in enumerate(keys):
            if key not in quant_config:
                quant_config[key] = {}
            if i == len(keys) - 1:
                quant_config[key] = value
            else:
                quant_config = quant_config[key]

    params = trial.params

    quant_config = {}
    for name, value in params.items():
        keys = name.removeprefix("root:").split(":")
        if isinstance(value, str) and value.startswith("!ast!"):
            value = ast.literal_eval(value.removeprefix("!ast!"))
        parse_and_create_item(quant_config, keys, value)
    if save_path is not None:
        save_config(quant_config, save_path)
    return quant_config


def extract_quant_config(
    study: optuna.Study | str,
    target_idx,
    save_path: str = None,
):
    if not isinstance(study, optuna.Study):
        with open(study, "rb") as f:
            study: optuna.Study = joblib.load(f)

    target_trial = study.trials[target_idx]
    quant_config = save_trial_to_quant_config(target_trial, save_path)
    return quant_config
