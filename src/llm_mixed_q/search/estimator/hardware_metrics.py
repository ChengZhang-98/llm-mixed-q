import re
import optuna


def get_matched_keys(pattern: str, keys: list[str]) -> list[str]:
    """Get matched keys from a list of keys."""
    matched_keys = []
    for key in keys:
        match = re.fullmatch(pattern, key)
        if match:
            matched_keys.append(key)
    return matched_keys


def quantisation_sampler_llama(
    trial: optuna.Trial, quant_config: dict, root_name: str = "root"
):
    # model layer config
    model_layer_keys = None