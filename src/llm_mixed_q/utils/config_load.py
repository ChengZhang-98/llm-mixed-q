from pathlib import Path

import toml


def convert_str_na_to_none(d):
    """
    Since toml does not support None, we use "NA" to represent None.
    """
    if isinstance(d, dict):
        for k, v in d.items():
            d[k] = convert_str_na_to_none(v)
    elif isinstance(d, list):
        d = [convert_str_na_to_none(v) for v in d]
    elif isinstance(d, tuple):
        d = tuple(convert_str_na_to_none(v) for v in d)
    else:
        if d == "NA":
            return None
        else:
            return d
    return d


def convert_none_to_str_na(d):
    """
    Since toml does not support None, we use "NA" to represent None.
    Otherwise the none-value key will be missing in the toml file.
    """
    if isinstance(d, dict):
        for k, v in d.items():
            d[k] = convert_none_to_str_na(v)
    elif isinstance(d, list):
        d = [convert_none_to_str_na(v) for v in d]
    elif isinstance(d, tuple):
        d = tuple(convert_none_to_str_na(v) for v in d)
    else:
        if d is None:
            return "NA"
        else:
            return d
    return d


def load_config(config_path):
    """Load from a toml config file and convert "NA" to None."""
    with open(config_path, "r") as f:
        config = toml.load(f)
    config = convert_str_na_to_none(config)
    return config


def save_config(config, config_path):
    """Convert None to "NA" and save to a toml config file."""
    config = convert_none_to_str_na(config)
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        toml.dump(config, f)
