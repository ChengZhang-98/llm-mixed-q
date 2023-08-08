import os
import sys
from pathlib import Path

import torch
from accelerate import init_empty_weights
import toml
from pprint import pprint

sys.path.append(str(Path(__file__).parent.parent / "src"))
from llm_mixed_q.models import (
    get_model_cls,
    get_config_cls,
    get_tokenizer_cls,
    get_model_profiler,
)
from llm_mixed_q.models.quantize.stat_profile_to_quant_config import (
    transform_stat_profile_to_int_quant_config,
    find_int_frac_width,
)

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"


def test_find_int_frac_width():
    print(find_int_frac_width(4, max_half_range=7 / 8, frac_choices=None))


def test_transform_stat_profile_to_int_quant_config():
    profile = toml.load("./stat_profile.toml")

    result = transform_stat_profile_to_int_quant_config(
        profile, range_entry="range_min_max", width=8, frac_choices=None
    )
    pprint(result)


if __name__ == "__main__":
    # test_find_int_frac_width()
    test_transform_stat_profile_to_int_quant_config()
