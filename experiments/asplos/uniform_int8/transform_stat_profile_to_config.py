import os
import sys
from pathlib import Path

import transformers
import datasets as hf_datasets

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from llm_mixed_q.utils import set_logging_verbosity
from llm_mixed_q.cli import cli_transform_stat_profile_to_int_quant_config

if __name__ == "__main__":
    hf_datasets.logging.set_verbosity_error()
    transformers.logging.set_verbosity_error()
    set_logging_verbosity("info")
    cli_transform_stat_profile_to_int_quant_config()
