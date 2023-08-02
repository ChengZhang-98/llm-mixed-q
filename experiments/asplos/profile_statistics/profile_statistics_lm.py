import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from llm_mixed_q.utils import set_logging_verbosity
from llm_mixed_q.cli import profile_statistics_lm_runner
import transformers
import datasets as hf_datasets

if __name__ == "__main__":
    # hf_datasets.logging.set_verbosity_info()
    # transformers.logging.set_verbosity_info()
    # hf_datasets.disable_caching()
    set_logging_verbosity("info")
    profile_statistics_lm_runner()
