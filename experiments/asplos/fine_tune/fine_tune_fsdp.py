import os
import sys
from pathlib import Path
import transformers
import datasets as hf_datasets

sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from llm_mixed_q.utils import set_logging_verbosity
from llm_mixed_q.cli import fsdp_train_runner

if __name__ == "__main__":
    transformers.utils.logging.set_verbosity_error()
    hf_datasets.utils.logging.set_verbosity_error()
    set_logging_verbosity("info")
    fsdp_train_runner()
