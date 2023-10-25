import os
import sys
from pathlib import Path

sys.path.append((Path(__file__).resolve().parents[4] / "src").as_posix())

import datasets as hf_datasets
import transformers

from llm_mixed_q.cli import cli_eval_prompting_cls
from llm_mixed_q.utils import set_logging_verbosity


def main():
    transformers.logging.set_verbosity_error()
    hf_datasets.logging.set_verbosity_error()
    set_logging_verbosity("info")

    cli_eval_prompting_cls()


if __name__ == "__main__":
    main()
