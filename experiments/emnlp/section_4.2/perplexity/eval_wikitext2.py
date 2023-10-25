from pathlib import Path
import os
import sys

sys.path.append((Path(__file__).resolve().parents[4] / "src").as_posix())

from llm_mixed_q.cli import cli_eval_lm_wikitext2
from llm_mixed_q.utils import set_logging_verbosity
import transformers
import datasets as hf_datasets


def main():
    transformers.logging.set_verbosity_error()
    hf_datasets.logging.set_verbosity_error()
    set_logging_verbosity("info")

    cli_eval_lm_wikitext2()


if __name__ == "__main__":
    main()
