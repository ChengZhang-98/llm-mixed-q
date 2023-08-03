import os
import sys
from pathlib import Path
import transformers
import datasets as hf_datasets

sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from llm_mixed_q.utils import set_logging_verbosity
from llm_mixed_q.cli import cli_profile_statistics_cls_glue

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    transformers.utils.logging.set_verbosity_error()
    hf_datasets.utils.logging.set_verbosity_error()
    set_logging_verbosity("info")
    cli_profile_statistics_cls_glue()
