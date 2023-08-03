import os
import sys
from pathlib import Path
import transformers
import datasets as hf_datasets

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from llm_mixed_q.cli import cli_conditional_search_quant_on_cls_glue
from llm_mixed_q.utils import set_logging_verbosity

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    hf_datasets.logging.set_verbosity_error()
    transformers.logging.set_verbosity_error()
    set_logging_verbosity("info")
    cli_conditional_search_quant_on_cls_glue()
