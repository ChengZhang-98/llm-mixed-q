import os
import sys
from pathlib import Path
import transformers
import datasets as hf_datasets
from sklearn.exceptions import InconsistentVersionWarning
import warnings

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from llm_mixed_q.cli import cli_eval_cls_glue
from llm_mixed_q.utils import set_logging_verbosity

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    transformers.utils.logging.set_verbosity_error()
    hf_datasets.utils.logging.set_verbosity_error()
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    set_logging_verbosity("info")
    cli_eval_cls_glue()
