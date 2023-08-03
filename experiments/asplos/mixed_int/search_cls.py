import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from llm_mixed_q.cli import search_conditional_quantisation_for_cls_runner
from llm_mixed_q.utils import set_logging_verbosity

if __name__ == "__main__":
    set_logging_verbosity("info")
    search_conditional_quantisation_for_cls_runner()