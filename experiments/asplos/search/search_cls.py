import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from llm_mixed_q.search import search_quantisation_for_cls
from llm_mixed_q.utils import get_logger, set_logging_verbosity

if __name__ == "__main__":
    set_logging_verbosity("debug")
    search_quantisation_for_cls()
