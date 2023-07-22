import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from llm_mixed_q.search import search_quantisation_for_cls
from llm_mixed_q.tools import get_logger, set_logging_verbosity

if __name__ == "__main__":
    # logger = get_logger("search_cls")
    set_logging_verbosity("debug")
    search_quantisation_for_cls()
