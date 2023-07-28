import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from llm_mixed_q.utils import set_logging_verbosity
from llm_mixed_q.eval import eval_lm_runner

if __name__ == "__main__":
    set_logging_verbosity("info")
    eval_lm_runner()
