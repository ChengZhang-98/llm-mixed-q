import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from llm_mixed_q.utils import set_logging_verbosity
from llm_mixed_q.cli import ddp_train_runner

if __name__ == "__main__":
    set_logging_verbosity("info")
    ddp_train_runner()
