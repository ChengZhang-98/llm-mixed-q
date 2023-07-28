import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from llm_mixed_q.train.train_ddp import trainer_ddp
from llm_mixed_q.utils import set_logging_verbosity

if __name__ == "__main__":
    set_logging_verbosity("info")
    trainer_ddp()
