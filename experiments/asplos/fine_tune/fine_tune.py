import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from llm_mixed_q.train.train_ddp import main

if __name__ == "__main__":
    main()
