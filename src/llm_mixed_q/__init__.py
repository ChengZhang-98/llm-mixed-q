from pathlib import Path
import sys

# A hack to use lm_eval_harness without installing it
sys.path.append(str(Path(__file__).parent.parent / "lm_eval"))
