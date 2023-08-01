import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from llm_mixed_q.utils import set_logging_verbosity, get_logger
from llm_mixed_q.cli import evaluate_prompting_cls_runner

logger = get_logger(__name__)


if __name__ == "__main__":
    set_logging_verbosity("info")
    evaluate_prompting_cls_runner()
