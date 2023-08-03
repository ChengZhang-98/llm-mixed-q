import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from llm_mixed_q.cli import cli_search_quantisation_on_prompting_cls_tasks
from llm_mixed_q.utils import set_logging_verbosity

if __name__ == "__main__":
    set_logging_verbosity("info")
    cli_search_quantisation_on_prompting_cls_tasks()
