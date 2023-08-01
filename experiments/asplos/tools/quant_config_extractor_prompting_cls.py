import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from llm_mixed_q.utils import set_logging_verbosity
from llm_mixed_q.cli import extract_quant_config_and_eval_prompting_runner

if __name__ == "__main__":
    set_logging_verbosity("info")
    extract_quant_config_and_eval_prompting_runner()
