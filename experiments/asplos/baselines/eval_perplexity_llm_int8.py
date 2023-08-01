import sys
from pathlib import Path
from pathlib import Path
import transformers
import datasets as hf_datasets

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent / "src"))
from llm_mixed_q.cli import eval_perplexity_wikitext_runner
from llm_mixed_q.utils import get_logger, set_logging_verbosity

logger = get_logger(__name__)


if __name__ == "__main__":
    transformers.logging.set_verbosity_error()
    hf_datasets.logging.set_verbosity_error()
    set_logging_verbosity("info")
    eval_perplexity_wikitext_runner()
