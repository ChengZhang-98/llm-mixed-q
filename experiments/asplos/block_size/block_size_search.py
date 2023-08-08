import os
from argparse import ArgumentParser
import toml
import json
import sys
from pathlib import Path
from copy import deepcopy
import transformers
import datasets as hf_datasets
from tqdm import tqdm
import pandas as pd
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from llm_mixed_q.utils import set_logging_verbosity, get_logger
from llm_mixed_q.eval import eval_lm_wikitext2
from llm_mixed_q.datasets import preprocess_dataset_dict, get_raw_dataset_dict
from llm_mixed_q.models import get_tokenizer_cls, get_model_cls, get_config_cls


os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = get_logger(__name__)


def create_quant_config(quant_config, block_size: list[int]):
    quant_config = deepcopy(quant_config)
    quant_config["default"]["data_in_block_size"] = block_size
    quant_config["default"]["weight_in_block_size"] = block_size
    quant_config["default"]["bias_block_size"] = block_size[-1:]
    return quant_config


def build_model(model_arch, model_name, quant_config):
    config = get_config_cls(model_arch).from_pretrained(
        model_name, quant_config=quant_config
    )
    model = get_model_cls(model_arch, "lm").from_pretrained(model_name, config=config)
    return model


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_arch", type=str, default="opt")
    parser.add_argument("--model_name", type=str, default="facebook/opt-1.3b")
    parser.add_argument(
        "--quant_config_blueprint",
        "--quant_config",
        dest="quant_config",
        type=str,
        required=True,
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--save_dir", type=str, required=True)

    args = parser.parse_args()

    logger.info(vars(args))

    tokenizer = get_tokenizer_cls(args.model_arch).from_pretrained(args.model_name)

    raw_dataset = get_raw_dataset_dict("wikitext2")
    preprocessed_dataset_dict = preprocess_dataset_dict(
        raw_dataset,
        task="wikitext2",
        tokenizer=tokenizer,
        padding="max_length",
        max_length=args.max_length,
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    eval_dataloader = DataLoader(
        preprocessed_dataset_dict["test"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=os.cpu_count(),
    )

    default_quant_config = toml.load(args.quant_config)

    # fmt: off
    block_sizes = [
        [3, 3], [5,5], [7,7], [9,9],
        [1, 2], [1, 4], [1, 8], [1, 16], [1, 32],
        [2, 1], [2, 2], [2, 4], [2, 8], [2,16], [2,32],
        [4, 1], [4, 2], [4, 4], [4, 8], [4,16], [4,32],
        [8, 1], [8, 2], [8, 4], [8, 8], [8,16], [8,32],
        [16, 1], [16, 2], [16, 4], [16, 8], [16,16], [16,32],
        [32, 1], [32, 2], [32, 4], [32, 8], [32,16], [32,32],
    ]
    # block_sizes = [[1,16], [1,32]] # llama-7b, block_size=[1, 32], perplexity=6.05
    block_sizes = [[1, 16], [1,32]]
    block_sizes = list(reversed(block_sizes))
    # fmt: on

    csv_df = pd.DataFrame(columns=["block_size", "perplexity"])
    progress_bar = tqdm(block_sizes, total=len(block_sizes), desc="Search block size")

    for block_size in progress_bar:
        quant_config_i = create_quant_config(default_quant_config, block_size)
        model = build_model(args.model_arch, args.model_name, quant_config_i).to("cuda")

        results = eval_lm_wikitext2(
            model,
            eval_dataloader,
            progress_bar=False,
            input_device="cuda",
        )
        csv_df.loc[len(csv_df)] = [block_size, results["perplexity"]]
        progress_bar.set_postfix(
            {"block_size": block_size, "perplexity": results["perplexity"]}
        )
        progress_bar.update(1)
        del model

    csv_path = Path(args.save_dir) / "block_size.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_df.to_csv(csv_path, index=False)

    logger.info(f"Saved block size search results to {csv_path}")


if __name__ == "__main__":
    transformers.utils.logging.set_verbosity_error()
    hf_datasets.utils.logging.set_verbosity_error()
    set_logging_verbosity("info")
    main()
