import os
from pathlib import Path
from argparse import ArgumentParser
import sys
from pathlib import Path
import transformers
from transformers import default_data_collator
from torch.utils.data import DataLoader
import datasets as hf_datasets
import logging
import joblib
import optuna
import pandas as pd
from tqdm import tqdm
from sklearn.exceptions import InconsistentVersionWarning
import warnings

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from llm_mixed_q.cli import cli_conditional_search_quantization_on_cls_glue
from llm_mixed_q.utils import set_logging_verbosity
from llm_mixed_q.datasets import (
    get_raw_dataset_dict,
    is_regression_task,
    preprocess_dataset_dict,
)
from llm_mixed_q.eval import eval_cls_glue, eval_dse_results
from llm_mixed_q.models import (
    get_tokenizer_cls,
    get_model_cls,
    get_config_cls,
    get_model_profiler,
)
from llm_mixed_q.utils import extract_quant_config


os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


def build_config(model_arch, model_name, quant_config):
    config = get_config_cls(model_arch).from_pretrained(
        model_name, quant_config=quant_config
    )
    return config


def build_model(model_arch, model_name, quant_config):
    config = build_config(model_arch, model_name, quant_config)
    model = (
        get_model_cls(model_arch, "cls")
        .from_pretrained(model_name, config=config)
        .to("cuda")
    )
    return model


def get_avg_bitwidth(model_config, profiler, seq_len):
    profile = profiler(model_config, seq_len)
    total_bits = profile["act_bits"] + profile["param_bits"]
    num_values = profile["num_acts"] + profile["num_params"]

    avg_bitwidth = total_bits / num_values
    return avg_bitwidth


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_arch", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--study", type=str, required=True)
    parser.add_argument("--task", type=str, default="sst2")
    parser.add_argument("--max_length", type=int, default=196)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--save_dir", type=str, required=True)

    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("============== Evaluate all trials ==============")

    with open(args.study, "rb") as f:
        study: optuna.Study = joblib.load(f)
    logger.info(f"Study loaded from {args.study}")

    tokenizer = get_tokenizer_cls(args.model_arch).from_pretrained(args.model_name)
    model_profiler = get_model_profiler(args.model_arch)
    raw_dataset_dict = get_raw_dataset_dict(args.task)
    preprocessed_dataset_dict = preprocess_dataset_dict(
        raw_dataset_dict,
        task=args.task,
        tokenizer=tokenizer,
        padding="max_length",
        max_length=args.max_length,
    )
    eval_dataloader = DataLoader(
        preprocessed_dataset_dict["validation"],
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        shuffle=False,
    )

    accuracy_df = pd.DataFrame(
        columns=[
            "trial_id",
            "accuracy",
            "avg_bitwidth",
            "fps",
            "fps_per_lut",
            "quant_config",
            "datetime_start",
            "datetime_end",
        ]
    )
    qc_dir = save_dir / "quant_configs"
    qc_dir.mkdir(parents=True, exist_ok=True)
    progress_bar = tqdm(range(len(study.trials)))
    for trial_id in progress_bar:
        qc_path_i = qc_dir / f"quant_config_trial_{trial_id}.toml"
        quant_config_i = extract_quant_config(study, trial_id, qc_path_i)
        model_i = build_model(args.model_arch, args.model_name, quant_config_i)
        config_i = model_i.config
        sw_results_i = eval_cls_glue(
            model=model_i,
            task=args.task,
            eval_dataloader=eval_dataloader,
            is_regression=is_regression_task(args.task),
        )
        dse_results = eval_dse_results(
            config_i,
            is_mixed=True,
        )
        accuracy_df.loc[len(accuracy_df)] = [
            trial_id,
            sw_results_i["accuracy"],
            get_avg_bitwidth(config_i, model_profiler, args.max_length),
            dse_results["best_fps"],
            dse_results["best_fps"] / dse_results["resource"],
            qc_path_i.name,
            study.trials[trial_id].datetime_start.strftime("%Y-%m-%d %H:%M:%S"),
            study.trials[trial_id].datetime_complete.strftime("%Y-%m-%d %H:%M:%S"),
        ]
        progress_bar.set_postfix(
            {"trial_id": trial_id, "accuracy": sw_results_i["accuracy"]}
        )
        progress_bar.update(1)

    accuracy_df.to_csv(save_dir / "all_trials.csv", index=False)


if __name__ == "__main__":
    hf_datasets.logging.set_verbosity_error()
    transformers.logging.set_verbosity_error()
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    set_logging_verbosity("info")
    main()
