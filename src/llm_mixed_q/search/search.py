import ast
import copy
import os
import sys
from pprint import pprint as pp
from pprint import pformat
from pathlib import Path
import ast
from copy import deepcopy
from functools import partial
from argparse import ArgumentParser

import datasets
import joblib
import optuna
import pandas as pd
import toml
import torch
from tabulate import tabulate
import logging
from transformers import default_data_collator
import transformers
from torch.utils.data import DataLoader

from .estimator.software_metrics import evaluate_cls_task
from ..models import (
    get_model_cls,
    get_config_cls,
    get_tokenizer_cls,
    get_q_profiler,
    get_quant_config_parser,
)
from ..datasets import (
    get_num_labels,
    get_raw_dataset_dict,
    preprocess_dataset_dict,
    is_regression_task,
)
from ..tools import (
    load_config,
    save_config,
    convert_none_to_str_na,
    convert_str_na_to_none,
    set_logging_verbosity,
)

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

transformers.utils.logging.set_verbosity_error()
datasets.utils.logging.set_verbosity_error()

logger = logging.getLogger(__name__)


class SearchBase:
    def __init__(
        self,
        model_arch: str,
        model_name: str,
        task: str,
        search_config: dict | str,
        save_dir: str,
        device: str,
    ) -> None:
        self.model_name = model_name
        self.model_cls = get_model_cls(model_arch, task)
        self.config_cls = get_config_cls(model_arch)
        self.tokenizer = get_tokenizer_cls(model_arch).from_pretrained(model_name)
        self.model_config = self.config_cls.from_pretrained(model_name)
        self.device = device

        self.search_config = (
            search_config
            if isinstance(search_config, dict)
            else load_config(search_config)
        )
        self.save_dir = Path(save_dir)
        self._create_logger()

    def _create_logger(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(type(self).__name__)
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(self.save_dir / "search.log")
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
        self.logger = logger

    def rebuild_model(self, quant_config):
        if quant_config is None:
            config = self.config_cls.from_pretrained(self.model_name)
        else:
            config = self.config_cls.from_pretrained(
                self.model_name, quant_config=quant_config
            )
        model = self.model_cls.from_pretrained(self.model_name, config=config)
        model.to(self.device)
        return model


class SearchQuantisationForClassification(SearchBase):
    def __init__(
        self,
        model_arch: str,
        model_name: str,
        search_config: dict | str,
        save_dir: str,
        num_labels: int,
        device: str,
    ) -> None:
        super().__init__(model_arch, model_name, "cls", search_config, save_dir, device)
        self.q_profiler = get_q_profiler(model_arch)
        self.q_config_parser = get_quant_config_parser(model_arch)
        self.num_labels = num_labels

    def rebuild_model(self, quant_config):
        if quant_config is None:
            config = self.config_cls.from_pretrained(
                self.model_name, num_labels=self.num_labels
            )
        else:
            config = self.config_cls.from_pretrained(
                self.model_name, quant_config=quant_config, num_labels=self.num_labels
            )
        model = self.model_cls.from_pretrained(self.model_name, config=config)
        model.to(self.device)
        return model

    def search(
        self,
        eval_dataloader,
        task: str,
        is_regression: bool,
        seq_len: int,
    ):
        def sample_config(trial: optuna.Trial, config, name: str) -> dict:
            if isinstance(config, dict):
                for k, v in config.items():
                    config[k] = sample_config(trial, v, f"{name}:{k}")
                return config
            elif isinstance(config, list):
                sampled = trial.suggest_categorical(name, config)
                # logger.debug(f"name `{name}` sampled")
                if isinstance(sampled, str) and sampled.startswith("!ast!"):
                    sampled = ast.literal_eval(sampled.removeprefix("!ast!"))
                return sampled
            else:
                logger.error(f"Unknown config: {type(config)} = {config}")
                raise ValueError(f"Unknown config: {type(config)} = {config}")

        def compute_software_metric(
            model, task, eval_dataloader, is_regression, num_samples
        ):
            results = evaluate_cls_task(
                model,
                task,
                eval_dataloader,
                is_regression=is_regression,
                num_samples=num_samples,
            )
            match task:
                case "sst2":
                    metric = results["accuracy"]
                case _:
                    raise NotImplementedError(f"task {task} not implemented")
            return metric

        def compute_hardware_metric(profiler, config, seq_len, compare_to=32):
            results = profiler(config, seq_len)
            num_params = results["num_params"]
            num_acts = results["num_acts"]
            param_bits = results["param_bits"]
            act_bits = results["act_bits"]

            param_bits_fp32 = compare_to * num_params
            act_bits_fp32 = compare_to * num_acts

            mem_density = (param_bits_fp32 + act_bits_fp32) / (param_bits + act_bits)
            logger.debug(f"hardware_metric_results: {results}")
            return mem_density

        def objective(trial: optuna.Trial, quant_config_seed, seq_len: int):
            quant_config_seed = deepcopy(quant_config_seed)
            sampled_config = sample_config(
                trial=trial,
                config=quant_config_seed,
                name="root",
            )
            parsed_quant_config = self.q_config_parser(
                sampled_config, num_hidden_layers=self.model_config.num_hidden_layers
            )

            logger.debug(f"============= Sampled Config =============")
            logger.debug("\n" + pformat(sampled_config))
            model = self.rebuild_model(parsed_quant_config)
            s_metric = compute_software_metric(
                model=model,
                task=task,
                eval_dataloader=eval_dataloader,
                is_regression=is_regression,
                num_samples=self.search_config["search_estimator"]["num_samples"],
            )
            logger.debug(f"model.config.quant_config: {model.config.quant_config}")
            h_metric = compute_hardware_metric(
                self.q_profiler,
                model.config,
                seq_len=seq_len,
                compare_to=self.search_config["search_estimator"]["compare_to"],
            )
            logger.debug("Memory Density: " + str(h_metric))
            logger.debug(
                "Avg bitwidth: "
                + str(self.search_config["search_estimator"]["compare_to"] / h_metric)
            )
            a_s_metric = self.search_config["search_estimator"]["alpha_acc"]
            a_h_metric = self.search_config["search_estimator"]["alpha_mem_density"]

            s_metric = a_s_metric * s_metric
            h_metric = a_h_metric * h_metric
            return s_metric, h_metric

        def logger_callback(
            study: optuna.Study, frozen_trail: optuna.trial.FrozenTrial
        ):
            s_metric, h_metric = frozen_trail.values
            # fmt: off
            ori_s_metric = s_metric / self.search_config["search_estimator"]["alpha_acc"]
            ori_h_metric = h_metric / self.search_config["search_estimator"]["alpha_mem_density"]
            # fmt: on
            self.logger.info(
                f"Trial {frozen_trail.number}: "
                f"(scaled_acc, scaled_mem_density) = "
                f"({s_metric:.4f}, {h_metric:.4f}), "
                f"(unscaled_acc, unscaled_mem_density) = "
                f"({ori_s_metric:.4f}, {ori_h_metric:.4f}),"
            )

        # create sampler and study
        match self.search_config["search_strategy"]["sampler"].lower():
            case "random":
                sampler = optuna.samplers.RandomSampler()
            case "tpe":
                sampler = optuna.samplers.MOTPESampler()
            case "nsgaii":
                sampler = optuna.samplers.NSGAIISampler()
            case _:
                raise ValueError(
                    f"Unknown sampler name: {self.search_config['search_strategy']['sampler']}"
                )
        study = optuna.create_study(
            directions=["maximize", "maximize"],
            sampler=sampler,
        )

        # sample configs
        q_config_seed = self.search_config["search_space"]["quant_config_seed"]

        study.optimize(
            func=partial(
                objective,
                quant_config_seed=q_config_seed,
                seq_len=seq_len,
            ),
            n_trials=self.search_config["search_strategy"]["n_trials"],
            n_jobs=self.search_config["search_strategy"]["n_jobs"],
            timeout=self.search_config["search_strategy"].get("timeout", None),
            show_progress_bar=True,
            callbacks=[logger_callback],
        )

        self.save_study_and_results(study)
        return study

    @staticmethod
    def save_trial_to_quant_config(trial: optuna.trial.FrozenTrial, save_path: str):
        def parse_and_create_item(quant_config: dict, keys: list[str], value):
            for i, key in enumerate(keys):
                if key not in quant_config:
                    quant_config[key] = {}
                if i == len(keys) - 1:
                    quant_config[key] = value
                else:
                    quant_config = quant_config[key]

        params = trial.params

        quant_config = {}
        for name, value in params.items():
            keys = name.removeprefix("root:").split(":")
            if isinstance(value, str) and value.startswith("!ast!"):
                value = ast.literal_eval(value.removeprefix("!ast!"))
            parse_and_create_item(quant_config, keys, value)

        save_config(quant_config, save_path)

    def save_study_and_results(self, study: optuna.Study):
        save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        study_path = save_dir / "study.pkl"
        result_table_path = save_dir / "results.csv"
        result_df = pd.DataFrame(
            columns=[
                "accuracy",
                "memory density",
                "average_bitwidth",
                "quant_config",
                "scaled_acc",
                "scaled_mem_density",
            ]
        )
        for i, trial in enumerate(study.best_trials):
            quant_config_path = save_dir / f"quant_config_{i}.toml"
            SearchQuantisationForClassification.save_trial_to_quant_config(
                trial, quant_config_path
            )
            scaled_s_metric, scaled_h_metric = trial.values[0], trial.values[1]
            s_metric = (
                scaled_s_metric / self.search_config["search_estimator"]["alpha_acc"]
            )
            h_metric = (
                scaled_h_metric
                / self.search_config["search_estimator"]["alpha_mem_density"]
            )
            avg_bitwidth = (
                self.search_config["search_estimator"]["compare_to"] / h_metric
            )
            result_df.loc[i] = [
                s_metric,
                h_metric,
                avg_bitwidth,
                quant_config_path,
                scaled_s_metric,
                scaled_h_metric,
            ]
        result_df.to_csv(result_table_path, index=False)
        joblib.dump(study, study_path)
        logger.info("========== Best Trials ==========")
        logger.info(
            f"(alpha_acc, alpha_mem_density) = {self.search_config['search_estimator']['alpha_acc']}, {self.search_config['search_estimator']['alpha_mem_density']}"
        )
        logger.info(
            "\n"
            + tabulate(result_df, headers="keys", tablefmt="pretty", floatfmt=".4f")
        )
        logger.info(f"Results saved to {save_dir}")
        logger.info(f"Study saved to {study_path}")


def search_quantisation_for_cls():
    parser = ArgumentParser()
    parser.add_argument("--model_arch", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--task", type=str, choices=["sst2"], required=True)
    parser.add_argument("--search_config", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--padding", type=str, default="max_length")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--accelerator", type=str, default="cuda")

    args = parser.parse_args()

    logger.info("========== Search Config ==========")
    logger.info(pformat(vars(args)))
    logger.info("========== Search Starts ==========")

    search_obj = SearchQuantisationForClassification(
        model_arch=args.model_arch,
        model_name=args.model_name,
        search_config=args.search_config,
        save_dir=args.save_dir,
        num_labels=get_num_labels(args.task),
        device=args.accelerator,
    )

    raw_dataset_dict = get_raw_dataset_dict(args.task)
    preprocessed_dataset_dict = preprocess_dataset_dict(
        raw_dataset_dict,
        args.task,
        split="train",
        tokenizer=search_obj.tokenizer,
        padding=args.padding,
        max_length=args.max_length,
    )
    is_regression = is_regression_task(args.task)
    eval_dataloader = DataLoader(
        preprocessed_dataset_dict["train"],
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        num_workers=os.cpu_count(),
    )

    study = search_obj.search(
        eval_dataloader=eval_dataloader,
        task=args.task,
        is_regression=is_regression,
        seq_len=args.max_length,
    )

    logger.info("========== Search Ends ==========")
