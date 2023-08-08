import ast
import json
import logging
from functools import partial
from pathlib import Path
from pprint import pformat

import datasets
import joblib
import optuna
import pandas as pd
import transformers
from accelerate import (
    infer_auto_device_map,
    init_empty_weights,
    load_checkpoint_and_dispatch,
)
from tabulate import tabulate

from ..eval import eval_prompting_tasks
from ..eval import eval_cls_glue as evaluate_cls_task
from ..models import (
    get_model_profiler,
    get_config_cls,
    get_model_cls,
    get_quant_config_parser,
    get_quant_config_sampler,
    get_stat_config_formatter,
    get_tokenizer_cls,
)
from ..models.quantize import transform_stat_profile_to_int_quant_config
from ..utils import flatten_dict, load_config, save_config

optuna.logging.set_verbosity(optuna.logging.ERROR)

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
        model_parallel: bool = False,
    ) -> None:
        self.model_arch = model_arch
        self.model_name = model_name
        self.model_cls = get_model_cls(model_arch, task)
        self.config_cls = get_config_cls(model_arch)
        self.tokenizer = get_tokenizer_cls(model_arch).from_pretrained(model_name)
        self.model_config = self.config_cls.from_pretrained(model_name)
        self.device = device
        self.model_parallel = model_parallel

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
        fh = logging.FileHandler(self.save_dir / "search_log.csv")
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
        if "cuda" in self.device:
            if self.model_parallel:
                with init_empty_weights():
                    model = self.model_cls(config)
                device_map = infer_auto_device_map(
                    model,
                    no_split_module_classes=model._no_split_modules,
                )
                model = load_checkpoint_and_dispatch(
                    model, checkpoint=self.model_name, device_map=device_map
                )
            else:
                model = self.model_cls.from_pretrained(
                    self.model_name, config=config
                ).to(self.device)
        elif self.device == "cpu":
            model = self.model_cls.from_pretrained(self.model_name, config=config)
        else:
            raise ValueError(f"Unknown device: {self.device}")
        return model


class SearchIntQuantisationForClassification(SearchBase):
    """
    Perform quantisation search for bert-like models on classification tasks
    - Bert-like model refers to a network consisting of the base model and a classifier head, already fine-tuned on downstream tasked.
    - This class calls a evaluation function to get the results on glue tasks
    """

    def __init__(
        self,
        model_arch: str,
        model_name: str,
        search_config: dict | str,
        save_dir: str,
        num_labels: int,
        device: str,
        model_parallel: bool = False,
    ) -> None:
        super().__init__(
            model_arch,
            model_name,
            "cls",
            search_config,
            save_dir,
            device,
            model_parallel,
        )
        self.q_bitwidth_profiler = get_model_profiler(model_arch)
        self.q_config_parser = get_quant_config_parser(model_arch)
        # TODO: use a general recursive quant config parser, which traverses dict to sample leaf values (each leaf value is a list of choices)
        self.q_config_sampler = get_quant_config_sampler(model_arch)
        self.q_config_formatter = get_stat_config_formatter(model_arch)
        self.num_labels = num_labels

        self._pre_search_check()

    def _pre_search_check(self):
        if self.search_config["search_estimator"]["alpha_accuracy"] == 0:
            assert (
                self.search_config["search_strategy"]["accuracy_threshold"] == 0
            ), "alpha_accuracy is 0, please set accuracy_threshold to 0 as well"
        if self.search_config["search_estimator"]["alpha_memory_density"] == 0:
            assert (
                self.search_config["search_strategy"]["avg_bitwidth_threshold"] == 0
            ), "alpha_memory_density is 0, please set avg_bitwidth_threshold to 0 as well"
        if self.search_config["search_estimator"]["alpha_ops_per_bit"] == 0:
            assert (
                self.search_config["search_strategy"]["ops_per_bit_threshold"] == 0
            ), "alpha_ops_per_bit is 0, please set ops_per_bit_threshold to 0 as well"

    def rebuild_model(self, quant_config):
        if quant_config is None:
            config = self.config_cls.from_pretrained(
                self.model_name, num_labels=self.num_labels
            )
        else:
            config = self.config_cls.from_pretrained(
                self.model_name, quant_config=quant_config, num_labels=self.num_labels
            )
        if "cuda" in self.device:
            if self.model_parallel:
                with init_empty_weights():
                    model = self.model_cls(config)
                device_map = infer_auto_device_map(
                    model,
                    no_split_module_classes=model._no_split_modules,
                )
                model = load_checkpoint_and_dispatch(
                    model, checkpoint=self.model_name, device_map=device_map
                )
                logger.debug("Model parallelism enabled")
            else:
                model = self.model_cls.from_pretrained(
                    self.model_name, config=config
                ).to(self.device)
                logger.debug(f"Running on single device: {self.device}")
        elif self.device == "cpu":
            model = self.model_cls.from_pretrained(self.model_name, config=config)
            logger.debug("Running on CPU")
        else:
            raise ValueError(f"Unknown device: {self.device}")
        return model

    def search(
        self,
        eval_dataloader,
        task: str,
        is_regression: bool,
        seq_len: int,
        stat_profiler: dict,
        range_entry: str = "range_min_max",
        num_samples_per_trial: int = 512,
    ):
        def compute_software_metric(
            model, task, eval_dataloader, is_regression, num_samples
        ) -> dict:
            results = evaluate_cls_task(
                model,
                task,
                eval_dataloader,
                is_regression=is_regression,
                num_samples=num_samples,
            )
            match task:
                case "sst2":
                    accuracy = results["accuracy"]
                case _:
                    raise NotImplementedError(f"task {task} not implemented")
            s_metric = {
                "accuracy": accuracy,
            }
            return s_metric

        def compute_hardware_metric(profiler, config, seq_len, compare_to=32) -> dict:
            results = profiler(config, seq_len)
            num_params = results["num_params"]
            num_acts = results["num_acts"]
            param_bits = results["param_bits"]
            act_bits = results["act_bits"]
            flops = results["flops"]

            param_bits_fp32 = compare_to * num_params
            act_bits_fp32 = compare_to * num_acts

            mem_density = (param_bits_fp32 + act_bits_fp32) / (param_bits + act_bits)
            ops_per_bit = flops / (param_bits + act_bits)
            h_metric = {
                "memory_density": mem_density,
                "ops_per_bit": ops_per_bit,
            }
            return h_metric

        def objective(
            trial: optuna.Trial,
            # quant_config_sampler,
            quant_config_seed,
            seq_len: int,
            stat_profile: dict,
            range_entry: str,
            file_logger: logging.Logger,
            num_samples: int,
        ):
            """
            seed -> sampled_config -> extend sampled to full config -> determine frac_width -> evaluate
            """
            if self.search_config["search_space"]["extend_quant_config_seed_first"]:
                quant_config_seed = self.q_config_parser(
                    quant_config_seed, self.model_config.num_hidden_layers, strict=False
                )

            # TODO: create a general recursive quant config parser
            sampled_config = self.q_config_sampler(
                trial=trial,
                name="root",
                config_seed=quant_config_seed,
            )

            sampled_config = self.q_config_parser(
                sampled_config, self.model_config.num_hidden_layers, strict=False
            )
            flattened_sampled_config = {}
            flatten_dict(sampled_config, new_d=flattened_sampled_config)

            sampled_config_complete = sampled_config
            sampled_config = transform_stat_profile_to_int_quant_config(
                stat_profile,
                range_entry=range_entry,
                width=flattened_sampled_config,
                frac_choices=None,
                root_name="root",
                is_ptq=True,
                bypass=False,
            )

            self.q_config_formatter(
                sampled_config,
                self.model_config.num_hidden_layers,
                default_config=sampled_config_complete,
                is_ptq=True,
                bypass=False,
            )

            model = self.rebuild_model(sampled_config)

            s_metric = compute_software_metric(
                model=model,
                task=task,
                eval_dataloader=eval_dataloader,
                is_regression=is_regression,
                num_samples=num_samples,
            )
            h_metric = compute_hardware_metric(
                self.q_bitwidth_profiler,
                model.config,
                seq_len=seq_len,
                compare_to=self.search_config["search_estimator"]["compare_to"],
            )
            metric_name_list = list(s_metric.keys()) + list(h_metric.keys())
            scaled_metric_list = []
            metric_list = list(s_metric.values()) + list(h_metric.values())

            # accuracy
            for metric_name, metric in s_metric.items():
                scaled_metric_list.append(
                    metric
                    * self.search_config["search_estimator"][f"alpha_{metric_name}"]
                )
            # memory density, ops_per_bit
            for metric_name, metric in h_metric.items():
                scaled_metric_list.append(
                    metric
                    * self.search_config["search_estimator"][f"alpha_{metric_name}"]
                )

            if trial.number == 0:
                file_logger.info(
                    f"trial_id,"
                    + ",".join(metric_name_list)
                    + ","
                    + ",".join(map(lambda x: f"scaled_{x}", metric_name_list))
                )

            file_logger.info(
                f"{trial.number},"
                + ",".join(map(str, metric_list))
                + ","
                + ",".join(map(str, scaled_metric_list))
            )

            return (*scaled_metric_list,)

        def logger_callback(
            study: optuna.Study, frozen_trail: optuna.trial.FrozenTrial
        ):
            acc, mem_density, ops_per_bits = frozen_trail.values
            # fmt: off
            ori_acc = acc / (self.search_config["search_estimator"]["alpha_accuracy"] + 1e-8)
            ori_mem_density = mem_density / (self.search_config["search_estimator"]["alpha_memory_density"] + 1e-8)
            ori_ops_per_bits = ops_per_bits / (self.search_config["search_estimator"]["alpha_ops_per_bit"] + 1e-8)

            avg_bitwidth = self.search_config["search_estimator"]["compare_to"] / ori_mem_density
            # fmt: on
            logger.info(
                f"Trial {frozen_trail.number} is done: "
                f"unscaled (accuracy, mem_density, ops_per_bit) = "
                f"({ori_acc:.4f}, {ori_mem_density:.2f}, {ori_ops_per_bits:.2f}), "
                f"scaled (...) = "
                f"({acc:.4f}, {mem_density:.2f}, {ops_per_bits:.2f}), "
                f"avg_bitwidth = {avg_bitwidth:.1f}"
            )

        # create sampler and study
        match self.search_config["search_strategy"]["sampler"].lower():
            case "random":
                sampler = optuna.samplers.RandomSampler()
            case "tpe":
                sampler = optuna.samplers.TPESampler()
            case "nsgaii":
                sampler = optuna.samplers.NSGAIISampler()
            case "nsgaiii":
                sampler = optuna.samplers.NSGAIIISampler()
            case "qmc":
                sampler = optuna.samplers.QMCSampler()
            case _:
                raise ValueError(
                    f"Unknown sampler name: {self.search_config['search_strategy']['sampler']}"
                )
        logger.info(f"Using sampler: {sampler.__class__.__name__}")
        study = optuna.create_study(
            directions=["maximize", "maximize", "maximize"],
            sampler=sampler,
        )

        # sample configs
        q_config_seed = self.search_config["search_space"]["quant_config_seed"]

        study.optimize(
            func=partial(
                objective,
                quant_config_seed=q_config_seed,
                seq_len=seq_len,
                stat_profile=stat_profiler,
                range_entry=range_entry,
                file_logger=self.logger,
                num_samples=num_samples_per_trial,
            ),
            n_trials=self.search_config["search_strategy"]["n_trials"],
            n_jobs=self.search_config["search_strategy"]["n_jobs"],
            timeout=self.search_config["search_strategy"].get("timeout", None),
            show_progress_bar=True,
            callbacks=[logger_callback],
        )

        self.save_study_and_results(study, stat_profiler, range_entry)
        return study

    @staticmethod
    def save_trial_to_quant_config(
        trial: optuna.trial.FrozenTrial,
        q_config_parser: callable,
        q_config_formatter: callable,
        num_hidden_layers: int,
        stat_profile: dict,
        range_entry: str,
        save_path: str = None,
    ):
        def parse_and_create_item(quant_config: dict, keys: list[str], value):
            for i, key in enumerate(keys):
                if key not in quant_config:
                    quant_config[key] = {}
                if i == len(keys) - 1:
                    quant_config[key] = value
                else:
                    quant_config = quant_config[key]

        params = trial.params

        sampled_config = {}
        for name, value in params.items():
            keys = name.removeprefix("root:").split(":")
            if isinstance(value, str) and value.startswith("!ast!"):
                value = ast.literal_eval(value.removeprefix("!ast!"))
            parse_and_create_item(sampled_config, keys, value)
        # here we got sampled_config = self.q_config_sampler in self.search

        sampled_config = q_config_parser(
            sampled_config, num_hidden_layers, strict=False
        )
        flattened_sampled_config = {}
        flatten_dict(sampled_config, new_d=flattened_sampled_config)

        sampled_config_complete = sampled_config
        sampled_config = transform_stat_profile_to_int_quant_config(
            stat_profile,
            range_entry=range_entry,
            width=flattened_sampled_config,
            frac_choices=None,
            root_name="root",
            is_ptq=True,
            bypass=False,
        )
        q_config_formatter(
            sampled_config,
            num_hidden_layers,
            default_config=sampled_config_complete,
            is_ptq=True,
            bypass=False,
        )

        if save_path is not None:
            save_config(sampled_config, save_path)
        return sampled_config

    @staticmethod
    def get_result_df(
        study: optuna.Study,
        q_config_parser: callable,
        q_config_formatter: callable,
        num_hidden_layers: int,
        stat_profile: dict,
        range_entry: str,
        save_dir,
        alpha_acc: float,
        alpha_mem_density: float,
        alpha_ops_per_bit: float,
        compare_to: int,
    ) -> pd.DataFrame:
        result_df = pd.DataFrame(
            columns=[
                "trial_id",
                "accuracy",
                "memory_density",
                "ops_per_bit",
                "scaled_accuracy",
                "scaled_memory_density",
                "scaled_ops_per_bit",
                "quant_config_path",
                "avg_bitwidth",
                "quant_config",
            ]
        )
        quant_config_dir = save_dir / "quant_configs"
        quant_config_dir.mkdir(parents=True, exist_ok=True)
        for i, trial in enumerate(study.best_trials):
            trial_id = trial.number
            quant_config_path = quant_config_dir / f"quant_config_{i}.toml"
            quant_config = (
                SearchIntQuantisationForClassification.save_trial_to_quant_config(
                    trial,
                    q_config_parser=q_config_parser,
                    q_config_formatter=q_config_formatter,
                    num_hidden_layers=num_hidden_layers,
                    stat_profile=stat_profile,
                    range_entry=range_entry,
                    save_path=quant_config_path,
                )
            )
            scaled_acc, scaled_mem_density, scaled_ops_per_bit = trial.values
            acc = scaled_acc / (alpha_acc + 1e-8)
            mem_density = scaled_mem_density / (alpha_mem_density + 1e-8)
            ops_per_bit = scaled_ops_per_bit / (alpha_ops_per_bit + 1e-8)
            avg_bitwidth = compare_to / mem_density
            result_df.loc[i] = [
                trial_id,
                acc,
                mem_density,
                ops_per_bit,
                scaled_acc,
                scaled_mem_density,
                scaled_ops_per_bit,
                quant_config_path,
                avg_bitwidth,
                quant_config,
            ]
        result_df = result_df.sort_values(by="accuracy", ascending=False)
        return result_df

    def save_study_and_results(self, study: optuna.Study, stat_profile, range_entry):
        save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        study_path = save_dir / "study.pkl"
        result_table_path = save_dir / "results.csv"
        search_config_path = save_dir / "search_config.toml"
        save_config(self.search_config, search_config_path)

        # fmt: off
        result_df = SearchIntQuantisationForClassification.get_result_df(
            study,
            q_config_parser=self.q_config_parser,
            q_config_formatter=self.q_config_formatter,
            num_hidden_layers=self.model_config.num_hidden_layers,
            stat_profile=stat_profile,
            range_entry=range_entry,
            save_dir=save_dir,
            alpha_acc=self.search_config["search_estimator"]["alpha_accuracy"],
            alpha_mem_density=self.search_config["search_estimator"]["alpha_memory_density"],
            alpha_ops_per_bit=self.search_config["search_estimator"]["alpha_ops_per_bit"],
            compare_to=self.search_config["search_estimator"]["compare_to"],
        )
        # fmt: on

        result_df.drop("quant_config", axis=1).to_csv(result_table_path, index=False)
        joblib.dump(study, study_path)
        logger.info("========== Best Trials ==========")
        logger.info(
            f"(alpha_accuracy, alpha_memory_density, alpha_ops_per_bit) = "
            f"{self.search_config['search_estimator']['alpha_accuracy']}, "
            f"{self.search_config['search_estimator']['alpha_memory_density']}, "
            f"{self.search_config['search_estimator']['alpha_ops_per_bit']}"
        )

        result_df = result_df.drop("quant_config", axis=1)
        result_df["quant_config_name"] = result_df["quant_config_path"].apply(
            lambda x: "$save_dir/quant_configs/" + str(Path(x).name)
        )
        result_df = result_df.applymap(
            lambda x: f"{x:.4f}" if isinstance(x, float) else x
        )
        result_df = result_df.drop("quant_config_path", axis=1)
        logger.info(
            "\n"
            + tabulate(
                result_df,
                headers="keys",
                tablefmt="pretty",
            )
        )
        logger.info(f"Results saved to {save_dir}")
        logger.info(f"Study saved to {study_path}")

    def evaluate_best_trials(
        self,
        study: optuna.Study,
        stat_profile: dict,
        range_entry: str,
        eval_dataloader,
        task,
        is_regression,
    ):
        # fmt: off
        acc_threshold = self.search_config["search_strategy"]["accuracy_threshold"]
        avg_bitwidth_threshold = self.search_config["search_strategy"]["avg_bitwidth_threshold"]
        ops_per_bit_threshold = self.search_config["search_strategy"]["ops_per_bit_threshold"]
        # fmt: on
        sort_by = self.search_config["search_strategy"]["sort_by"]

        for i, s in enumerate(sort_by):
            assert s in [
                "accuracy",
                "avg_bitwidth",
                "ops_per_bit",
            ], f"Unknown sort_by: {s}, must be one of ['accuracy', 'avg_bitwidth', 'ops_per_bit']"

        # fmt: off
        result_df = SearchIntQuantisationForClassification.get_result_df(
            study,
            q_config_parser=self.q_config_parser,
            q_config_formatter=self.q_config_formatter,
            num_hidden_layers=self.model_config.num_hidden_layers,
            stat_profile=stat_profile,
            range_entry=range_entry,
            save_dir=self.save_dir,
            alpha_acc=self.search_config["search_estimator"]["alpha_accuracy"],
            alpha_mem_density=self.search_config["search_estimator"]["alpha_memory_density"],
            alpha_ops_per_bit=self.search_config["search_estimator"]["alpha_ops_per_bit"],
            compare_to=self.search_config["search_estimator"]["compare_to"],
        )
        # fmt: on

        filtered_df = result_df.loc[result_df["accuracy"] >= acc_threshold]
        filtered_df = filtered_df.loc[
            filtered_df["avg_bitwidth"] <= avg_bitwidth_threshold
        ]
        filtered_df = filtered_df.loc[
            filtered_df["ops_per_bit"] >= ops_per_bit_threshold
        ]
        if len(filtered_df) == 0:
            logger.warning(
                f"No trials found with acc >= {acc_threshold}, avg_bitwidth <= {avg_bitwidth_threshold}, ops_per_bit >= {ops_per_bit_threshold}"
            )
            return

        ascending_mapping = {
            "accuracy": False,
            "avg_bitwidth": True,
            "ops_per_bit": False,
        }

        filtered_df = filtered_df.sort_values(
            sort_by, ascending=[ascending_mapping[s] for s in sort_by]
        )

        best_quant_config = filtered_df.iloc[0]["quant_config"]
        save_config(best_quant_config, self.save_dir / "best_quant_config.toml")

        logger.info("========== Evaluating the Best ==========")
        model = self.rebuild_model(best_quant_config)
        results = evaluate_cls_task(
            model,
            task,
            eval_dataloader,
            is_regression=is_regression,
            num_samples=None,
            progress_bar=True,
        )
        with open(self.save_dir / "best_eval.json", "w") as f:
            json.dump(results, f, indent=4)

        logger.info(
            f"Best quant config avg bitwidth: {filtered_df.iloc[0]['avg_bitwidth']: .2f}"
        )
        logger.info(
            f"Best quant config software metric: {pformat(results)}, saved to {self.save_dir / 'best_eval.json'})"
        )


class SearchIntQuantisationForPromptingCLS(SearchBase):
    """
    Perform quantisation search for GPT-like models on downstream tasks
    - GPT-like model refers to a network performing language modeling only (no classifier head).
    - This class calls lm-eval-harness to get the results on downstream tasks.
    """

    def __init__(
        self,
        model_arch: str,
        model_name: str,
        search_config: dict | str,
        save_dir: str,
    ) -> None:
        super().__init__(
            model_arch,
            model_name,
            "lm",
            search_config,
            save_dir,
            device=None,
            model_parallel=False,
        )
        self.q_bitwidth_profiler = get_model_profiler(model_arch)
        self.q_config_parser = get_quant_config_parser(model_arch)
        self.q_config_sampler = get_quant_config_sampler(model_arch)
        self.q_config_formatter = get_stat_config_formatter(model_arch)

        self._pre_search_check()

    def _pre_search_check(self):
        if self.search_config["search_estimator"]["alpha_accuracy"] == 0:
            assert (
                self.search_config["search_strategy"]["accuracy_threshold"] == 0
            ), "alpha_accuracy is 0, please set accuracy_threshold to 0 as well"
        if self.search_config["search_estimator"]["alpha_memory_density"] == 0:
            assert (
                self.search_config["search_strategy"]["avg_bitwidth_threshold"] == 0
            ), "alpha_memory_density is 0, please set avg_bitwidth_threshold to 0 as well"
        if self.search_config["search_estimator"]["alpha_ops_per_bit"] == 0:
            assert (
                self.search_config["search_strategy"]["ops_per_bit_threshold"] == 0
            ), "alpha_ops_per_bit is 0, please set ops_per_bit_threshold to 0 as well"

    def rebuild_model(self, quant_config):
        raise NotImplementedError

    def rebuild_model_config(self, quant_config):
        self.model_config = self.config_cls.from_pretrained(
            self.model_name, quant_config=quant_config
        )
        return self.model_config

    def search(
        self,
        tasks: list[str],
        num_fewshot: int,
        batch_size: int,
        max_batch_size: int,
        device: str,
        num_samples_per_trial: int,
        stat_profile: dict,
        range_entry: str = "range_min_max",
        profiler_seq_len: int = 256,
    ):
        def compute_software_metric(
            model_arch,
            model_name,
            quant_config,
            tasks,
            num_fewshot,
            batch_size,
            max_batch_size,
            device,
            limit,
        ) -> dict:
            results = eval_prompting_tasks(
                model_wrapper="llm-mixed-q",
                model_arch=model_arch,
                model_name=model_name,
                quant_config=quant_config,
                tasks=tasks,
                num_fewshot=num_fewshot,
                batch_size=batch_size,
                max_batch_size=max_batch_size,
                device=device,
                limit=limit,
                no_cache=True,
            )
            results = results[
                "results"
            ]  # {"results": ..., "config": ..., version: ...}
            if len(results) > 1:
                logger.debug("software_metric_results: " + str(results))
                logger.warning(
                    f"More than one task results returned, simply averaging the accuracy if available"
                )
            logger.debug("software_metric_results: " + str(results))
            acc_list = []
            for task_name, task_metric in results.items():
                if "acc" in task_metric:
                    acc_list.append(task_metric["acc"])
                else:
                    logger.warning(f"Task {task_name} does not have accuracy, skipping")
            avg_acc = sum(acc_list) / len(acc_list)
            s_metric = {
                "accuracy": avg_acc,
            }

            return s_metric

        def compute_hardware_metric(profiler, config, seq_len, compare_to=32):
            results = profiler(config, seq_len)
            num_params = results["num_params"]
            num_acts = results["num_acts"]
            param_bits = results["param_bits"]
            act_bits = results["act_bits"]
            flops = results["flops"]

            param_bits_fp32 = compare_to * num_params
            act_bits_fp32 = compare_to * num_acts

            mem_density = (param_bits_fp32 + act_bits_fp32) / (param_bits + act_bits)
            ops_per_bit = flops / (param_bits + act_bits)
            h_metric = {
                "memory_density": mem_density,
                "ops_per_bit": ops_per_bit,
            }
            return h_metric

        def objective(
            trial: optuna.Trial,
            quant_config_seed,
            seq_len: int,
            tasks: list[str],
            num_fewshot: int,
            batch_size: int,
            max_batch_size: int,
            device: str,
            limit: int,
            stat_profile: dict,
            range_entry: str,
            file_logger: logging.Logger,
        ):
            if self.search_config["search_space"]["extend_quant_config_seed_first"]:
                quant_config_seed = self.q_config_parser(
                    quant_config_seed, self.model_config.num_hidden_layers, strict=False
                )

            sampled_config = self.q_config_sampler(
                trial=trial,
                name="root",
                config_seed=quant_config_seed,
            )

            sampled_config = self.q_config_parser(
                sampled_config, self.model_config.num_hidden_layers, strict=False
            )

            flattened_sampled_config = {}
            flatten_dict(sampled_config, new_d=flattened_sampled_config)

            sampled_config_complete = sampled_config
            sampled_config = transform_stat_profile_to_int_quant_config(
                stat_profile,
                range_entry=range_entry,
                width=flattened_sampled_config,
                frac_choices=None,
                root_name="root",
                is_ptq=True,
                bypass=False,
            )

            self.q_config_formatter(
                sampled_config,
                self.model_config.num_hidden_layers,
                default_config=sampled_config_complete,
                is_ptq=True,
                bypass=False,
            )
            s_metric = compute_software_metric(
                model_arch=self.model_arch,
                model_name=self.model_name,
                quant_config=sampled_config,
                tasks=tasks,
                num_fewshot=num_fewshot,
                batch_size=batch_size,
                max_batch_size=max_batch_size,
                device=device,
                limit=limit,
            )
            config = self.rebuild_model_config(sampled_config)
            h_metric = compute_hardware_metric(
                self.q_bitwidth_profiler,
                config,
                seq_len=seq_len,
                compare_to=self.search_config["search_estimator"]["compare_to"],
            )

            metric_name_list = list(s_metric.keys()) + list(h_metric.keys())
            scaled_metric_list = []
            metric_list = list(s_metric.values()) + list(h_metric.values())

            # accuracy
            for metric_name, metric in s_metric.items():
                scaled_metric_list.append(
                    metric
                    * self.search_config["search_estimator"][f"alpha_{metric_name}"]
                )
            # memory density, ops_per_bit
            for metric_name, metric in h_metric.items():
                scaled_metric_list.append(
                    metric
                    * self.search_config["search_estimator"][f"alpha_{metric_name}"]
                )

            if trial.number == 0:
                file_logger.info(
                    f"trial_id,"
                    + ",".join(metric_name_list)
                    + ","
                    + ",".join(map(lambda x: f"scaled_{x}", metric_name_list))
                )

            file_logger.info(
                f"{trial.number},"
                + ",".join(map(str, metric_list))
                + ","
                + ",".join(map(str, scaled_metric_list))
            )
            return (*scaled_metric_list,)

        def logger_callback(
            study: optuna.Study, frozen_trail: optuna.trial.FrozenTrial
        ):
            acc, mem_density, ops_per_bits = frozen_trail.values
            # fmt: off
            ori_acc = acc / (self.search_config["search_estimator"]["alpha_accuracy"] + 1e-8)
            ori_mem_density = mem_density / (self.search_config["search_estimator"]["alpha_memory_density"] + 1e-8)
            ori_ops_per_bits = ops_per_bits / (self.search_config["search_estimator"]["alpha_ops_per_bit"] + 1e-8)

            avg_bitwidth = self.search_config["search_estimator"]["compare_to"] / ori_mem_density
            # fmt: on
            logger.info(
                f"Trial {frozen_trail.number} is done: "
                f"unscaled (accuracy, mem_density, ops_per_bit) = "
                f"({ori_acc:.4f}, {ori_mem_density:.2f}, {ori_ops_per_bits:.2f}), "
                f"scaled (...) = "
                f"({acc:.4f}, {mem_density:.2f}, {ops_per_bits:.2f}), "
                f"avg_bitwidth = {avg_bitwidth:.1f}"
            )

        # create sampler and study
        match self.search_config["search_strategy"]["sampler"].lower():
            case "random":
                sampler = optuna.samplers.RandomSampler()
            case "tpe":
                sampler = optuna.samplers.TPESampler()
            case "nsgaii":
                sampler = optuna.samplers.NSGAIISampler()
            case "nsgaiii":
                sampler = optuna.samplers.NSGAIIISampler()
            case "qmc":
                sampler = optuna.samplers.QMCSampler()
            case _:
                raise ValueError(
                    f"Unknown sampler name: {self.search_config['search_strategy']['sampler']}"
                )
        logger.info(f"Using sampler: {sampler.__class__.__name__}")
        study = optuna.create_study(
            directions=["maximize", "maximize", "maximize"],
            sampler=sampler,
        )

        # sample configs
        q_config_seed = self.search_config["search_space"]["quant_config_seed"]

        study.optimize(
            func=partial(
                objective,
                quant_config_seed=q_config_seed,
                seq_len=profiler_seq_len,
                tasks=tasks,
                num_fewshot=num_fewshot,
                batch_size=batch_size,
                max_batch_size=max_batch_size,
                device=device,
                limit=num_samples_per_trial,
                stat_profile=stat_profile,
                range_entry=range_entry,
                file_logger=self.logger,
            ),
            n_trials=self.search_config["search_strategy"]["n_trials"],
            n_jobs=self.search_config["search_strategy"]["n_jobs"],
            timeout=self.search_config["search_strategy"].get("timeout", None),
            show_progress_bar=True,
            callbacks=[logger_callback],
        )

        self.save_study_and_results(study, stat_profile, range_entry)
        return study

    @staticmethod
    def save_trial_to_quant_config(
        trial: optuna.trial.FrozenTrial,
        q_config_parser: callable,
        q_config_formatter: callable,
        num_hidden_layers: int,
        stat_profile: dict,
        range_entry: str,
        save_path: str = None,
    ):
        def parse_and_create_item(quant_config: dict, keys: list[str], value):
            for i, key in enumerate(keys):
                if key not in quant_config:
                    quant_config[key] = {}
                if i == len(keys) - 1:
                    quant_config[key] = value
                else:
                    quant_config = quant_config[key]

        params = trial.params

        sampled_config = {}
        for name, value in params.items():
            keys = name.removeprefix("root:").split(":")
            if isinstance(value, str) and value.startswith("!ast!"):
                value = ast.literal_eval(value.removeprefix("!ast!"))
            parse_and_create_item(sampled_config, keys, value)
        # here we got sampled_config = self.q_config_sampler in self.search

        sampled_config = q_config_parser(
            sampled_config, num_hidden_layers, strict=False
        )
        flattened_sampled_config = {}
        flatten_dict(sampled_config, new_d=flattened_sampled_config)

        sampled_config_complete = sampled_config
        sampled_config = transform_stat_profile_to_int_quant_config(
            stat_profile,
            range_entry=range_entry,
            width=flattened_sampled_config,
            frac_choices=None,
            root_name="root",
            is_ptq=True,
            bypass=False,
        )
        q_config_formatter(
            sampled_config,
            num_hidden_layers,
            default_config=sampled_config_complete,
            is_ptq=True,
            bypass=False,
        )

        if save_path is not None:
            save_config(sampled_config, save_path)
        return sampled_config

    @staticmethod
    def get_result_df(
        study: optuna.Study,
        q_config_parser: callable,
        q_config_formatter: callable,
        num_hidden_layers: int,
        stat_profile: dict,
        range_entry: str,
        save_dir,
        alpha_acc: float,
        alpha_mem_density: float,
        alpha_ops_per_bit: float,
        compare_to: int,
    ) -> pd.DataFrame:
        result_df = pd.DataFrame(
            columns=[
                "trial_id",
                "accuracy",
                "memory_density",
                "ops_per_bit",
                "scaled_accuracy",
                "scaled_memory_density",
                "scaled_ops_per_bit",
                "quant_config_path",
                "avg_bitwidth",
                "quant_config",
            ]
        )
        quant_config_dir = save_dir / "quant_configs"
        quant_config_dir.mkdir(parents=True, exist_ok=True)
        for i, trial in enumerate(study.best_trials):
            trial_id = trial.number
            quant_config_path = quant_config_dir / f"quant_config_{i}.toml"
            quant_config = (
                SearchIntQuantisationForPromptingCLS.save_trial_to_quant_config(
                    trial,
                    q_config_parser=q_config_parser,
                    q_config_formatter=q_config_formatter,
                    num_hidden_layers=num_hidden_layers,
                    stat_profile=stat_profile,
                    range_entry=range_entry,
                    save_path=quant_config_path,
                )
            )
            scaled_acc, scaled_mem_density, scaled_ops_per_bit = trial.values
            acc = scaled_acc / (alpha_acc + 1e-8)
            mem_density = scaled_mem_density / (alpha_mem_density + 1e-8)
            ops_per_bit = scaled_ops_per_bit / (alpha_ops_per_bit + 1e-8)
            avg_bitwidth = compare_to / mem_density
            result_df.loc[i] = [
                trial_id,
                acc,
                mem_density,
                ops_per_bit,
                scaled_acc,
                scaled_mem_density,
                scaled_ops_per_bit,
                quant_config_path,
                avg_bitwidth,
                quant_config,
            ]
        result_df = result_df.sort_values(by="accuracy", ascending=False)
        return result_df

    def save_study_and_results(self, study: optuna.Study, stat_profile, range_entry):
        save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        study_path = save_dir / "study.pkl"
        result_table_path = save_dir / "results.csv"
        search_config_path = save_dir / "search_config.toml"
        save_config(self.search_config, search_config_path)

        # fmt: off
        result_df = SearchIntQuantisationForPromptingCLS.get_result_df(
            study,
            q_config_parser=self.q_config_parser,
            q_config_formatter=self.q_config_formatter,
            num_hidden_layers=self.model_config.num_hidden_layers,
            stat_profile=stat_profile,
            range_entry=range_entry,
            save_dir=save_dir,
            alpha_acc=self.search_config["search_estimator"]["alpha_accuracy"],
            alpha_mem_density=self.search_config["search_estimator"]["alpha_memory_density"],
            alpha_ops_per_bit=self.search_config["search_estimator"]["alpha_ops_per_bit"],
            compare_to=self.search_config["search_estimator"]["compare_to"],
        )
        # fmt: on
        result_df.drop("quant_config", axis=1).to_csv(result_table_path, index=False)
        joblib.dump(study, study_path)
        logger.info("========== Best Trials ==========")
        logger.info(
            f"(alpha_accuracy, alpha_memory_density, alpha_ops_per_bit) = "
            f"{self.search_config['search_estimator']['alpha_accuracy']}, "
            f"{self.search_config['search_estimator']['alpha_memory_density']}, "
            f"{self.search_config['search_estimator']['alpha_ops_per_bit']}"
        )

        result_df = result_df.drop("quant_config", axis=1)
        result_df["quant_config_name"] = result_df["quant_config_path"].apply(
            lambda x: "$save_dir/quant_configs/" + str(Path(x).name)
        )
        result_df = result_df.applymap(
            lambda x: f"{x:.4f}" if isinstance(x, float) else x
        )
        result_df = result_df.drop("quant_config_path", axis=1)
        logger.info(
            "\n"
            + tabulate(
                result_df,
                headers="keys",
                tablefmt="pretty",
            )
        )
        logger.info(f"Results saved to {save_dir}")
        logger.info(f"Study saved to {study_path}")

    def evaluate_best_trials(
        self,
        study: optuna.Study,
        tasks: list[str],
        num_fewshot: int,
        batch_size: int,
        max_batch_size: int,
        device: str,
        stat_profile: dict,
        range_entry: str,
    ):
        # fmt: off
        acc_threshold = self.search_config["search_strategy"]["accuracy_threshold"]
        avg_bitwidth_threshold = self.search_config["search_strategy"]["avg_bitwidth_threshold"]
        ops_per_bit_threshold = self.search_config["search_strategy"]["ops_per_bit_threshold"]
        # fmt: on
        sort_by = self.search_config["search_strategy"]["sort_by"]

        for i, s in enumerate(sort_by):
            assert s in [
                "accuracy",
                "avg_bitwidth",
                "ops_per_bit",
            ], f"Unknown sort_by: {s}, must be one of ['accuracy', 'avg_bitwidth', 'ops_per_bit']"
        # fmt: off
        result_df = SearchIntQuantisationForPromptingCLS.get_result_df(
            study,
            q_config_parser=self.q_config_parser,
            q_config_formatter=self.q_config_formatter,
            num_hidden_layers=self.model_config.num_hidden_layers,
            stat_profile=stat_profile,
            range_entry=range_entry,
            save_dir=self.save_dir,
            alpha_acc=self.search_config["search_estimator"]["alpha_accuracy"],
            alpha_mem_density=self.search_config["search_estimator"]["alpha_memory_density"],
            alpha_ops_per_bit=self.search_config["search_estimator"]["alpha_ops_per_bit"],
            compare_to=self.search_config["search_estimator"]["compare_to"],
        )
        # fmt: on

        filtered_df = result_df.loc[result_df["accuracy"] >= acc_threshold]
        filtered_df = filtered_df.loc[
            filtered_df["avg_bitwidth"] <= avg_bitwidth_threshold
        ]
        filtered_df = filtered_df.loc[
            filtered_df["ops_per_bit"] >= ops_per_bit_threshold
        ]
        if len(filtered_df) == 0:
            logger.warning(
                f"No trials found with acc >= {acc_threshold}, avg_bitwidth <= {avg_bitwidth_threshold}, ops_per_bit >= {ops_per_bit_threshold}"
            )
            return

        ascending_mapping = {
            "accuracy": False,
            "avg_bitwidth": True,
            "ops_per_bit": False,
        }

        filtered_df = filtered_df.sort_values(
            sort_by, ascending=[ascending_mapping[s] for s in sort_by]
        )

        best_quant_config = filtered_df.iloc[0]["quant_config"]
        save_config(best_quant_config, self.save_dir / "best_quant_config.toml")

        logger.info("========== Evaluating the Best ==========")
        results = eval_prompting_tasks(
            model_wrapper="llm-mixed-q",
            model_arch=self.model_arch,
            model_name=self.model_name,
            quant_config=best_quant_config,
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            device=device,
            no_cache=True,
        )
        results["trial_id"] = str(filtered_df.iloc[0]["trial_id"])
        with open(self.save_dir / "best_eval.json", "w") as f:
            json.dump(results, f, indent=4)

        logger.info(
            f"Best quant config avg bitwidth: {filtered_df.iloc[0]['avg_bitwidth']: .2f}"
        )
        logger.info(
            f"Best quant config software metric: {pformat(results)}, saved to {self.save_dir / 'best_eval.json'})"
        )
