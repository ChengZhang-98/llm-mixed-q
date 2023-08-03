import ast
import os
from pprint import pformat
from pathlib import Path
import ast
from functools import partial
import json

from accelerate import (
    load_checkpoint_and_dispatch,
    infer_auto_device_map,
    init_empty_weights,
)
import datasets
import joblib
import optuna
import pandas as pd
from tabulate import tabulate
import logging
import transformers

from ..eval import evaluate_cls_glue as evaluate_cls_task
from ..eval import eval_prompting_tasks
from ..models import (
    get_model_cls,
    get_config_cls,
    get_tokenizer_cls,
    get_bitwidth_profiler,
    get_quant_config_parser,
    get_quant_config_sampler,
    get_stat_config_formatter,
)
from ..models.quantize import transform_stat_profile_to_int_quant_config

from ..utils import (
    load_config,
    save_config,
    flatten_dict,
)

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

transformers.utils.logging.set_verbosity_error()
datasets.utils.logging.set_verbosity_error()
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
        self.q_bitwidth_profiler = get_bitwidth_profiler(model_arch)
        self.q_config_parser = get_quant_config_parser(model_arch)
        # TODO: use a general recursive quant config parser, which traverses dict to sample leaf values (each leaf value is a list of choices)
        self.q_config_sampler = get_quant_config_sampler(model_arch)
        self.q_config_formatter = get_stat_config_formatter(model_arch)
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
    ):
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

        def objective(
            trial: optuna.Trial,
            # quant_config_sampler,
            quant_config_seed,
            seq_len: int,
            stat_profile: dict,
            range_entry: str = "range_min_max",
        ):
            """
            seed -> sampled_config -> extend sampled to full config -> determine frac_width -> evaluate
            """
            if self.search_config["search_space"]["extend_quant_config_seed_first"]:
                quant_config_seed = self.q_config_parser(
                    quant_config_seed, self.model_config.num_hidden_layers, strict=False
                )
            # logger.debug(f"0️⃣ Quant Config Seed:")
            # logger.debug("\n" + pformat(quant_config_seed))
            # TODO: create a general recursive quant config parser
            sampled_config = self.q_config_sampler(
                trial=trial,
                name="root",
                config_seed=quant_config_seed,
            )
            # logger.debug(f"1️⃣ Sampled Config (before parsing):")
            # logger.debug("\n" + pformat(sampled_config))
            sampled_config = self.q_config_parser(
                sampled_config, self.model_config.num_hidden_layers, strict=False
            )
            flattened_sampled_config = {}
            flatten_dict(sampled_config, new_d=flattened_sampled_config)
            logger.debug(f"2️⃣ Flattened Sampled Config:")
            logger.debug("\n" + pformat(flattened_sampled_config))
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
            # logger.debug(f"3️⃣ Config from stat_profile:")
            # logger.debug("\n" + pformat(sampled_config))
            self.q_config_formatter(
                sampled_config,
                self.model_config.num_hidden_layers,
                default_config=sampled_config_complete,
                is_ptq=True,
                bypass=False,
            )
            # logger.debug("4️⃣ Config after formatting:")
            # logger.debug("\n" + pformat(sampled_config))
            model = self.rebuild_model(sampled_config)
            # logger.debug(f"============== model layer 0 =============")
            # logger.debug(f"Model layer 0:{model.bert.encoder.layer[0]}")
            # logger.debug(
            #     f"Model layer 0:{model.bert.encoder.layer[0].attention.self.query.w_quantizer}"
            # )
            s_metric = compute_software_metric(
                model=model,
                task=task,
                eval_dataloader=eval_dataloader,
                is_regression=is_regression,
                num_samples=self.search_config["search_estimator"]["num_samples"],
            )
            h_metric = compute_hardware_metric(
                self.q_bitwidth_profiler,
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

            avg_bitwidth = self.search_config["search_estimator"]["compare_to"] / ori_h_metric
            # fmt: on
            self.logger.info(
                f"{frozen_trail.number},"
                f"{ori_s_metric:.4f},{ori_h_metric:.4f},{avg_bitwidth:.2f},"
                f"{s_metric:.4f}, {h_metric:.4f}"
            )
            logger.info(
                f"Trial {frozen_trail.number} is done: "
                f"unscaled (acc, mem_density, avg_bitwidth) = "
                f"({ori_s_metric:.4f}, {ori_h_metric:.4f}, {avg_bitwidth:.2f}),"
                f"scaled (acc, mem_density) = "
                f"({s_metric:.4f}, {h_metric:.4f}), "
            )
            # logger.debug(f"============= Trial {frozen_trail.number} =============")
            # logger.debug(f"Trial {frozen_trail.number} config: {frozen_trail.params}")

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
            directions=["maximize", "maximize"],
            sampler=sampler,
        )

        # sample configs
        q_config_seed = self.search_config["search_space"]["quant_config_seed"]

        self.logger.info(
            f"trial,unscaled_acc,unscaled_mem_density,avg_bitwidth,scaled_acc,scaled_mem_density"
        )
        study.optimize(
            func=partial(
                objective,
                # quant_config_sampler=self.q_config_sampler,
                quant_config_seed=q_config_seed,
                seq_len=seq_len,
                stat_profile=stat_profiler,
                range_entry=range_entry,
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
        alpha_s: float,
        alpha_h: float,
        compare_to: int,
        sort_by=["accuracy", "memory_density"],
    ) -> pd.DataFrame:
        result_df = pd.DataFrame(
            columns=[
                "trial_id",
                "accuracy",
                "memory_density",
                "avg_bitwidth",
                "quant_config_path",
                "scaled_acc",
                "scaled_mem_density",
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
            alpha_acc = alpha_s
            alpha_mem_density = alpha_h
            scaled_s_metric, scaled_h_metric = trial.values[0], trial.values[1]
            s_metric = scaled_s_metric / alpha_acc
            h_metric = scaled_h_metric / alpha_mem_density
            avg_bitwidth = compare_to / h_metric
            result_df.loc[i] = [
                trial_id,
                s_metric,
                h_metric,
                avg_bitwidth,
                quant_config_path,
                scaled_s_metric,
                scaled_h_metric,
                quant_config,
            ]
        result_df = result_df.sort_values(by=sort_by, ascending=False)
        return result_df

    def save_study_and_results(self, study: optuna.Study, stat_profile, range_entry):
        save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        study_path = save_dir / "study.pkl"
        result_table_path = save_dir / "results.csv"
        search_config_path = save_dir / "search_config.toml"
        save_config(self.search_config, search_config_path)

        result_df = SearchIntQuantisationForClassification.get_result_df(
            study,
            q_config_parser=self.q_config_parser,
            q_config_formatter=self.q_config_formatter,
            num_hidden_layers=self.model_config.num_hidden_layers,
            stat_profile=stat_profile,
            range_entry=range_entry,
            save_dir=save_dir,
            alpha_s=self.search_config["search_estimator"]["alpha_acc"],
            alpha_h=self.search_config["search_estimator"]["alpha_mem_density"],
            compare_to=self.search_config["search_estimator"]["compare_to"],
        )
        # result_df.to_json(result_json_path, orient="index")
        result_df.drop("quant_config", axis=1).to_csv(result_table_path, index=False)
        joblib.dump(study, study_path)
        logger.info("========== Best Trials ==========")
        logger.info(
            f"(alpha_acc, alpha_mem_density) = {self.search_config['search_estimator']['alpha_acc']}, {self.search_config['search_estimator']['alpha_mem_density']}"
        )

        result_df = result_df.drop("quant_config", axis=1)
        result_df["quant_config_name"] = result_df["quant_config_path"].apply(
            lambda x: "$save_dir/quant_configs/" + str(Path(x).name)
        )
        result_df = result_df.drop("quant_config_path", axis=1)
        logger.info(
            "\n"
            + tabulate(
                result_df,
                headers="keys",
                tablefmt="pretty",
                floatfmt=(None, ".4f", ".2f", ".2f", None, ".4f", ".2f"),
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
        acc_threshold = self.search_config["search_strategy"]["acc_threshold"]
        avg_bitwidth_threshold = self.search_config["search_strategy"][
            "avg_bitwidth_threshold"
        ]
        sort_by = self.search_config["search_strategy"]["sort_by"]

        for i, s in enumerate(sort_by):
            assert s in [
                "acc",
                "accuracy",
                "avg_bitwidth",
            ], f"Unknown sort_by: {s}, must be one of ['acc', 'accuracy', 'avg_bitwidth']"
            if s == "acc":
                sort_by[i] = "accuracy"
        result_df = SearchIntQuantisationForClassification.get_result_df(
            study,
            q_config_parser=self.q_config_parser,
            q_config_formatter=self.q_config_formatter,
            num_hidden_layers=self.model_config.num_hidden_layers,
            stat_profile=stat_profile,
            range_entry=range_entry,
            save_dir=self.save_dir,
            alpha_s=self.search_config["search_estimator"]["alpha_acc"],
            alpha_h=self.search_config["search_estimator"]["alpha_mem_density"],
            compare_to=self.search_config["search_estimator"]["compare_to"],
        )

        filtered_df = result_df.loc[result_df["accuracy"] >= acc_threshold]
        filtered_df = filtered_df.loc[
            filtered_df["avg_bitwidth"] <= avg_bitwidth_threshold
        ]
        if len(filtered_df) == 0:
            logger.warning(
                f"No trials found with acc >= {acc_threshold} and avg_bitwidth <= {avg_bitwidth_threshold}"
            )
            return

        sort_by = [s if s != "avg_bitwidth" else "memory_density" for s in sort_by]
        filtered_df = filtered_df.sort_values(sort_by, ascending=False)

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
        self.q_bitwidth_profiler = get_bitwidth_profiler(model_arch)
        self.q_config_parser = get_quant_config_parser(model_arch)
        self.q_config_sampler = get_quant_config_sampler(model_arch)
        self.q_config_formatter = get_stat_config_formatter(model_arch)

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
        limit: int,
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
        ):
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
            metric = sum(acc_list) / len(acc_list)
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
            range_entry: str = "range_min_max",
        ):
            if self.search_config["search_space"]["extend_quant_config_seed_first"]:
                quant_config_seed = self.q_config_parser(
                    quant_config_seed, self.model_config.num_hidden_layers, strict=False
                )
            logger.debug(f"0️⃣ Quant Config Seed")
            logger.debug("\n" + pformat(quant_config_seed))
            sampled_config = self.q_config_sampler(
                trial=trial,
                name="root",
                config_seed=quant_config_seed,
            )
            # logger.debug(f"1️⃣ Sampled Config (before parsing):")
            # logger.debug("\n" + pformat(sampled_config))
            sampled_config = self.q_config_parser(
                sampled_config, self.model_config.num_hidden_layers, strict=False
            )
            # logger.debug(f"2️⃣ Sampled Config (after 2nd parsing):")
            # logger.debug("\n" + pformat(sampled_config))
            flattened_sampled_config = {}
            flatten_dict(sampled_config, new_d=flattened_sampled_config)
            # logger.debug(f"3️⃣ Flattened Sampled Config:")
            # logger.debug("\n" + pformat(flattened_sampled_config))
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
            # logger.debug(f"4️⃣ Config from stat_profile:")
            # logger.debug("\n" + pformat(sampled_config))
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

            avg_bitwidth = self.search_config["search_estimator"]["compare_to"] / ori_h_metric
            # fmt: on
            self.logger.info(
                f"{frozen_trail.number},"
                f"{ori_s_metric:.4f},{ori_h_metric:.4f},{avg_bitwidth:.2f},"
                f"{s_metric:.4f}, {h_metric:.4f}"
            )
            logger.info(
                f"Trial {frozen_trail.number} is done: "
                f"unscaled (acc, mem_density, avg_bitwidth) = "
                f"({ori_s_metric:.4f}, {ori_h_metric:.4f}, {avg_bitwidth:.2f}),"
                f"scaled (acc, mem_density) = "
                f"({s_metric:.4f}, {h_metric:.4f}), "
            )
            # logger.debug(f"============= Trial {frozen_trail.number} =============")
            # logger.debug(f"Trial {frozen_trail.number} config: {frozen_trail.params}")

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
            directions=["maximize", "maximize"],
            sampler=sampler,
        )

        # sample configs
        q_config_seed = self.search_config["search_space"]["quant_config_seed"]

        self.logger.info(
            f"trial,unscaled_acc,unscaled_mem_density,avg_bitwidth,scaled_acc,scaled_mem_density"
        )
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
                limit=limit,
                stat_profile=stat_profile,
                range_entry=range_entry,
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
        alpha_s: float,
        alpha_h: float,
        compare_to: int,
        sort_by=["accuracy", "memory_density"],
    ) -> pd.DataFrame:
        result_df = pd.DataFrame(
            columns=[
                "trial_id",
                "accuracy",
                "memory_density",
                "avg_bitwidth",
                "quant_config_path",
                "scaled_acc",
                "scaled_mem_density",
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
            alpha_acc = alpha_s
            alpha_mem_density = alpha_h
            scaled_s_metric, scaled_h_metric = trial.values[0], trial.values[1]
            s_metric = scaled_s_metric / alpha_acc
            h_metric = scaled_h_metric / alpha_mem_density
            avg_bitwidth = compare_to / h_metric
            result_df.loc[i] = [
                trial_id,
                s_metric,
                h_metric,
                avg_bitwidth,
                quant_config_path,
                scaled_s_metric,
                scaled_h_metric,
                quant_config,
            ]
        result_df = result_df.sort_values(by=sort_by, ascending=False)
        return result_df

    def save_study_and_results(self, study: optuna.Study, stat_profile, range_entry):
        save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        study_path = save_dir / "study.pkl"
        result_table_path = save_dir / "results.csv"
        search_config_path = save_dir / "search_config.toml"
        save_config(self.search_config, search_config_path)

        result_df = SearchIntQuantisationForPromptingCLS.get_result_df(
            study,
            q_config_parser=self.q_config_parser,
            q_config_formatter=self.q_config_formatter,
            num_hidden_layers=self.model_config.num_hidden_layers,
            stat_profile=stat_profile,
            range_entry=range_entry,
            save_dir=save_dir,
            alpha_s=self.search_config["search_estimator"]["alpha_acc"],
            alpha_h=self.search_config["search_estimator"]["alpha_mem_density"],
            compare_to=self.search_config["search_estimator"]["compare_to"],
        )
        # result_df.to_json(result_json_path, orient="index")
        result_df.drop("quant_config", axis=1).to_csv(result_table_path, index=False)
        joblib.dump(study, study_path)
        logger.info("========== Best Trials ==========")
        logger.info(
            f"(alpha_acc, alpha_mem_density) = {self.search_config['search_estimator']['alpha_acc']}, {self.search_config['search_estimator']['alpha_mem_density']}"
        )

        result_df = result_df.drop("quant_config", axis=1)
        result_df["quant_config_name"] = result_df["quant_config_path"].apply(
            lambda x: "$save_dir/quant_configs/" + str(Path(x).name)
        )
        result_df = result_df.drop("quant_config_path", axis=1)
        logger.info(
            "\n"
            + tabulate(
                result_df,
                headers="keys",
                tablefmt="pretty",
                floatfmt=(None, ".4f", ".2f", ".2f", None, ".4f", ".2f"),
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
        limit: int,
        stat_profile: dict,
        range_entry: str,
    ):
        acc_threshold = self.search_config["search_strategy"]["acc_threshold"]
        avg_bitwidth_threshold = self.search_config["search_strategy"][
            "avg_bitwidth_threshold"
        ]
        sort_by = self.search_config["search_strategy"]["sort_by"]

        for i, s in enumerate(sort_by):
            assert s in [
                "acc",
                "accuracy",
                "avg_bitwidth",
            ], f"Unknown sort_by: {s}, must be one of ['acc', 'accuracy', 'avg_bitwidth']"
            if s == "acc":
                sort_by[i] = "accuracy"
        result_df = SearchIntQuantisationForPromptingCLS.get_result_df(
            study,
            q_config_parser=self.q_config_parser,
            q_config_formatter=self.q_config_formatter,
            num_hidden_layers=self.model_config.num_hidden_layers,
            stat_profile=stat_profile,
            range_entry=range_entry,
            save_dir=self.save_dir,
            alpha_s=self.search_config["search_estimator"]["alpha_acc"],
            alpha_h=self.search_config["search_estimator"]["alpha_mem_density"],
            compare_to=self.search_config["search_estimator"]["compare_to"],
        )

        filtered_df = result_df.loc[result_df["accuracy"] >= acc_threshold]
        filtered_df = filtered_df.loc[
            filtered_df["avg_bitwidth"] <= avg_bitwidth_threshold
        ]
        if len(filtered_df) == 0:
            logger.warning(
                f"No trials found with acc >= {acc_threshold} and avg_bitwidth <= {avg_bitwidth_threshold}"
            )
            return

        sort_by = [s if s != "avg_bitwidth" else "memory_density" for s in sort_by]
        filtered_df = filtered_df.sort_values(sort_by, ascending=False)

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
            limit=limit,
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
