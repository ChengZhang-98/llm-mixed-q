import os
import sys
from pathlib import Path
from pprint import pprint as pp
import random
import joblib

from accelerate import init_empty_weights
from torch.utils.data import DataLoader
from transformers import default_data_collator

sys.path.append(str(Path(__file__).parent.parent / "src"))
from llm_mixed_q.models import get_model_cls, get_config_cls, get_tokenizer_cls
from llm_mixed_q.models.llama_quantized.quant_config_llama import (
    parse_llama_quantized_config,
)
from llm_mixed_q.models.quantize.quant_config_parser import parse_node_config
from llm_mixed_q.search.search import SearchQuantisationForClassification
from llm_mixed_q.eval import eval_cls_glue
from llm_mixed_q.datasets import (
    get_num_labels,
    get_raw_dataset_dict,
    preprocess_dataset_dict,
)
from llm_mixed_q.utils import set_logging_verbosity

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"


def extract_quant_config_llama_160m():
    # study_pkl = "/home/zz7522/Projects/llm-mixed-q/checkpoints/asplos/table_sampler_comparison/llama_160m_sst2/random/study.pkl"
    # study_pkl = "/home/zz7522/Projects/llm-mixed-q/checkpoints/asplos/table_sampler_comparison/llama_160m_sst2/tpe_1/study.pkl"
    # study_pkl = "/home/zz7522/Projects/llm-mixed-q/checkpoints/asplos/table_sampler_comparison/llama_160m_sst2/nsgaii_1/study.pkl"
    study_pkl = "/home/zz7522/Projects/llm-mixed-q/checkpoints/asplos/table_sampler_comparison/llama_160m_sst2/nsgaiii_0/study.pkl"
    # save_path = "./extracted_quant_config_llama_160m_sst2.toml"
    save_path = None
    target_idx = 113
    model_arch = "llama"
    model_name = (
        "/home/zz7522/Projects/llm-mixed-q/checkpoints/asplos/fine_tune/llama_160m_sst2"
    )

    with open(study_pkl, "rb") as f:
        study = joblib.load(f)

    target_trial = study.trials[target_idx]
    quant_config = SearchQuantisationForClassification.save_trial_to_quant_config(
        target_trial, save_path
    )

    tokenizer = get_tokenizer_cls(model_arch).from_pretrained(model_name)
    config = get_config_cls(model_arch).from_pretrained(
        model_name, quant_config=quant_config
    )
    model = (
        get_model_cls(model_arch, task="cls")
        .from_pretrained(model_name, config=config)
        .to("cuda:2")
    )

    raw_dataset_dict = get_raw_dataset_dict("sst2")
    preprocessed_dataset_dict = preprocess_dataset_dict(
        raw_dataset_dict,
        task="sst2",
        tokenizer=tokenizer,
        padding="max_length",
        max_length=196,
    )
    eval_dataloader = DataLoader(
        preprocessed_dataset_dict["validation"],
        batch_size=32,
        collate_fn=default_data_collator,
        shuffle=False,
    )
    results = eval_cls_glue(
        model,
        task="sst2",
        eval_dataloader=eval_dataloader,
        is_regression=False,
        progress_bar=True,
    )
    print(results)


def extract_quant_config_bert_base():
    study_pkl = "/home/zz7522/Projects/llm-mixed-q/checkpoints/asplos/search/bert_base_sst2/random_0/study.pkl"
    save_path = None
    target_idx = 121
    model_arch = "bert"
    model_name = (
        "/home/zz7522/Projects/llm-mixed-q/checkpoints/asplos/fine_tune/bert_base_sst2"
    )

    with open(study_pkl, "rb") as f:
        study = joblib.load(f)

    target_trial = study.trials[target_idx]
    quant_config = SearchQuantisationForClassification.save_trial_to_quant_config(
        target_trial, save_path
    )

    tokenizer = get_tokenizer_cls(model_arch).from_pretrained(model_name)
    config = get_config_cls(model_arch).from_pretrained(
        model_name, quant_config=quant_config
    )
    model = (
        get_model_cls(model_arch, task="cls")
        .from_pretrained(model_name, config=config)
        .to("cuda:2")
    )

    raw_dataset_dict = get_raw_dataset_dict("sst2")
    preprocessed_dataset_dict = preprocess_dataset_dict(
        raw_dataset_dict,
        task="sst2",
        tokenizer=tokenizer,
        padding="max_length",
        max_length=196,
    )
    eval_dataloader = DataLoader(
        preprocessed_dataset_dict["validation"],
        batch_size=32,
        collate_fn=default_data_collator,
        shuffle=False,
    )
    results = eval_cls_glue(
        model,
        task="sst2",
        eval_dataloader=eval_dataloader,
        is_regression=False,
        progress_bar=True,
    )
    print(results)
    breakpoint()


if __name__ == "__main__":
    set_logging_verbosity("info")
    extract_quant_config_llama_160m()
    # extract_quant_config_bert_base()
