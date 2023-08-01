import os
import sys
from pathlib import Path

import torch
from accelerate import init_empty_weights

sys.path.append(str(Path(__file__).parent.parent / "src"))
from llm_mixed_q.models import (
    get_model_cls,
    get_config_cls,
    get_tokenizer_cls,
    get_bitwidth_profiler,
)

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"


def test_bert():
    arch = "bert"
    task = "cls"
    name = "bert-base-uncased"
    quant_config = "./bert.toml"

    model_cls = get_model_cls(arch, task)
    config_cls = get_config_cls(arch)

    config = config_cls.from_pretrained(name, quant_config=quant_config)
    model = model_cls.from_pretrained(name, config=config, device_map=None).to("cuda")

    x = torch.randint(0, 1000, (16, 128))
    y = model(x)
    print(model.bert.encoder.layer[0].attention.self.query.weight[0, :10])


def test_llama():
    arch = "llama"
    task = "cls"
    name = "Cheng98/llama-160m"
    # name = "lmsys/vicuna-7b-v1.3"
    # quant_config = "./llama.toml"
    quant_config = "/home/zz7522/Projects/llm-mixed-q/checkpoints/asplos/search/llama_160m_sst2/1/best_quant_config.toml"
    # quant_config = "/home/zz7522/Projects/llm-mixed-q/checkpoints/asplos/search/vicuna_7b/buggy_default/best_quant_config.toml"

    model_cls = get_model_cls(arch, task)
    config_cls = get_config_cls(arch)

    config = config_cls.from_pretrained(name, quant_config=quant_config)
    model = model_cls.from_pretrained(name, config=config).to("cuda")
    breakpoint()
    x = torch.randint(0, 1000, (16, 128)).to("cuda")
    y = model(x)
    breakpoint()


def test_opt():
    arch = "opt"
    task = "cls"
    name = "facebook/opt-125m"
    quant_config = "./opt.toml"

    model_cls = get_model_cls(arch, task)
    config_cls = get_config_cls(arch)

    config = config_cls.from_pretrained(name, quant_config=quant_config)
    model = model_cls.from_pretrained(name, config=config).to("cuda:0")

    x = torch.randint(0, 1000, (16, 128)).to("cuda:0")
    y = model(x)

    profiler = get_bitwidth_profiler(arch)
    profile = profiler(config=config, seq_len=128)
    print(profile)
    print("avg act bitwidth:", profile["act_bits"] / profile["num_acts"])
    print("avg param bitwidth:", profile["param_bits"] / profile["num_params"])


if __name__ == "__main__":
    # test_bert()
    test_llama()
    # test_opt()
