import os
import sys
from pathlib import Path

import torch
from accelerate import init_empty_weights

sys.path.append(str(Path(__file__).parent.parent / "src"))
from llm_mixed_q.models import get_model_cls, get_config_cls

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"


def test_bert():
    arch = "bert"
    task = "cls"
    name = "bert-base-uncased"
    quant_config = "./bert.toml"

    model_cls = get_model_cls(arch, task)
    config_cls = get_config_cls(arch)

    config = config_cls.from_pretrained(name, quant_config=quant_config)
    model = model_cls.from_pretrained(name, config=config, device_map=None)

    x = torch.randint(0, 1000, (16, 128))
    y = model(x)
    print(model.bert.encoder.layer[0].attention.self.query.weight[0, :10])


def test_llama():
    arch = "llama"
    task = "cls"
    name = "Cheng98/llama-160m"
    quant_config = "./llama.toml"

    model_cls = get_model_cls(arch, task)
    config_cls = get_config_cls(arch)

    config = config_cls.from_pretrained(name, quant_config=quant_config)
    model = model_cls.from_pretrained(name, config=config, device_map=None)

    breakpoint()


if __name__ == "__main__":
    # test_bert()
    test_llama()
