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
    model = model_cls.from_pretrained(name, config=config)

    print(model)
    print(model.bert.encoder.layer[0].attention.self.query.weight[0, :10])

    x = torch.randint(0, 1000, (2, 128))
    y = model(x)
    print(model.bert.encoder.layer[0].attention.self.query.weight[0, :10])


if __name__ == "__main__":
    test_bert()
