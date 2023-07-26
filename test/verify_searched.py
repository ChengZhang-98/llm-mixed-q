import os
import sys
from pathlib import Path

import torch
from accelerate import init_empty_weights

sys.path.append(str(Path(__file__).parent.parent / "src"))
from llm_mixed_q.models import get_model_cls, get_config_cls

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"


def test_searched_bert_base():
    model_cls = get_model_cls("bert", "cls")
    config_cls = get_config_cls("bert")

    ckpt = (
        "/data/zz7522/Projects/llm-mixed-q/checkpoints/asplos/fine_tune/bert_base_sst2"
    )
    quant_config = "/data/zz7522/Projects/llm-mixed-q/checkpoints/asplos/search/bert_base_sst2/0/best_quant_config.toml"
    quant_config_bypass = "/data/zz7522/Projects/llm-mixed-q/experiments/asplos/configs/quantize/bypass.toml"
    config = config_cls.from_pretrained(
        ckpt,
        quant_config=quant_config,
    )
    config_bypass = config_cls.from_pretrained(
        ckpt,
        quant_config=quant_config_bypass,
    )

    model_q = model_cls.from_pretrained(ckpt, config=config)
    model = model_cls.from_pretrained(ckpt, config=config_bypass)

    with torch.no_grad():
        x = torch.randint(0, 1000, (2, 128))
        _ = model(x)
        print(model.bert.encoder.layer[0].attention.self.query.weight[0, :10])

        _ = model_q(x)
        print(model_q.bert.encoder.layer[0].attention.self.query.weight[0, :10])


if __name__ == "__main__":
    test_searched_bert_base()
