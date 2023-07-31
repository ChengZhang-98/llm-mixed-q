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
    get_q_profiler,
)
from llm_mixed_q.statstic_profiler.stat_manager import StatManager

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"


def test_stat_manager():
    import torch

    fc = torch.nn.Linear(10, 20).to("cuda")

    stat_manager = StatManager(
        act_stats=["range_min_max"], weight_stats=["range_min_max"]
    )

    fc.register_forward_pre_hook(stat_manager.get_pre_forward_act_hook_("fc.data_in"))
    fc.register_forward_pre_hook(
        stat_manager.get_pre_forward_weight_hook(name="fc.weight", weight_name="weight")
    )
    fc.register_forward_hook(stat_manager.get_post_forward_act_hook_("fc.data_out"))

    x = torch.randn((16, 10)).to("cuda")

    y = fc(x)

    stats = stat_manager.finalize()
    print(stats)


if __name__ == "__main__":
    test_stat_manager()
