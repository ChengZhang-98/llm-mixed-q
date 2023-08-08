import os
import sys
from pathlib import Path
from pprint import pprint as pp

import os
from pathlib import Path

from transformers import default_data_collator
from torch.utils.data import DataLoader
from transformers.utils.logging import set_verbosity_error

sys.path.append(str(Path(__file__).parent.parent / "src"))

from llm_mixed_q.eval import eval_cls_glue
from llm_mixed_q.utils import set_logging_verbosity

from llm_mixed_q.models import (
    get_model_cls,
    get_config_cls,
    get_tokenizer_cls,
)
from llm_mixed_q.datasets import (
    get_num_labels,
    get_raw_dataset_dict,
    preprocess_dataset_dict,
    is_regression_task,
)
from llm_mixed_q.models.bert_quantized.quant_config_bert import (
    parse_bert_quantized_config,
    create_a_layer_config,
)
import toml
import torch


os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"


def test_bert_layer_parser():
    config = """
[default]
name = "integer"
bypass = false
is_ptq = true
data_in_width = 16
data_in_frac_width = 12
weight_width = 4
weight_frac_width = 4
bias_width = 4
bias_frac_width = 6

[model_layer_0.attention.query]
name = "integer"
bypass = false
is_ptq = true
data_in_width = 16
data_in_frac_width = 16
weight_width = 4
weight_frac_width = 4
bias_width = 4
bias_frac_width = 8

[model_layer_0.attention.key]
name = "integer"
bypass = false
is_ptq = true
data_in_width = 16
data_in_frac_width = 14
weight_width = 4
weight_frac_width = 6
bias_width = 4
bias_frac_width = 6

[model_layer_0.attention.value]
name = "integer"
bypass = false
is_ptq = true
data_in_width = 16
data_in_frac_width = 10
weight_width = 4
weight_frac_width = 7
bias_width = 4
bias_frac_width = 8

[model_layer_0.attention.matmul_0]
name = "integer"
bypass = false
is_ptq = true
data_in_width = 16
data_in_frac_width = 14
weight_width = 4
weight_frac_width = 8
bias_width = 4
bias_frac_width = 8

[model_layer_0.attention.matmul_1]
name = "integer"
bypass = false
is_ptq = true
data_in_width = 16
data_in_frac_width = 12
weight_width = 4
weight_frac_width = 5
bias_width = 4
bias_frac_width = 5

[model_layer_0.intermediate.dense]
name = "integer"
bypass = false
is_ptq = true
data_in_width = 16
data_in_frac_width = 14
weight_width = 4
weight_frac_width = 7
bias_width = 4
bias_frac_width = 8

[model_layer_0.output.dense]
name = "integer"
bypass = false
is_ptq = true
data_in_width = 16
data_in_frac_width = 13
weight_width = 4
weight_frac_width = 8
bias_width = 4


[model_layer_0.attention.output.dense]
name = "integer"
bypass = false
is_ptq = true
data_in_width = 16
data_in_frac_width = 15
weight_width = 4
weight_frac_width = 6
bias_width = 4
bias_frac_width = 8
"""

    """
    bias_frac_width = 8
    """

    config = toml.loads(config)
    parsed = create_a_layer_config(layer_qc=config["model_layer_0"], strict=False)
    pp(parsed)


if __name__ == "__main__":
    test_bert_layer_parser()
