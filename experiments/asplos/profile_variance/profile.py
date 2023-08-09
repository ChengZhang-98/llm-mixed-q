import os
import sys
from pathlib import Path
import transformers
from transformers import DataCollatorForLanguageModeling
import datasets as hf_datasets
from torch.utils.data import DataLoader
import json
import pickle
from argparse import ArgumentParser

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from llm_mixed_q.cli import cli_eval_cls_glue
from llm_mixed_q.utils import set_logging_verbosity
from llm_mixed_q.eval import eval_lm_wikitext2
from llm_mixed_q.datasets import get_raw_dataset_dict, preprocess_dataset_dict

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from llm_mixed_q.statstic_profiler import StatManager
from models.modeling_llama import LlamaForCausalLM
from models.identity import Identity


def main():
    parser = ArgumentParser()
    parser.add_argument("--save_name", type=str, default="variance_online.pkl")
    args = parser.parse_args()

    save_name = Path(args.save_name)
    save_name.parent.mkdir(exist_ok=True, parents=True)

    stat_manager = StatManager(act_stats=("variance_online",), weight_stats=())

    model = LlamaForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.3").to("cuda:0")
    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        "lmsys/vicuna-7b-v1.3", legacy=False
    )
    tokenizer.pad_token = tokenizer.eos_token

    raw_dataset_dict = get_raw_dataset_dict("wikitext2")
    preprocessed_dataset_dict = preprocess_dataset_dict(
        raw_dataset_dict,
        task="wikitext2",
        tokenizer=tokenizer,
        padding="max_length",
        max_length=2048,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    eval_dataloader = DataLoader(
        preprocessed_dataset_dict["train"],
        batch_size=1,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=os.cpu_count(),
    )

    for _, layer in model.named_modules():
        if isinstance(layer, Identity):
            entry_name = layer.name
            layer_id = layer.block_id
            layer_name = f"model_layer_{layer_id}:{entry_name}"
            if not "matmul" in layer.name:
                layer.register_forward_hook(
                    stat_manager.get_post_forward_act_hook(layer_name)
                )
            else:
                layer.register_forward_pre_hook(
                    stat_manager.get_pre_forward_act_hook(layer_name)
                )

    eval_lm_wikitext2(
        model=model,
        eval_dataloader=eval_dataloader,
        num_samples=64,
        progress_bar=True,
    )
    results = stat_manager.finalize(show_progress_bar=True)
    with open(save_name, "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
