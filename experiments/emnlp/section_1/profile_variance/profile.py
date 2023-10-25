import os
import pickle
import sys
from argparse import ArgumentParser
from pathlib import Path

import transformers
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

print(str(Path(__file__).parents[4].resolve() / "src"))
sys.path.append(str(Path(__file__).parents[4].resolve() / "src"))

from llm_mixed_q.datasets import get_raw_dataset_dict, preprocess_dataset_dict
from llm_mixed_q.eval import eval_lm_wikitext2

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from models.identity import Identity
from models.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM

from llm_mixed_q.statstic_profiler import StatManager


def main():
    parser = ArgumentParser()
    parser.add_argument("--save_name", type=str, default="variance_online.pkl")
    args = parser.parse_args()

    save_name = Path(args.save_name)
    save_name.parent.mkdir(exist_ok=True, parents=True)

    stat_manager = StatManager(
        act_stats=("variance_online",), weight_stats=("variance_precise",)
    )

    # fmt: off
    device_map = {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 1, 'model.layers.12': 1, 'model.layers.13': 1, 'model.layers.14': 1, 'model.layers.15': 1, 'model.layers.16': 1, 'model.layers.17': 1, 'model.layers.18': 1, 'model.layers.19': 1, 'model.layers.20': 1, 'model.layers.21': 1, 'model.layers.22': 1, 'model.layers.23': 2, 'model.layers.24': 2, 'model.layers.25': 2, 'model.layers.26': 2, 'model.layers.27': 2, 'model.layers.28': 2, 'model.layers.29': 2, 'model.layers.30': 2, 'model.layers.31': 2, 'model.norm': 2, 'lm_head': 2}
    # fmt: on
    model = LlamaForCausalLM.from_pretrained(
        "lmsys/vicuna-7b-v1.3", device_map=device_map
    )
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
            layer.register_forward_pre_hook(
                stat_manager.get_pre_forward_act_hook(layer_name)
            )

        if isinstance(layer, LlamaDecoderLayer):
            layer_id = layer.layer_id

            attn = layer.self_attn
            mlp = layer.mlp
            # fmt: off
            attn.q_proj.register_forward_pre_hook(stat_manager.get_pre_forward_weight_hook(f"model_layer_{layer_id}:q_proj_weight", "weight"))
            attn.k_proj.register_forward_pre_hook(stat_manager.get_pre_forward_weight_hook(f"model_layer_{layer_id}:k_proj_weight", "weight"))
            attn.v_proj.register_forward_pre_hook(stat_manager.get_pre_forward_weight_hook(f"model_layer_{layer_id}:v_proj_weight", "weight"))
            attn.o_proj.register_forward_pre_hook(stat_manager.get_pre_forward_weight_hook(f"model_layer_{layer_id}:o_proj_weight", "weight"))
            mlp.up_proj.register_forward_pre_hook(stat_manager.get_pre_forward_weight_hook(f"model_layer_{layer_id}:up_proj_weight", "weight"))
            mlp.down_proj.register_forward_pre_hook(stat_manager.get_pre_forward_weight_hook(f"model_layer_{layer_id}:down_proj_weight", "weight"))
            mlp.gate_proj.register_forward_pre_hook(stat_manager.get_pre_forward_weight_hook(f"model_layer_{layer_id}:gate_proj_weight", "weight"))
            # fmt: on

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
