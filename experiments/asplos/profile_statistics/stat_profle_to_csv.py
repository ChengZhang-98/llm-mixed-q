from argparse import ArgumentParser
from pathlib import Path
import toml
import pandas as pd


def main():
    parser = ArgumentParser()

    parser.add_argument("--stat_profile", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--num_blocks", type=int, default=32)
    parser.add_argument("--var_entry", type=str, default="variance_online")

    args = parser.parse_args()

    stat_profile = toml.load(args.stat_profile)
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(columns=["block_id", "layer", "weight", "data_in"])

    for block_id in range(args.num_blocks):
        for layer_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            new_row = [block_id, layer_name]
            for tensor_name in ["weight", "data_in"]:
                entry = (
                    f"root:model_layer_{block_id}:self_attn:{layer_name}:{tensor_name}"
                )
                new_row.append(stat_profile[entry]["variance_online"]["variance"])
            df.loc[len(df)] = new_row
        for layer_name in ["gate_proj", "down_proj", "up_proj"]:
            new_row = [block_id, layer_name]
            for tensor_name in ["weight", "data_in"]:
                entry = f"root:model_layer_{block_id}:mlp:{layer_name}:{tensor_name}"
                new_row.append(stat_profile[entry]["variance_online"]["variance"])
            df.loc[len(df)] = new_row

    df.to_csv(save_path, index=False)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
