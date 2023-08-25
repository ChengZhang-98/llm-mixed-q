from pathlib import Path
from argparse import ArgumentParser
import pickle
import pandas as pd


def main():
    parser = ArgumentParser()
    parser.add_argument("--pkl", type=str, default="variance_online.pkl")
    parser.add_argument("--csv", type=str, default="variance_online.csv")
    args = parser.parse_args()

    pkl = Path(args.pkl)
    csv = Path(args.csv)
    csv.parent.mkdir(exist_ok=True, parents=True)

    with open(pkl, "rb") as f:
        results: dict = pickle.load(f)

    df = pd.DataFrame(columns=["block_id", "layer_name", "variance"])

    for entry, var in results.items():
        block_id, layer_name = entry.split(":")
        block_id = int(block_id.removeprefix("model_layer_"))

        if "variance_online" in var:
            var_i = var["variance_online"]["variance"]
        elif "variance_precise" in var:
            var_i = var["variance_precise"]["variance"]
        else:
            raise ValueError(f"Unknown variance type: {var.keys()}")

        df.loc[len(df)] = [
            block_id,
            layer_name,
            var_i,
        ]
    df.to_csv(csv, index=False)


if __name__ == "__main__":
    main()
