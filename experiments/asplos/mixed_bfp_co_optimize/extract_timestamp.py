import optuna
import pandas as pd
from pathlib import Path
import argparse
import joblib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=str, required=True)
    parser.add_argument("--save_name", type=str, default="timestamp.csv")

    args = parser.parse_args()

    save_name = Path(args.save_name)
    save_name.parent.mkdir(exist_ok=True, parents=True)

    with open(args.study, "rb") as f:
        study: optuna.Study = joblib.load(f)

    df = pd.DataFrame(columns=["trial_id", "datetime_start", "datetime_end"])
    for i, trial in enumerate(study.trials):
        df.loc[len(df)] = [
            i,
            trial.datetime_start.strftime("%Y-%m-%d %H:%M:%S"),
            trial.datetime_complete.strftime("%Y-%m-%d %H:%M:%S"),
        ]

    df.to_csv(save_name, index=False)


if __name__ == "__main__":
    main()
