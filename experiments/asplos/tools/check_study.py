import optuna
from pathlib import Path
import joblib
import argparse


def check_search_duration(study: optuna.Study):
    num_trials = len(study.trials)
    start_time = study.trials[0].datetime_start
    end_time = study.trials[-1].datetime_start
    average_duration = (end_time - start_time) / (num_trials - 1)
    overall_duration = average_duration * num_trials
    print(f"Number of trials: {num_trials}")
    print(f"Average duration: {average_duration}")
    print(f"Overall duration: {overall_duration}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("study_path", type=str)
    args = parser.parse_args()
    study_path = Path(args.study_path)
    study = joblib.load(study_path)
    print("Checking the study at {}".format(study_path))
    check_search_duration(study)


if __name__ == "__main__":
    main()
