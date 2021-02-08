# Authors: Léo Grosjean <leo.grosjean@live.fr>
# License: GPL3

import sys
import argparse
from pathlib import Path

from .train import train
from .predict import predict
from .evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(
        description="Arguments for task (training, evaluate, predict)"
    )
    parser.add_argument("task", choices=["train", "evaluate", "predict"])
    parser.add_argument("model", help="choice of model", choices=["model1", "model2"])
    parser.add_argument("data_dir", help="location of data directory")
    parser.add_argument(
        "save_dir", help="location of the directory to save both mlflow runs and model"
    )
    parser.add_argument("--kwargs", help="extra args", nargs="*", default={})

    task_dict = {
        "train": train,
        "evaluate": evaluate,
        "predict": predict,
    }

    args = parser.parse_args()
    args_dict = vars(args)
    keyword_args = args_dict["kwargs"]
    model = args_dict.get("model")
    data_dir = args_dict.get("data_dir")
    if data_dir is not None:
        data_dir = Path(data_dir).absolute()
    root_dir = args_dict.get("save_dir")

    fun = task_dict.get(args.task)

    return fun(model=model, data_dir=data_dir, root_dir=root_dir, **keyword_args)


if __name__ == "__main__":
    sys.exit(main())
