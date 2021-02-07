import sys
import argparse
from .train import train as _train


def format_msg(msg, inputs=None):
    if inputs:
        msg_ = f"{msg} {inputs}!"
    else:
        msg_ = f"{msg}!"
    return msg_


def train(inputs=None):
    msg = "Training"
    msg_ = format_msg(msg, inputs=inputs)
    print(msg_)


def evaluate(inputs=None):
    msg = "Evaluating"
    msg_ = format_msg(msg, inputs=inputs)
    print(msg_)


def predict(inputs=None):
    msg = "Predicting"
    msg_ = format_msg(msg, inputs=inputs)
    print(msg_)


def main():
    parser = argparse.ArgumentParser(description="Enter task")
    parser.add_argument("task", type=str, help="Enter task")
    parser.add_argument("inputs", type=str, help="Inputs data for task")
    args = parser.parse_args()

    task_dict = {
        "train": train,
        "evaluate": evaluate,
        "predict": predict,
    }

    fun = task_dict.get(args.task)

    return fun(args.inputs)


if __name__ == "__main__":
    sys.exit(main())
