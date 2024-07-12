import argparse

from srcs.logistic_regression.logreg_train import LogRegTrain
from srcs.logistic_regression.logreg_predict import LogRegPredict


def train_or_predict(csv_path: str, mode: str) -> None:
    """
    Train or predict the model based on the mode

    Parameters
    ----------
    csv_path : str
        The path to the CSV file
    mode : str
        The mode to run the script
    """
    if mode == "predict":
        response = input("Please indicate model path: ")
        model_path = response
        LogRegPredict(csv_path, model_path)

    elif mode == "train":
        response = input("Please indicate model save path: ")
        model_save = response
        logreg = LogRegTrain(csv_path, model_save)
        evaluation = logreg.train()
        for house, metrics in evaluation.items():
            print_training_evaluation(house, metrics)
    else:
        raise ValueError("Invalid mode. Use either 'train' or 'predict'.")


def print_training_evaluation(house, metrics):
    print('-' * 50)
    print(f"House: {house}")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"mae: {metrics['mae']}")
    print(f"mse: {metrics['mse']}")
    print(f"rmse: {metrics['rmse']}")
    print(f"r2: {metrics['r2']}")
    print(f"precision: {metrics['precision']}")
    print(f"recall: {metrics['recall']}")
    print(f"f1: {metrics['f1']}")


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--csv_path", type=str, required=True)
    arg_parser.add_argument("--mode", type=str, required=True)
    args = arg_parser.parse_args()

    train_or_predict(args.csv_path, args.mode)


if __name__ == "__main__":
    main()