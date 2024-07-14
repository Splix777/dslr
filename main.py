import argparse

from srcs.logistic_regression.logreg_train import LogRegTrain
from srcs.logistic_regression.logreg_predict import LogRegPredict
from srcs.data_analysis.describe import describe
from srcs.logistic_regression.utils.checkers import (
    check_path_mode,
    check_model_path,
    check_save_path
)


def train_or_predict(csv_path: str, mode: str) -> None:
    """
    Train or predict the model based on the mode

    Args:
        csv_path (str): path to the csv file
        mode (str): mode to run the model in

    Raises:
        ValueError: If mode is not 'train' or 'predict'
    """
    if mode == "describe":
        describe(csv_path=csv_path)
    elif mode == "predict":
        predict(csv_path=csv_path)
    elif mode == "train":
        train(csv_path=csv_path)
    else:
        raise ValueError("Invalid mode. Use 'train', 'predict' 'describe'")


def predict(csv_path: str) -> None:
    """
    Predict the house of the students in the csv file
    and print the evaluation metrics

    Args:
        csv_path (str): path to the csv file

    Returns:
        None
    """
    response = input("Please indicate model path: ")
    model_path = check_model_path(response)
    correct, accuracy = LogRegPredict(csv_path, model_path).predict()
    print(f"Predictions saved to the same directory as {csv_path}")
    print(f"Number of correct predictions: {correct}")
    print(f"Accuracy: {accuracy * 100:.2f}%")


def train(csv_path: str) -> None:
    """
    Train the model and print the evaluation metrics

    Args:
        csv_path (str): path to the csv file

    Returns:
        None
    """
    response: str = input("Where would you like to save the model?: ")
    save_path = check_save_path(response)
    plot = input("Do you want to plot the results? (y/n): ")
    logreg = (
        LogRegTrain(csv_path, save_path, plot=True)
        if plot.lower() == "y"
        else LogRegTrain(csv_path, save_path)
    )
    evaluation = logreg.train()
    for house, metrics in evaluation.items():
        print_training_evaluation(house, metrics)


def print_training_evaluation(house: str, metrics: dict) -> None:
    """
    Print the evaluation metrics

    Args:
        house (str): house name
        metrics (dict): evaluation metrics

    Returns:
        None
    """
    print('-' * 50)
    print(f"House: {house}")
    print(f"Accuracy: {metrics['accuracy']: .2f}%")

    print(f"precision: {metrics['precision']}")
    print("Precision is the ratio of correctly predicted positive observations"
          " to the total predicted positives. High precision indicates a low "
          "false positive rate.")

    print(f"recall: {metrics['recall']}")
    print("Recall, or sensitivity, is the ratio of correctly predicted "
          "positive observations to all observations in the actual class. High"
          " recall indicates a low false negative rate.")

    print(f"f1: {metrics['f1']}")
    print("The F1 Score is the weighted average of Precision and Recall. It "
          "provides a balance between Precision and Recall, especially useful "
          "for imbalanced classes.")


def main():
    """"
    Main function to run the program
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--csv_path",
        help="Path to the train or predict csv file",
        type=str,
        required=True)
    arg_parser.add_argument(
        "--mode",
        help="train, predict or describe",
        type=str,
        required=True)
    args = arg_parser.parse_args()

    csv_path, mode = check_path_mode(args)
    train_or_predict(csv_path=csv_path, mode=mode)


if __name__ == "__main__":
    """🧙🏼"""
    try:
        main()

    except Exception as e:
        print(e)
