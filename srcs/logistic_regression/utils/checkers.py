import argparse
import os


def check_path_mode(args: argparse.Namespace) -> tuple[str, str]:
    """
    Check the path and mode provided in the arguments

    Args : argparse.Namespace
        The arguments provided in the command line.

    Returns : tuple[str, str]
    """
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"File not found: {args.csv_path}")
    if not args.csv_path.endswith(".csv"):
        raise ValueError("Invalid file type. Use a CSV file.")
    if not os.access(args.csv_path, os.R_OK):
        raise PermissionError("No Read Permissions")

    if args.mode not in ["train", "predict", "describe"]:
        raise ValueError("Invalid mode. Use either 'train' or 'predict'.")

    return args.csv_path, args.mode


def check_model_path(response: str) -> str:
    """
    Check the model path provided by the user

    Args : str
        The path provided by the user

    Returns : str
    """
    if not os.path.exists(response):
        raise FileNotFoundError(f"File not found: {response}")
    if not response.endswith(".pkl"):
        raise ValueError("Invalid file type. Use a pickle file.")
    if not os.access(response, os.R_OK):
        raise PermissionError("No Read Permissions")

    return response


def check_save_path(response: str) -> str:
    """
    Check the save path provided by the user

    Args : str
        The path provided by the user

    Returns : str
    """
    if not os.path.exists(response):
        try:
            os.makedirs(response, exist_ok=True)
        except PermissionError as e:
            raise PermissionError("No Permissions") from e
    if not os.path.isdir(response):
        raise ValueError("Invalid directory path.")

    return response
