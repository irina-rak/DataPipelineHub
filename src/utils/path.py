import json
import os

from pathlib import Path
from typing import Dict, Any, Tuple, List

from omegaconf import OmegaConf


def get_configurations(conf_file_path: Path) -> Dict[str, Any]:
    """
        Load configurations from the specified file or directory.

        Args:
            conf_file_path (Path): Path to the configuration file or directory.

        Returns:
            Dict[str, Any]: A dictionary containing the loaded configurations.

        Raises:
            ValueError: If the specified path is None.
            IsADirectoryError: If the specified path is a directory.
            FileNotFoundError: If the specified file does not exist.
    """

    config = {}
    if conf_file_path is None:
        raise ValueError("No config file provided")
    if conf_file_path.is_file():
        config = OmegaConf.load(conf_file_path)
    elif conf_file_path.is_dir():
        raise IsADirectoryError("Config file is a directory")
    elif not conf_file_path.exists():
        raise FileNotFoundError("Config file does not exist")

    return config


def validate_path(path: Path) -> None:
    """
        Validate if the provided path exists and is a directory.

        Args:
            path (Path): The path to validate.

        Returns:
            None

        Raises:
            NotADirectoryError: If the provided path does not exist or is not a directory.
    """

    if not os.path.exists(path):
        raise NotADirectoryError(f"{path} is not a directory")


def create_dir(dir_parent: Path, dir_name: str) -> Path:
    """
        Create a new directory within the specified parent directory.

        Args:
            dir_parent (Path): The path to the parent directory where the new directory will be created.
            dir_name (str): The name of the new directory to be created.

        Returns:
            Path: The path object representing the newly created directory.
    """

    validate_path(dir_parent)

    new_dir = os.path.join(dir_parent, dir_name)
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

    return Path(new_dir)


def get_data_paths(directory: Path) -> Tuple[List[Path], List[Path]]:
    """
        Get paths of image and label files within the specified directory.

        Args:
            directory (Path): The directory containing image and label files.

        Returns:
            Tuple[List[Path], List[Path]]: A tuple containing lists of image and label file paths.
    """

    images = [Path(os.path.join(directory, f)) for f in os.listdir(directory) if "_image" in f]
    labels = [Path(os.path.join(directory, f)) for f in os.listdir(directory) if "_label" in f]

    return images, labels


def load_img_params(path: Path) -> Dict[str, Any]:
    """
    Load image processing parameters from a JSON file.

    This function reads image processing parameters from the specified JSON file and returns them.

    Args:
        path (Path): The path to the JSON file containing the image processing parameters.

    Returns:
        Dict[str, Any]: The loaded image processing parameters.
    """

    validate_path(path)
    with open(path) as file:
        return json.load(file)
