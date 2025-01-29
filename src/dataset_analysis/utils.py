from os import listdir, makedirs
from os.path import isdir, isfile, join
from typing import Dict, List, Union

import cv2
import numpy as np
import pandas as pd

from rich.progress import Progress


def get_images(path: str, extensions: Union[str, List[str]]) -> List[str]:
    """Gets all images in a directory with the given extensions.

    Args:
    -----
        path (str): The path to the directory.
        extensions (Union[str, List[str]]): The extensions of the images.

    Returns:
    --------
        List[str]: A list of the paths to the images.
    """
    files = listdir(path)

    images = []
    if type(extensions) == str:
        extensions = [extensions]
    for file in files:
        if file.split(".")[-1] in extensions:
            images.append(file)
    return images


def get_images_stats(path: str, img_list: List[str]) -> Dict[str, Dict[str, float]]:
    """Gets the statistics of the list of images in the given directory.
    It calculates the mean, standard deviation, min, and max of the images.

    Args:
    -----
        path (str): The path to the directory.
        img_list (List[str]): The paths to the images.

    Returns:
    --------
        Dict[str, Dict[str, float]]: A dictionary containing the statistics of the images.
    """
    stats = {}

    with Progress() as progress:
        task = progress.add_task(
            "[cyan]Calculating the statistics...", total=len(img_list)
        )
        for img in img_list:
            image = cv2.imread(join(path, img), cv2.IMREAD_GRAYSCALE)
            # image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

            stats[img] = {
                "mean": np.mean(image),
                "std": np.std(image),
                "median": np.median(image),
                "min": np.min(image),
                "max": np.max(image),
            }
            for key, value in stats[img].items():
                stats[img][key] = np.round(value, 4)
            progress.update(task, advance=1)

    return stats


def get_histograms(path: str, img_list: List[str]) -> np.ndarray:
    """Gets the histograms of the list of images in the given directory.

    Args:
    -----
        path (str): The path to the directory.
        img_list (List[str]): The paths to the images.

    Returns:
    --------
        np.ndarray: An array containing the histograms of the images.
    """
    hists = []

    with Progress() as progress:
        task = progress.add_task(
            "[cyan]Calculating the histograms...", total=len(img_list)
        )
        for img in img_list:
            image = cv2.imread(join(path, img), cv2.IMREAD_GRAYSCALE)

            # hists[img] = {"hist": cv2.calcHist([image], [0], None, [256], [0, 256])}
            hists.append(cv2.calcHist([image], [0], None, [256], [0, 256]))
            progress.update(task, advance=1)

    return np.array(hists)


def save_stats(stats: Dict[str, Dict[str, float]], output_dir: str, file_name: str):
    """Saves the statistics of the images to a file.

    Args:
    -----
        stats (Dict[str, Dict[str, float]): The statistics of the images.
        output_dir (str): The path to the file where the statistics will be saved.
        file_name (str): The name of the file.

    Returns:
    --------
        None
    """
    if not isdir(output_dir):
        makedirs(output_dir)

    df = pd.DataFrame(stats).T
    df.index.name = "Image"
    df.to_csv(join(output_dir, file_name))
