import os
import re
import shutil

from itertools import product
from os.path import isdir, join
from typing import List, Dict

import numpy as np

from rich.progress import Progress


def replicate_dirs(
    directory: str, num_clients: int, ignored_dirs: List[str]
) -> Dict[str, List[str]]:
    """
    Creates a dictionary of directories that replicates the structure of the input directory.

    Args:
    -----
        directory (str): Path to the directory.
        num_clients (int): Number of clients.
        ignored_dirs (List[str]): List of directories to be ignored.

    Returns:
    --------
        clients (dict): Dictionary of clients and their respective directories.
    """
    clients = {}
    for idx in range(num_clients):
        client_name = f"client_{idx + 1}"
        clients[client_name] = {}

        for _dir in os.listdir(directory):
            dir_path = join(directory, _dir)
            if isdir(dir_path) and _dir not in ignored_dirs:
                clients[client_name][_dir] = {}
                for _data in os.listdir(dir_path):
                    if isdir(join(dir_path, _data)) and _data not in ignored_dirs:
                        clients[client_name][_dir][_data] = []
                    else:
                        clients[client_name][_dir] = []

    return clients


def split_files(
    paths: List[str], clients: dict, num_clients: int
) -> Dict[str, List[str]]:
    """
    Splits the files in the dataset into num_clients clients based on the directory structure (i.e., dictionary keys in clients).

    Args:
    -----
        structure (dict): Dictionary containing the directory structure of the future clients.
        num_clients (int): Number of clients.

    Returns:
    --------
        clients (dict): Dictionary of clients and their respective data.
    """

    classes = "|".join(clients["client_1"]["train"].keys())
    for pth in paths:
        _set = re.search(r"(train|val|test)", pth).group(0)
        _cat = re.search(classes, pth).group(0)

        files = os.listdir(pth)
        files_distribution = [len(files) // num_clients] * num_clients

        if sum(files_distribution) < len(files):
            files_distribution[0] += len(files) - sum(files_distribution)

        for idx, client_id in enumerate(clients.keys()):
            # clients[client_id][_set][_cat] = files[: files_distribution.pop(0)]
            # files = files[files_distribution[0] :]

            # Take random files_distribution[idx] files from the list of files
            # and remove them from the list
            indices = np.random.choice(
                len(files), files_distribution[idx], replace=False
            )
            clients[client_id][_set][_cat] = [files[i] for i in indices]
            files = [f for i, f in enumerate(files) if i not in indices]

    return clients


def save_clients(input_root_dir, clients: dict, save_path: str) -> None:
    """
    Save the clients' data into separate directories.

    Args:
    -----
        clients (dict): Dictionary of clients and their respective data.
        save_path (str): Path to save the clients' data.
    """
    with Progress() as progress:
        file_nb = sum(
            len(clients[client][_dir][_data])
            for client in clients
            for _dir in clients[client]
            for _data in clients[client][_dir]
        )
        task = progress.add_task(
            f"Splitting dataset into {len(clients)} clients...", total=file_nb
        )
        for client in clients:
            for _dir in clients[client]:
                for _data in clients[client][_dir]:
                    data_path = join(save_path, client, _dir, _data)
                    os.makedirs(data_path, exist_ok=True)
                    for file in clients[client][_dir][_data]:
                        shutil.copy(
                            join(input_root_dir, _dir, _data, file),
                            join(data_path, file),
                        )
                        progress.update(task, advance=1)
            progress.console.print(f"Client {client} data saved.")


def split_dataset(
    dataset_path: str,
    num_clients: int,
    ignored_dirs: List[str] = [],
    train_test_split: bool = False,
) -> Dict[str, List[str]]:
    """
    Split the dataset into num_clients clients.

    Args:
    -----
        dataset_path (str): Path to the dataset.
        num_clients (int): Number of clients.
        ignored_dirs (List[str]): List of directories to be ignored.

    Returns:
    --------
        clients (dict): Dictionary of clients and their respective data.
    """
    clients = replicate_dirs(dataset_path, num_clients, ignored_dirs)

    paths = {
        join(dataset_path, _dir, _data)
        for client_data in clients.values()
        for _dir, data_list in client_data.items()
        for _data in data_list
    }

    clients = split_files(paths, clients, num_clients)

    return clients


if __name__ == "__main__":
    dataset_path = "datasets/splitted_dataset"
    num_clients = 8
    ignored_dirs = ["test", "masks"]
    clients = split_dataset(dataset_path, num_clients, ignored_dirs)
    save_clients(dataset_path, clients, f"datasets/fl_split/{num_clients}_clients/")
