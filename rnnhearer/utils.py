import configparser
import os.path
import pickle
from pathlib import Path
from typing import Any, Union


def write_pickle(obj: Any, filepath: str):
    with open(filepath, "wb") as save_file:
        pickle.dump(obj, save_file)


def read_pickle(filepath: Union[Path, str]) -> Any:
    with open(filepath, "rb") as f:
        content = pickle.load(f)

    return content


def read_config(filepath: Union[Path, str]):
    """Reads the given config file.
    Args:
        filepath (str): path to the config file
    Returns (config.ConfigParser):
        ConfigParser object with configuration values.
    """
    assert os.path.isfile(filepath)
    config = configparser.ConfigParser()
    config.read(filepath)

    return config
