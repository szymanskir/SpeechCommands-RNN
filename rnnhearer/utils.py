import pickle
from typing import Any


def write_pickle(obj: Any, filepath: str):
    with open(filepath, "wb") as save_file:
        pickle.dump(obj, save_file)
