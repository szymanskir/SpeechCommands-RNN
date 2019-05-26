import pytest
import pandas as pd

from os.path import dirname, join
from keras.callbacks import History
from rnnhearer.data_manipulation import *
from rnnhearer.utils import read_pickle


@pytest.fixture
def sample_history() -> History:
    return read_pickle(join(dirname(__file__), "resources", "model-history.pkl"))


def test_history_to_df_conversion(sample_history):
    history_df: pd.DataFrame = history_to_df(sample_history)
    assert (history_df["model_name"] == sample_history.model.name).all()
    assert (history_df["val_loss"] == sample_history.history["val_loss"]).all() 
    assert (history_df["loss"] == sample_history.history["loss"]).all() 
    assert (history_df["val_acc"] == sample_history.history["val_acc"]).all() 
    assert (history_df["acc"] == sample_history.history["acc"]).all() 