import pandas as pd
import seaborn as sns

sns.set()

from keras.callbacks import History
from typing import List

from .data_manipulation import history_to_df


def plot_accuracy(histories: List[History]):
    histories_df = pd.concat([history_to_df(history) for history in histories])
    histories_df = pd.melt(
        histories_df.reset_index(),
        id_vars=["index", "model_name"],
        value_vars=["acc", "val_acc"],
        var_name="acc_type",
        value_name="acc",
    )
    accuracy_plot = sns.FacetGrid(data=histories_df, col="model_name")
    accuracy_plot = accuracy_plot.map(
        sns.lineplot,
        x="index",
        y="acc",
        hue="acc_type",
        style="acc_type",
        data=histories_df,
    )

    accuracy_plot.add_legend()
    return accuracy_plot


def plot_loss(histories: List[History]):
    histories_df = pd.concat([history_to_df(history) for history in histories])
    histories_df = pd.melt(
        histories_df.reset_index(),
        id_vars=["index", "model_name"],
        value_vars=["loss", "val_loss"],
        var_name="loss_type",
        value_name="loss",
    )
    accuracy_plot = sns.FacetGrid(data=histories_df, col="model_name")
    accuracy_plot = accuracy_plot.map(
        sns.lineplot,
        x="index",
        y="loss",
        hue="loss_type",
        style="loss_type",
        data=histories_df,
    )

    accuracy_plot.add_legend()
    return accuracy_plot
