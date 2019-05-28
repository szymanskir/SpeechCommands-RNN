import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import interp
from keras.callbacks import History
from sklearn.metrics import auc, confusion_matrix, roc_curve
from typing import Any, Dict, List, Union
from .data_manipulation import history_to_df

sns.set()


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
    loss_plot = sns.FacetGrid(data=histories_df, col="model_name")
    loss_plot = loss_plot.map(
        sns.lineplot,
        x="index",
        y="loss",
        hue="loss_type",
        style="loss_type",
        data=histories_df,
    )

    loss_plot.add_legend()
    return loss_plot


def plot_roc_curves(y_score: np.ndarray, y_test: np.ndarray, labels: List[str]):
    plt.rcParams.update({"font.size": 14})

    lw = 2
    n_classes = len(labels)

    fpr: Dict[Union[int, str], Any] = dict()
    tpr: Dict[Union[int, str], Any] = dict()
    roc_auc: Dict[Union[int, str], Any] = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(1)
    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.4f})" "".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    for i in range(n_classes):
        plt.plot(
            fpr[i],
            tpr[i],
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.4f})"
            "".format(labels[i], roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic curves")
    plt.legend(loc="lower right")
    plt.show()


def plot_confusion_matrix(y_score: np.ndarray, y_test: np.ndarray, labels: List[str]):
    confm = confusion_matrix(y_test.argmax(axis=1), y_score.argmax(axis=1))
    df_cm = pd.DataFrame(confm, index=labels, columns=labels)

    return sns.heatmap(df_cm, cmap="Oranges", annot=True)
