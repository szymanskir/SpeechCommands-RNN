import click
import logging
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from os import listdir
from os.path import join
from .data_reader import DataReader
from .data_manipulation import encode_categorical_labels
from .networks import NetworkConfiguration, create_network_from_config
from .utils import read_config, read_pickle, write_pickle
from .visualization import (
    plot_loss,
    plot_accuracy,
    plot_roc_curves,
    plot_confusion_matrix,
)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )


@click.group()
def main():
    setup_logging()


@main.command()
@click.argument("input_config", type=click.Path(exists=True))
@click.option("--train_data", type=click.Path(exists=True), required=True)
@click.option("--output", type=click.Path())
def train(input_config: str, train_data: str, output: str):
    network_config = NetworkConfiguration.from_config(read_config(input_config))
    main_labels = [
        "yes",
        "no",
        "up",
        "down",
        "left",
        "right",
        "on",
        "off",
        "stop",
        "go",
    ]
    num_classes = len(main_labels) + 1

    data_reader = DataReader(train_data)
    data = data_reader.read()

    audio_samples = [d["audio_data"] for d in data]
    audio_samples_padded = pad_sequences(audio_samples)

    labels = [d["label"] if d["label"] in main_labels else "unknown" for d in data]
    encoded_labels = encode_categorical_labels(
        labels=labels, kept_labels=main_labels + ["unknown"]
    )

    model = create_network_from_config(
        network_configuration=network_config,
        input_shape=audio_samples[0].shape,
        num_classes=num_classes
    )
    model.compile(
        optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    history = model.fit(
        x=audio_samples_padded,
        y=encoded_labels,
        validation_split=0.2,
        epochs=network_config.epochs_count,
        batch_size=network_config.batch_size
    )

    if output:
        write_pickle(history, output)


@main.command()
@click.argument("histories_dir", type=click.Path(exists=True))
@click.option("--loss", is_flag=True)
@click.option("--acc", is_flag=True)
@click.option("--roc_auc", is_flag=True)
@click.option("--confusion_matrix", is_flag=True)
def visualize(
    histories_dir: str, loss: bool, acc: bool, roc_auc: bool, confusion_matrix: bool
):
    history_files = listdir(histories_dir)
    histories = [
        read_pickle(join(histories_dir, history_file)) for history_file in history_files
    ]

    main_labels = [
        "yes",
        "no",
        "up",
        "down",
        "left",
        "right",
        "on",
        "off",
        "stop",
        "go",
        "unknown",
    ]

    if loss:
        plot_loss(histories)
    if acc:
        plot_accuracy(histories)
    if roc_auc:
        for history in histories:
            y_scores = history.model.predict(history.validation_data[0])
            y_test = history.validation_data[1]
            plot_roc_curves(y_score=y_scores, y_test=y_test, labels=main_labels)
    if confusion_matrix:
        for history in histories:
            y_scores = history.model.predict(history.validation_data[0])
            y_test = history.validation_data[1]
            plot_confusion_matrix(y_score=y_scores, y_test=y_test, labels=main_labels)

    plt.show()
