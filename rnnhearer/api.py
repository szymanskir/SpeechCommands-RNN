import click
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from .data_reader import DataReader
from .data_manipulation import AudioDataGenerator
from .networks import NetworkConfiguration, create_network_from_config
from .utils import read_config, read_pickle, write_pickle
from .visualization import (
    plot_loss,
    plot_accuracy,
    plot_roc_curves,
    plot_confusion_matrix,
)
from os.path import join


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
@click.option("--data_dir", type=click.Path(exists=True), required=True)
@click.option("--output", type=click.Path())
def train(input_config: str, data_dir: str, output: str):
    train_inner(input_config=input_config, data_dir=data_dir, output=output)


def train_inner(input_config: str, data_dir: str, output: str):
    network_config = NetworkConfiguration.from_config(read_config(input_config))
    main_labels = {
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
    }
    num_classes = len(main_labels)

    data_reader = DataReader(data_dir, join(data_dir, "audio", "validation_list.txt"))
    train_data, validation_data = data_reader.read(main_labels)

    logging.info("Creating model...")

    generator = AudioDataGenerator(
        network_config.representation, kept_labels=list(main_labels)
    )

    model = create_network_from_config(
        network_configuration=network_config,
        input_shape=generator.get_data_shape(sample_filepath=train_data[0][0]),
        num_classes=num_classes,
    )
    model.compile(
        optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    history = model.fit_generator(
        generator=generator.flow(
            samples=train_data, batch_size=network_config.batch_size
        ),
        steps_per_epoch=len(train_data) / network_config.batch_size,
        epochs=network_config.epochs_count,
        validation_data=generator.flow_in_memory(
            samples=validation_data, batch_size=network_config.batch_size
        ),
        validation_steps=len(validation_data) / network_config.batch_size,
    )

    validation_dataset_generator = generator.flow_in_memory(
        samples=validation_data, batch_size=len(validation_data)
    )
    validation_dataset = next(validation_dataset_generator)
    history.validation_data = validation_dataset

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
    visualize_inner(
        histories_dir=histories_dir,
        loss=loss,
        acc=acc,
        roc_auc=roc_auc,
        confusion_matrix=confusion_matrix,
    )


def visualize_inner(
    histories_dir: str, loss: bool, acc: bool, roc_auc: bool, confusion_matrix: bool
):
    history_files = list(Path(histories_dir).glob("*.pkl"))
    histories = [read_pickle(history_file) for history_file in history_files]

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
        plt.show()
    if acc:
        plot_accuracy(histories)
        plt.show()
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

