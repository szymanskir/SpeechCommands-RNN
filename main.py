import click
import logging
import numpy as np
from rnnhearer.data_reader import DataReader
from rnnhearer.data_manipulation import encode_categorical_labels
from rnnhearer.networks import create_sample_rnn
from keras.preprocessing.sequence import pad_sequences


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
@click.option("--train_data", type=click.Path(exists=True), required=True)
def train(train_data: str):
    main_labels = [
        "yes", "no", "up", "down", "left",
        "right", "on", "off", "stop", "go"
    ]
    num_classes = len(main_labels) + 1

    data_reader = DataReader(train_data)
    data = data_reader.read()

    audio_samples = [d["audio_data"] for d in data]
    audio_samples_padded = pad_sequences(audio_samples)

    labels = [d["label"] if d["label"] in main_labels else "unknown" for d in data]
    encoded_labels = encode_categorical_labels(
        labels=labels,
        kept_labels=main_labels + ["unknown"]
    )

    model = create_sample_rnn(
        input_shape=audio_samples[0].shape,
        num_classes=num_classes
    )
    model.compile(
        optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(
        x=audio_samples_padded,
        y=encoded_labels,
        validation_split=0.2,
        epochs=10,
        batch_size=256,
    )


if __name__ == "__main__":
    main()
