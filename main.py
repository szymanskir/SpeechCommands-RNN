import logging
import numpy as np
from rnnhearer.data_reader import DataReader
from rnnhearer.utils import write_pickle
from scipy import sparse
from keras import models, layers
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    datefmt="%m-%d %H:%M:%S",
)

main_labels = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
num_classes = len(main_labels) + 1

data_reader = DataReader("data/train")
data = data_reader.read()
audio_samples = [d["audio_data"] for d in data]
labels = np.array(
    [d["label"] if d["label"] in main_labels else "unknown" for d in data]
)

label_encoder = LabelEncoder()
encoded_labels = to_categorical(
    y=label_encoder.fit_transform(labels), num_classes=num_classes
)
audio_samples_padded = pad_sequences(audio_samples)

model = models.Sequential()
model.add(layers.CuDNNLSTM(32, input_shape=(audio_samples_padded[0]).shape))
model.add(layers.Dense(num_classes, activation="softmax"))
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
