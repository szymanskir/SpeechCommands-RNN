import logging
import pickle
from rnnhearer.data_manipulation import DataReader

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M:%S'
    )
data_reader = DataReader("data/train")
data = data_reader.read()
pickle.dump(data, "data/speech_commands.pkl")