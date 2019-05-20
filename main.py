import logging
from rnnhearer.data_manipulation import DataReader
from rnnhearer.utils import write_pickle

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M:%S'
    )
data_reader = DataReader("data/train")
data = data_reader.read()
write_pickle(data, "data/speech_commands.pkl")
