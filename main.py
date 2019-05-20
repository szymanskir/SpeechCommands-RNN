from rnnhearer.data_manipulation import DataReader

data_reader = DataReader("data/train")
data = data_reader.read()
print(data[0])