import pytest
from os.path import dirname, join
from rnnhearer.networks import AudioRepresentation, NetworkConfiguration
from rnnhearer.utils import read_config


@pytest.fixture
def sample_config():
    config = read_config(join(dirname(__file__), "resources", "sample_config.ini"))
    return config


def test_NetworkConfiguration_from_file_init(sample_config):
    result: NetworkConfiguration = NetworkConfiguration.from_config(sample_config)
    assert result.units_per_layer == [32, 32]
    assert result.representation == AudioRepresentation.RAW
    assert result.dropout_probabilities == [0, 0.2]
    assert result.epochs_count == 10
    assert result.batch_size == 256
