import os
from os.path import join, basename
import pytest
from rnnhearer.data_reader import DataReader, _SPEECH_COMMANDS_SAMPLES_DIRECTORIES


@pytest.fixture
def dir_without_audio_dir(tmpdir):
    return tmpdir.mkdir("data/")


@pytest.fixture
def dir_without_word_dirs(tmpdir):
    d = tmpdir.mkdir("data")
    os.mkdir(join(d, "audio"))
    return d


@pytest.fixture
def proper_directory(tmpdir):
    d = tmpdir.mkdir("data/")
    os.mkdir(join(d, "audio"))
    [
        os.mkdir(join(d, "audio", word_dir))
        for word_dir in _SPEECH_COMMANDS_SAMPLES_DIRECTORIES
    ]
    return d


@pytest.fixture
def wav_directory(tmpdir):
    p = tmpdir.join("test1.wav")
    p.write("")
    p = tmpdir.join("test2.wav")
    p.write("")
    p = tmpdir.join("test3.mp3")
    p.write("")
    return tmpdir


def test_DataReader_errors_on_dir_without_audio_subdir(dir_without_audio_dir):
    with pytest.raises(FileNotFoundError):
        DataReader(dir_without_audio_dir)


def test_DataReader_errors_on_dir_without_word_subdirs(dir_without_word_dirs):
    with pytest.raises(FileNotFoundError):
        DataReader(dir_without_audio_dir)


def test_DataReader_proper_dir(proper_directory):
    DataReader(proper_directory)


def test_DataReader_find_wav_files(wav_directory):
    result = DataReader._find_all_wav_files(wav_directory)
    assert set([basename(r) for r in result]) == {"test1.wav", "test2.wav"}
