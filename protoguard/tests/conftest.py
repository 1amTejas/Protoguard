"""Shared fixtures. One simulated dataset is built once per test session."""
import pytest

from protoguard import GuardModel, build_windows, load_labels, load_messages
from protoguard.simulate import build_dataset


@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("sim")
    build_dataset(out, seed=3, train_duration=900.0, test_duration=1200.0)
    return out


@pytest.fixture(scope="session")
def train_windows(dataset):
    return build_windows(load_messages(dataset / "train_normal.pcap"), 5.0)


@pytest.fixture(scope="session")
def test_windows(dataset):
    return build_windows(load_messages(dataset / "test.pcap"), 5.0)


@pytest.fixture(scope="session")
def intervals(dataset):
    return load_labels(dataset / "labels.csv")


@pytest.fixture(scope="session")
def model(train_windows):
    return GuardModel().fit(train_windows)
