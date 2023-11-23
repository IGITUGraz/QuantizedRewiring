from filelock import FileLock
from torchvision.datasets import MNIST

from utils.path import get_data_path


def prepare():
    with FileLock(get_data_path() / 'data.lock'):
        MNIST(root=get_data_path(), train=True, download=True)
        MNIST(root=get_data_path(), train=False, download=True)


def get_train(train_transform=None):
    data = MNIST(root=get_data_path(), train=True, transform=train_transform)
    return data


def get_test(test_transform=None):
    data = MNIST(root=get_data_path(), train=False, transform=test_transform)
    return data
