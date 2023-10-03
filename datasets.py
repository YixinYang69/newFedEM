import os
import pickle
import string

import torch
from torchvision.datasets import CIFAR10, CIFAR100, EMNIST, MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import Dataset

import numpy as np
from PIL import Image
import copy as cp


class SyntheticDataset(Dataset):
    """
    Constructs a torch.utils.Dataset object from a given dataset;
    expects dataset to be a torch.utils.Dataset object;
    expects dataset to store tuples of the form (x, y) where x is vector and y is a scalar

    Attributes
    ----------
    dataset: torch.utils.Dataset object

    Methods
    -------
    __init__
    __len__
    __getitem__
    """

    def __init__(self, dataset, reserve_size):
        """
        :param dataset: torch.utils.Dataset object
        :param reserve_size: in multiple of the original dataset size
        """

        self.dataset = dataset

        reserve_size = int(reserve_size)
        self.data = torch.cat([self.dataset.data for _ in range(reserve_size)], dim=0)
        self.targets = torch.cat([self.dataset.targets for _ in range(reserve_size)], dim=0)
        self.init_size = len(self.data)

        self.orig_data = self.dataset.data
        self.orig_targets = self.dataset.targets

        self.synthetic_data = None
        self.synthetic_targets = None

        self.unharden_data = None
        self.unharden_targets = None

        self.transform = \
            Compose([
                ToTensor(),
                Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)
                )
            ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        all_data = self.data
        all_targets = self.targets

        img, target = all_data[index], int(all_targets[index])

        img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)
        
        return img, target, index

    def gen_synthetic_data(self, poriton):

        synthetic_data = torch.randint(
            self.orig_data.min(), 
            int(self.orig_data.max())+1, 
            size = (self.orig_data.size(0), self.orig_data.size(1), self.orig_data.size(2), self.orig_data.size(3)), 
            dtype = torch.uint8,
        )
        synthetic_targets = torch.randint(
            self.orig_targets.min(), 
            int(self.orig_targets.max())+1, 
            size = (self.orig_data.size(0),), 
            dtype=torch.int64,
        )

        self.synthetic_data = synthetic_data
        self.synthetic_targets = synthetic_targets

    def set_synthetic_targets(self, synthetic_targets):
        self.synthetic_targets = synthetic_targets

    def set_unharden(self, unharden_data, unharden_targets):
        self.unharden_data = unharden_data
        self.unharden_targets = unharden_targets
    
    def set_data(self, type):
        if type == 'orig':
            self.data = self.orig_data
            self.targets = self.orig_targets
        elif type == 'synthetic':
            self.data = self.synthetic_data
            self.targets = self.synthetic_targets
        elif type == 'orig+synthetic':
            self.data = torch.cat([self.orig_data, self.synthetic_data], dim=0)
            self.targets = torch.cat([self.orig_targets, self.synthetic_targets], dim=0)
        else:
            raise Exception('type must be synthetic or orig')
        
    def set_portions(self, orig_portion, synthetic_portion, unharden_portion):

        self.orig_portion = orig_portion
        self.synthetic_portion = synthetic_portion
        self.unharden_portion = unharden_portion

        self.orig_size = int(len(self.orig_data) * self.orig_portion)
        self.orig_sample = np.random.choice(len(self.orig_data), self.orig_size, replace=False)
        self.data = self.orig_data[self.orig_sample]
        self.targets = self.orig_targets[self.orig_sample]

        if self.synthetic_targets is None:
            self.synthetic_sample = np.array([])
        else:
            self.synthetic_size = int(len(self.synthetic_data) * self.synthetic_portion)
            self.synthetic_sample = np.random.choice(len(self.synthetic_data), self.synthetic_size, replace=False)
            self.data = torch.cat([self.data, self.synthetic_data[self.synthetic_sample]], dim=0)
            self.targets = torch.cat([self.targets, self.synthetic_targets[self.synthetic_sample]], dim=0)

        if self.unharden_targets is None:
            self.unharden_sample = np.array([])
        else:
            self.unharden_size = int(len(self.unharden_data) * self.unharden_portion)
            self.unharden_sample = np.random.choice(len(self.unharden_data), self.unharden_size, replace=False)
            self.data = torch.cat([self.data, self.unharden_data[self.unharden_sample]], dim=0)
            self.targets = torch.cat([self.targets, self.unharden_targets[self.unharden_sample]], dim=0)

        assert len(self.data) == len(self.targets), "data and targets must have the same length"
        assert len(self.data) <= self.init_size, "data must be smaller than the total size when the dataset object is initialized"

class TabularDataset(Dataset):
    """
    Constructs a torch.utils.Dataset object from a pickle file;
    expects pickle file stores tuples of the form (x, y) where x is vector and y is a scalar

    Attributes
    ----------
    data: iterable of tuples (x, y)

    Methods
    -------
    __init__
    __len__
    __getitem__
    """

    def __init__(self, path):
        """
        :param path: path to .pkl file
        """
        with open(path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.int64), idx


class SubFEMNIST(Dataset):
    """
    Constructs a subset of FEMNIST dataset corresponding to one client;
    Initialized with the path to a `.pt` file;
    `.pt` file is expected to hold a tuple of tensors (data, targets) storing the images and there corresponding labels.

    Attributes
    ----------
    transform
    data: iterable of integers
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """
    def __init__(self, path):
        self.transform = Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,))
        ])

        self.data, self.targets = torch.load(path)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        try:
            img = np.uint8(img.numpy() * 255)
        except:
            img = np.uint8(img.detach().numpy()*255)
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index


class SubEMNIST(Dataset):
    """
    Constructs a subset of EMNIST dataset from a pickle file;
    expects pickle file to store list of indices

    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """

    def __init__(self, path, emnist_data=None, emnist_targets=None, transform=None):
        """
        :param path: path to .pkl file; expected to store list of indices
        :param emnist_data: EMNIST dataset inputs
        :param emnist_targets: EMNIST dataset labels
        :param transform:
        """
        with open(path, "rb") as f:
            self.indices = pickle.load(f)

        if transform is None:
            self.transform =\
                Compose([
                    ToTensor(),
                    Normalize((0.1307,), (0.3081,))
                ])

        if emnist_data is None or emnist_targets is None:
            self.data, self.targets = get_emnist()
        else:
            self.data, self.targets = emnist_data, emnist_targets

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index


class SubCIFAR10(Dataset):
    """
    Constructs a subset of CIFAR10 dataset from a pickle file;
    expects pickle file to store list of indices

    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """
    def __init__(self, path, cifar10_data=None, cifar10_targets=None, transform=None):
        """
        :param path: path to .pkl file; expected to store list of indices
        :param cifar10_data: Cifar-10 dataset inputs stored as torch.tensor
        :param cifar10_targets: Cifar-10 dataset labels stored as torch.tensor
        :param transform:
        """
        with open(path, "rb") as f:
            self.indices = pickle.load(f)

        if transform is None:
            self.transform = \
                Compose([
                    ToTensor(),
                    Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)
                    )
                ])

        if cifar10_data is None or cifar10_targets is None:
            self.data, self.targets = get_cifar10()
        else:
            self.data, self.targets = cifar10_data, cifar10_targets

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        target = target

        return img, target, index


class SubCIFAR100(Dataset):
    """
    Constructs a subset of CIFAR100 dataset from a pickle file;
    expects pickle file to store list of indices

    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """
    def __init__(self, path, cifar100_data=None, cifar100_targets=None, transform=None):
        """
        :param path: path to .pkl file; expected to store list of indices:
        :param cifar100_data: CIFAR-100 dataset inputs
        :param cifar100_targets: CIFAR-100 dataset labels
        :param transform:
        """
        with open(path, "rb") as f:
            self.indices = pickle.load(f)

        if transform is None:
            self.transform = \
                Compose([
                    ToTensor(),
                    Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)
                    )
                ])

        if cifar100_data is None or cifar100_targets is None:
            self.data, self.targets = get_cifar100()

        else:
            self.data, self.targets = cifar100_data, cifar100_targets

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        target = target

        return img, target, index

    
class SubMNIST(Dataset):
    """
    Constructs a subset of EMNIST dataset from a pickle file;
    expects pickle file to store list of indices

    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """

    def __init__(self, path, mnist_data=None, mnist_targets=None, transform=None):
        """
        :param path: path to .pkl file; expected to store list of indices
        :param emnist_data: EMNIST dataset inputs
        :param emnist_targets: EMNIST dataset labels
        :param transform:
        """
        with open(path, "rb") as f:
            self.indices = pickle.load(f)

        if transform is None:
            self.transform =\
                Compose([
                    ToTensor(),
                    Normalize((0.1307,), (0.3081,))
                ])

        if mnist_data is None or mnist_targets is None:
            self.data, self.targets = get_mnist()
        else:
            self.data, self.targets = mnist_data, mnist_targets

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index
    

class CharacterDataset(Dataset):
    def __init__(self, file_path, chunk_len):
        """
        Dataset for next character prediction, each sample represents an input sequence of characters
         and a target sequence of characters representing to next sequence of the input
        :param file_path: path to .txt file containing the training corpus
        :param chunk_len: (int) the length of the input and target sequences
        """
        self.all_characters = string.printable
        self.vocab_size = len(self.all_characters)
        self.n_characters = len(self.all_characters)
        self.chunk_len = chunk_len

        with open(file_path, 'r') as f:
            self.text = f.read()

        self.tokenized_text = torch.zeros(len(self.text), dtype=torch.long)

        self.inputs = torch.zeros(self.__len__(), self.chunk_len, dtype=torch.long)
        self.targets = torch.zeros(self.__len__(), self.chunk_len, dtype=torch.long)

        self.__build_mapping()
        self.__tokenize()
        self.__preprocess_data()

    def __tokenize(self):
        for ii, char in enumerate(self.text):
            self.tokenized_text[ii] = self.char2idx[char]

    def __build_mapping(self):
        self.char2idx = dict()
        for ii, char in enumerate(self.all_characters):
            self.char2idx[char] = ii

    def __preprocess_data(self):
        for idx in range(self.__len__()):
            self.inputs[idx] = self.tokenized_text[idx:idx+self.chunk_len]
            self.targets[idx] = self.tokenized_text[idx+1:idx+self.chunk_len+1]

    def __len__(self):
        return max(0, len(self.text) - self.chunk_len)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], idx


def get_emnist():
    """
    gets full (both train and test) EMNIST dataset inputs and labels;
    the dataset should be first downloaded (see data/emnist/README.md)
    :return:
        emnist_data, emnist_targets
    """
    emnist_path = os.path.join("data", "emnist", "raw_data")
    assert os.path.isdir(emnist_path), "Download EMNIST dataset!!"

    emnist_train =\
        EMNIST(
            root=emnist_path,
            split="byclass",
            download=True,
            train=True
        )

    emnist_test =\
        EMNIST(
            root=emnist_path,
            split="byclass",
            download=True,
            train=True
        )

    emnist_data =\
        torch.cat([
            emnist_train.data,
            emnist_test.data
        ])

    emnist_targets =\
        torch.cat([
            emnist_train.targets,
            emnist_test.targets
        ])

    return emnist_data, emnist_targets


def get_cifar10():
    """
    gets full (both train and test) CIFAR10 dataset inputs and labels;
    the dataset should be first downloaded (see data/emnist/README.md)
    :return:
        cifar10_data, cifar10_targets
    """
    cifar10_path = os.path.join("data", "cifar10", "raw_data")
    assert os.path.isdir(cifar10_path), "Download cifar10 dataset!!"

    cifar10_train =\
        CIFAR10(
            root=cifar10_path,
            train=True, download=False
        )

    cifar10_test =\
        CIFAR10(
            root=cifar10_path,
            train=False,
            download=False)

    cifar10_data = \
        torch.cat([
            torch.tensor(cifar10_train.data),
            torch.tensor(cifar10_test.data)
        ])

    cifar10_targets = \
        torch.cat([
            torch.tensor(cifar10_train.targets),
            torch.tensor(cifar10_test.targets)
        ])

    return cifar10_data, cifar10_targets


def get_cifar100():
    """
    gets full (both train and test) CIFAR100 dataset inputs and labels;
    the dataset should be first downloaded (see data/cifar100/README.md)
    :return:
        cifar100_data, cifar100_targets
    """
    cifar100_path = os.path.join("data", "cifar100", "raw_data")
    assert os.path.isdir(cifar100_path), "Download cifar10 dataset!!"

    cifar100_train =\
        CIFAR100(
            root=cifar100_path,
            train=True, download=False
        )

    cifar100_test =\
        CIFAR100(
            root=cifar100_path,
            train=False,
            download=False)

    cifar100_data = \
        torch.cat([
            torch.tensor(cifar100_train.data),
            torch.tensor(cifar100_test.data)
        ])

    cifar100_targets = \
        torch.cat([
            torch.tensor(cifar100_train.targets),
            torch.tensor(cifar100_test.targets)
        ])

    return cifar100_data, cifar100_targets

def get_mnist():
    mnist_path = os.path.join("data", "mnist", "raw_data")
    assert os.path.isdir(mnist_path), "Download mnist dataset!!"
    
    mnist_train =\
        MNIST(
            root= mnist_path,
            train=True, download=False
        )

    mnist_test =\
        MNIST(
            root=mnist_path,
            train=False,
            download=False)

    mnist_data = \
        torch.cat([
            torch.tensor(mnist_train.data),
            torch.tensor(mnist_test.data)
        ])

    mnist_targets = \
        torch.cat([
            torch.tensor(mnist_train.targets),
            torch.tensor(mnist_test.targets)
        ])

    return mnist_data, mnist_targets