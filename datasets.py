"""
Partly adapted from:
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/sampling/permutations.html
https://github.com/jungtaekkim/error-free-differentiable-swap-functions/blob/main/src/datasets/dataset.py
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

import utils
from tsp_dataset import TSPGraphDataset


class NoisyMNIST(Dataset):
    def __init__(self, data_file):
        """
        data_file: Path to the .pt file containing the pre-processed data.
        """
        # Load the dataset from the .pt file
        self.data = torch.load(data_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        random_patches, gt_perm_list = self.data[idx]
        return random_patches, gt_perm_list


def chunk_image(image: torch.Tensor, num_pieces: int):
    """Randomly chunk a single image.
    Args:
        image: Image [channels, height, width].

    Returns:
        pieces: Image chunks in their original positions. [num_pieces, channels,
                height // num_pieces, width // num_pieces]
        random_pieces: Image chunks in their randomly permuted positions.
        permute_index: List of permuted indices.
    """
    # Get image dimensions.
    height, width = image.shape[-2:]

    # Get piece dimensions.
    piece_height = height // num_pieces
    piece_width = width // num_pieces
    pieces = []

    # Obtain indices for each of the image chunks.
    for p_h in range(num_pieces):
        for p_w in range(num_pieces):
            left = p_w * piece_width
            right = left + piece_width
            top = p_h * piece_height
            bottom = top + piece_height
            piece = image[:, top:bottom, left:right]
            pieces.append(piece)

    pieces = torch.stack(pieces, 0)

    # Randomly permute the index of the pieces.
    permute_index = torch.randperm(num_pieces**2)
    random_pieces = pieces[permute_index]

    return pieces, random_pieces, permute_index


def batch_chunk_image(images: torch.Tensor, num_pieces: int):
    """Randomly chunk a batch of images.
    Args:
        image: Images [batch, channels, height, width].

    Returns:
        pieces: Batch of image chunks in their original positions. [batch,
                num_pieces, channels, height // num_pieces, width // num_pieces]
        random_pieces: Batch of image chunks in their randomly permuted positions.
                       [batch, num_pieces, channels, height // num_pieces, width // num_pieces]
        permute_index: Batch of permutation lists. [batch, num_pieces**2]
    """
    batch_pieces, batch_random_pieces, batch_permute_index = [], [], []
    for image in images:
        pieces, random_pieces, permute_index = chunk_image(image, num_pieces)

        batch_pieces.append(pieces)
        batch_random_pieces.append(random_pieces)
        batch_permute_index.append(permute_index)
    return (
        torch.stack(batch_pieces, 0),
        torch.stack(batch_random_pieces, 0),
        torch.stack(batch_permute_index, 0),
    )


class MultiDigitDataset(Dataset):
    def __init__(
        self,
        images,
        labels,
        num_digits,
        num_compare,
        seed=0,
        determinism=True,
    ):
        super(MultiDigitDataset, self).__init__()
        self.images = images
        self.labels = labels
        self.num_digits = num_digits
        self.num_compare = num_compare
        self.seed = seed
        self.rand_state = None

        self.determinism = determinism

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        labels = []
        images = []
        labels_ = None
        for digit_idx in range(self.num_digits):
            id = torch.randint(len(self), (self.num_compare,))
            labels.append(self.labels[id])
            images.append(self.images[id].type(torch.float32) / 255.0)
            if labels_ is None:
                labels_ = torch.zeros_like(labels[0] * 1.0)
            labels_ = labels_ + 10.0 ** (self.num_digits - 1 - digit_idx) * self.labels[id]

        images = torch.cat(images, dim=-1)
        sort_order = torch.argsort(labels_)

        # Uniform sampling
        sorted_images = utils.permute_image_perm_list(sort_order, images)

        perm = torch.randperm(self.num_compare)
        images = utils.permute_image_perm_list(perm, sorted_images)
        sort_order = torch.argsort(perm)

        return images, sort_order


class MultiDigitSplits(object):
    def __init__(
        self, dataset, num_digits=4, num_compare=None, seed=0, deterministic_data_loader=True
    ):

        self.deterministic_data_loader = deterministic_data_loader

        if dataset == "MNIST":
            trva_real = datasets.MNIST(root="./data", download=True)
            xtr_real = trva_real.data.view(-1, 1, 28, 28)
            ytr_real = trva_real.targets

            te_real = datasets.MNIST(root="./data", train=False, download=True)
            xte_real = te_real.data.view(-1, 1, 28, 28)
            yte_real = te_real.targets

            self.train_dataset = MultiDigitDataset(
                images=xtr_real,
                labels=ytr_real,
                num_digits=num_digits,
                num_compare=num_compare,
                seed=seed,
                determinism=deterministic_data_loader,
            )
            self.test_dataset = MultiDigitDataset(
                images=xte_real,
                labels=yte_real,
                num_digits=num_digits,
                num_compare=num_compare,
                seed=seed,
            )

        else:
            raise NotImplementedError()

    def get_train_loader(self, batch_size, **kwargs):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            drop_last=True,
            num_workers=4 if not self.deterministic_data_loader else 0,
            **kwargs,
        )
        return train_loader

    def get_test_loader(self, batch_size, **kwargs):
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, **kwargs)
        return test_loader


def get_train_loader(config, relative="./"):
    if config.dataset == "unscramble-noisy-MNIST":
        trainset = NoisyMNIST(data_file=f"{relative}/data/noisy_MNIST/train_noisy_mnist.pt")

    elif config.dataset == "unscramble-CIFAR10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

    elif config.dataset == "sort-MNIST":
        data = MultiDigitSplits(
            "MNIST", num_digits=config.num_digits, num_compare=config.num_pieces
        )
        trainset = data.train_dataset

    elif config.dataset == "tsp":
        assert config.num_pieces in [20, 50, 100]
        trainset = TSPGraphDataset(f"./data/tsp/tsp{config.num_pieces}_train_concorde.txt")

    else:
        raise NotImplementedError

    train_sampler = DistributedSampler(trainset)
    g = utils.get_ddp_generator(config.seed)
    train_loader = DataLoader(
        dataset=trainset,
        batch_size=int(config.train.batch_size),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
        generator=g,
    )

    return train_loader


def get_test_loader(config, shuffle=True, relative="./"):
    if config.dataset == "unscramble-noisy-MNIST":
        testset = NoisyMNIST(
            data_file=f"{relative}/data/noisy_MNIST/test_noisy_mnist_num_pieces={config.num_pieces}.pt"
        )

    elif config.dataset == "unscramble-CIFAR10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    elif config.dataset == "sort-MNIST":
        data = MultiDigitSplits(
            "MNIST", num_digits=config.num_digits, num_compare=config.num_pieces
        )
        testset = data.test_dataset

    elif config.dataset == "tsp":
        assert config.num_pieces in [20, 50, 100]
        testset = TSPGraphDataset(f"./data/tsp/tsp{config.num_pieces}_test_concorde.txt")

    else:
        raise NotImplementedError

    test_loader = DataLoader(testset, config.eval_batch_size, drop_last=False, shuffle=shuffle)

    return test_loader
