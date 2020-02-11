import torch
import torchvision
import torchvision.transforms as transforms
import yaml

def config_args(config, heading):
    """Return hyperparameters in the configuration file.
    Keyword arguments:
    config -- the configuration file that contains the hyperparameters needed for the program.
    """
    with open('config.yaml', 'rb') as file:
        conf = yaml.safe_load(file)
    return conf[heading]

def load_dataset(batch_size):
    train_dataset = torchvision.datasets.MNIST(root='datasets/',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='datasets/',
                                              train=False,
                                              transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    return train_loader, test_loader