import numpy as np
from torchvision.datasets import MNIST, SVHN
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class UDADataLoader:
    def __init__(self):
        self.batch_size = 512
        self.mnist_train_dataset = MNIST(root="./MNIST", train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(), 
                transforms.Resize(32), # resize to [32,32]
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(lambda x: self.tile_image(x))
                ]))
        self.mnist_test_dataset = MNIST(root="./MNIST", train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(), 
                transforms.Resize(32), # resize to [32,32]
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(lambda x: self.tile_image(x))
                ]))
        
        self.svhn_transform_list = [transforms.ToTensor(), 
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.svhn_train_dataset = SVHN(root="./SVHN", split='train', download=True,
            transform=transforms.Compose(self.svhn_transform_list)) # [num, 3, 32, 32] range (0, 255)

        self.svhn_test_dataset = SVHN(root="./SVHN", split='test', download=True,
            transform=transforms.Compose(self.svhn_transform_list)) # [num, 3, 32, 32] range (0, 255)

        self.mnist_train_loader = DataLoader(self.mnist_train_dataset, batch_size=self.batch_size, shuffle=True)

        self.mnist_test_loader = DataLoader(self.mnist_test_dataset, batch_size=self.batch_size, shuffle=False)

        self.svhn_train_loader = DataLoader(self.svhn_train_dataset, batch_size=self.batch_size, shuffle=True)

        self.svhn_test_loader = DataLoader(self.svhn_test_dataset, batch_size=self.batch_size, shuffle=False)
        

    def tile_image(self, image):
        '''duplicate along channel axis'''
        return image.repeat(3,1,1)

