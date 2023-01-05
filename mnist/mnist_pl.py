import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pytorch_lightning as pl
torch.manual_seed(0)

# from torch.nn import functional as F

parser = argparse.ArgumentParser(description='Train a MLP with MNIST in PyTorch Lightning.')
parser.add_argument('--batch_size', type=int, default=64, help='batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
parser.add_argument('--no_save', help='do not save the model', action='store_true')
parser.add_argument('--new_model', help='create a new model and do not load the pretrained weights', action='store_true')

# load the last saved model
MNIST_data_dir = 'MNIST_data/'
model_save_dir = 'saved-models/pl/'
all_subdirs = [os.path.join(model_save_dir+'lightning_logs/', d) for d in os.listdir(model_save_dir+'lightning_logs/') if os.path.isdir(os.path.join(model_save_dir+'lightning_logs/', d))]
if len(all_subdirs) > 0:
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    model_save_path = os.path.join(latest_subdir,'checkpoints/')
    for file in os.listdir(model_save_path):
        if file.endswith(".ckpt"):
            model_save_path = os.path.join(model_save_path, file)
        else:
            model_save_path = None
else:
    model_save_path = None


class mnistFCnet(pl.LightningModule):
    '''A fully connected neural network for MNIST classification.'''
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.criterion = nn.NLLLoss()

        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.log_softmax(self.fc3(x), dim=1)
        return x

    def prepare_data(self):
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        datasets.MNIST(MNIST_data_dir, download=True, train=True)
        datasets.MNIST(MNIST_data_dir, download=True, train=False)

    def train_dataloader(self):
        mnist_train = datasets.MNIST(MNIST_data_dir, download=False, train=True, transform=self.transform)
        trainloader = DataLoader(mnist_train, batch_size=self.batch_size, shuffle=True, num_workers=12)
        return trainloader

    def test_dataloader(self):
        mnist_test = datasets.MNIST(MNIST_data_dir, download=False, train=False, transform=self.transform)
        testloader = DataLoader(mnist_test, batch_size=self.batch_size)
        return testloader

    def configure_optimizers(self):
        return torch.optim.SGD(model.parameters(), lr=args.lr)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        output = self(images)
        loss = self.criterion(output, labels)
        self.log('train_loss', loss)
        return loss

if __name__ == '__main__':
    args = parser.parse_args()

    trainer = pl.Trainer(max_epochs=args.epochs, accelerator='gpu', devices=1, default_root_dir=model_save_dir, enable_checkpointing=not args.no_save)

    if args.new_model or model_save_path is None:
        model = mnistFCnet()
    else:
        model = mnistFCnet.load_from_checkpoint(model_save_path)

    trainer.fit(model)