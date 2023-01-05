import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import datasets, transforms
torch.manual_seed(0)

parser = argparse.ArgumentParser(description='Train a MLP with MNIST in PyTorch.')
parser.add_argument('--batch_size', type=int, default=64, help='batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
parser.add_argument('--silent', help='hide epoch progress', action='store_true')
parser.add_argument('--no_save', help='do not save the model', action='store_true')
parser.add_argument('--new_model', help='create a new model and do not load the pretrained weights', action='store_true')


def get_data_loader(batch_size=64):
    '''Downloads MNIST dataset and returns a data loader for training.'''
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    return trainloader

class mnistFCnet(nn.Module):
    '''A fully connected neural network for MNIST classification.'''
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.log_softmax(self.fc3(x), dim=1)
        return x

def train_model(model, trainloader, criterion, optimizer, epochs=20, verbose=False):
    '''Train the model.'''
    for e in range(epochs):
        running_loss = 0
        pbar = tqdm(trainloader) if verbose else trainloader
        for images, labels in pbar:
            # transfer to gpu
            images, labels = images.to(device), labels.to(device)
            
            # forward pass
            output = model(images)
            loss = criterion(output, labels)
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        else:
            print(f'Epoch {e+1} - Training loss: {running_loss/len(trainloader)}')

def test_model(model, testloader, verbose=False):
    '''Test the model on a single image.'''
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    img = images[0].view(1, 784)

    # turn off gradients to speed up this part
    with torch.no_grad():
        logps = model(img)

    # output of the network are log-probabilities, need to take exponential for probabilities
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    print("Predicted Digit =", probab.index(max(probab)))

    # plot the image
    if verbose:
        plt.imshow(images[0].numpy().squeeze(), cmap='Greys_r')
        plt.show()

if __name__ == '__main__':
    args = parser.parse_args()

    # checking if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using {device} for training.')

    trainloader = get_data_loader(args.batch_size)

    if args.new_model or not os.path.exists('saved-models/mnist_fc.pth'):
        model = mnistFCnet()
    else:
        model = mnistFCnet()
        model.load_state_dict(torch.load('saved-models/mnist_fc.pth'))

    # define the loss function
    criterion = nn.NLLLoss()

    # define the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # transfer to gpu
    model.to(device)

    train_model(model, trainloader, criterion, optimizer, args.epochs, not args.silent)

    if not args.no_save:
        torch.save(model.state_dict(), 'saved-models/mnist_fc.pth')

    # test the model
    testloader = get_data_loader(1)
    test_model(model.to('cpu'), testloader, not args.silent)