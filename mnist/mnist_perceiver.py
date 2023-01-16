import os
import argparse
import numpy as np
from einops import rearrange
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
torch.manual_seed(0)

from perceiver.model.core import InputAdapter, PerceiverEncoder, PerceiverDecoder, TrainableQueryProvider, OutputAdapter, FourierPositionEncoding

MNIST_data_dir = 'MNIST_data/'
model_save_dir = 'saved-models/mnist_perceiver.pth'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--workers', type=int, default=0, help='number of workers for dataloading (default: 0)')
    parser.add_argument('--new_model', help='create a new model and do not load the pretrained weights', action='store_true')
    parser.add_argument('--silent', help='hide epoch progress', action='store_true')
    args = parser.parse_args()
    return args

def get_data_loader(batch_size=64, num_workers=0):
    '''Downloads MNIST dataset and returns a data loader for training.'''
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST(MNIST_data_dir, download=True, train=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return trainloader

class ImageInputAdapter(InputAdapter):
    '''An input adapter for images.'''
    def __init__(self, image_shape, num_frequency_bands):
        *spatial_shape, num_image_channels = image_shape
        position_encoding = FourierPositionEncoding(input_shape=spatial_shape, num_frequency_bands=num_frequency_bands)

        super().__init__(num_input_channels=num_image_channels + position_encoding.num_position_encoding_channels())

        self.image_shape = image_shape
        self.position_encoding = position_encoding

    def forward(self, x):
        b, *d = x.shape

        if tuple(d) != self.image_shape:
            raise ValueError(f"Input vision shape {tuple(d)} different from required shape {self.image_shape}")

        x_enc = self.position_encoding(b)
        x = rearrange(x, "b ... c -> b (...) c")
        return torch.cat([x, x_enc], dim=-1)

class ClassificationOutputAdapter(OutputAdapter):
    '''An output adapter for classification.'''
    def __init__(self, num_classes, num_output_query_channels):
        super().__init__()
        self.linear = nn.Linear(num_output_query_channels, num_classes)

    def forward(self, x):
        return self.linear(x).squeeze(dim=1)

class mnistPerceiver(nn.Module):
    '''A Perceiver for MNIST classification.'''
    def __init__(self):
        super().__init__()
        self.image_size = (28, 28, 1)

        self.input_adapter = ImageInputAdapter(
            image_shape = self.image_size,
            num_frequency_bands = 14
        )
        self.query_provider = TrainableQueryProvider(
            num_queries = 1,
            num_query_channels = 10
        )
        self.output_adapter = ClassificationOutputAdapter(
            num_classes = 10,
            num_output_query_channels = 10
        )

        self.encoder = PerceiverEncoder(
            input_adapter = self.input_adapter,
            num_latents = 16,
            num_latent_channels = 4,
            num_cross_attention_layers = 2,
            num_self_attention_layers_per_block = 2,
            num_self_attention_blocks = 3
        )
        self.decoder = PerceiverDecoder(
            output_adapter = self.output_adapter,
            output_query_provider = self.query_provider,
            num_latent_channels = 4,
            num_cross_attention_heads=2
        )
        
    def forward(self, x):
        x = x.view(x.shape[0], *self.image_size)
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.log_softmax(x, dim=1)
        return x

def train_model(model, trainloader, criterion, optimizer, epochs=20, verbose=False, device='cpu'):
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
    

if __name__ == '__main__':
    args = get_args()

    # get data loader
    trainloader = get_data_loader(batch_size=args.batch_size, num_workers=args.workers)

    # initialize model
    model = mnistPerceiver()
    if args.new_model or not os.path.exists(model_save_dir):
        print('Creating a new model.')
    else:
        print('Loading saved model.')
        model.load_state_dict(torch.load(model_save_dir))

    # setup up loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using {device} for training.')
    model.to(device)

    # train model
    train_model(model, trainloader, criterion, optimizer, epochs=args.epochs, verbose=True, device=device)

    # save model
    torch.save(model.state_dict(), model_save_dir)