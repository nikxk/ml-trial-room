from typing import Tuple, List
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch_optimizer as optim

from einops import rearrange, reduce

import optuna
import logging
import sys

from perceiver.model.core import InputAdapter, OutputAdapter, PerceiverEncoder, PerceiverDecoder, FourierPositionEncoding, TrainableQueryProvider

from utils import *

dataset_loc = 'MNIST_data/'

def main():
    train_loader, val_loader, test_loader = get_mnist_dataset(dataset_loc=dataset_loc, batch_size=64, num_workers=4)

    example: torch.Tensor = next(iter(train_loader))
    num_classes: int = len(train_loader.dataset.dataset.classes)

    loss_fn = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # initialize model
    model = mnistPerceiver(
        image_shape = example[0].shape[1:],
        num_classes = num_classes,
        num_frequency_bands = 14,
        latent_shape = (8, 8), 
        output_query_shape = (1, 8),
        num_CA_blocks=1,
        num_SA_blocks=1,
        num_SA_layers=6,
        )

    # model.load_state_dict(torch.load('mnist_perceiver.pt'))

    # train model
    optimizer = optim.Lamb(model.parameters(), lr=1e-3)
    model = train_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        num_epochs=100,
        patience=3,
        verbose=True
        )

    # test model
    test_loss, test_acc = evaluate_classifier(model=model, loader=test_loader, loss_fn=loss_fn, device=device)
    print(f'--\nTest loss: {test_loss}, Test accuracy: {test_acc}')

    # save model
    torch.save(model.state_dict(), 'mnist_perceiver.pt')


def objective(trial: optuna.trial.Trial):
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    num_frequency_bands = trial.suggest_int('num_frequency_bands', 2, 14)
    latent_size_pow = trial.suggest_int('latent_size_pow', 2, 4)
    num_CA_blocks = trial.suggest_int('num_CA_blocks', 1, 3)
    num_SA_layers = trial.suggest_int('num_SA_layers', 2, 6)

    latent_size = 2**latent_size_pow

    print(f'lr: {lr}, num_frequency_bands: {num_frequency_bands}, latent_size: {latent_size}, num_CA_blocks: {num_CA_blocks}, num_SA_layers: {num_SA_layers}')

    train_loader, val_loader, test_loader = get_mnist_dataset(dataset_loc=dataset_loc, batch_size=64, num_workers=4)

    example: torch.Tensor = next(iter(train_loader))
    num_classes: int = len(train_loader.dataset.dataset.classes)

    loss_fn = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # initialize model
    model = mnistPerceiver(
        image_shape = example[0].shape[1:],
        num_classes = num_classes,
        num_frequency_bands = num_frequency_bands,
        latent_shape = (latent_size, latent_size), 
        output_query_shape = (1, 8),
        num_CA_blocks=num_CA_blocks,
        num_SA_blocks=num_CA_blocks,
        num_SA_layers=num_SA_layers,
        )

    # train model
    optimizer = optim.Lamb(model.parameters(), lr=lr)
    model = train_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        num_epochs=20,
        patience=3,
        verbose=True
        )

    # test model
    test_loss, test_acc = evaluate_classifier(model=model, loader=test_loader, loss_fn=loss_fn, device=device)
    print(f'--\nTest loss: {test_loss}, Test accuracy: {test_acc}')

    # save the parameters to a csv file, append to the file if it already exists
    with open('mnist_perceiver.csv', 'a') as f:
        f.write(f'{lr},{num_frequency_bands},{latent_size},{num_CA_blocks},{num_SA_layers},{test_loss},{test_acc}\n')

    return test_acc


class ImageInputAdapter(InputAdapter):
    '''An input adapter for images.'''
    def __init__(self, image_shape, num_frequency_bands):
        num_image_channels, *spatial_shape = image_shape
        position_encoding = FourierPositionEncoding(input_shape=spatial_shape, num_frequency_bands=num_frequency_bands)

        super().__init__(num_input_channels=num_image_channels + position_encoding.num_position_encoding_channels())

        self.image_shape = image_shape
        self.position_encoding = position_encoding

    def forward(self, x):
        b, *d = x.shape

        if tuple(d) != self.image_shape:
            raise ValueError(f"Input vision shape {tuple(d)} different from required shape {self.image_shape}")

        x_enc = self.position_encoding(b)
        x = rearrange(x, "b c ... -> b (...) c")
        return torch.cat([x, x_enc], dim=-1)

class ClassificationOutputAdapter(OutputAdapter):
    '''An output adapter for classification.'''
    def __init__(self, num_classes, num_output_query_channels):
        super().__init__()
        self.linear = nn.Linear(num_output_query_channels, num_classes)

    def forward(self, x):
        return reduce(self.linear(x), 'b q c -> b c', 'mean')

class mnistPerceiver(nn.Module):
    '''A Perceiver for MNIST classification.'''
    def __init__(self, 
        image_shape: Tuple[int, ...], 
        num_classes: int, 
        num_frequency_bands=14, 
        latent_shape: Tuple[int, int]=(8, 8), 
        output_query_shape: Tuple[int, int]=(1, 8),
        num_CA_blocks: int=1,
        num_SA_blocks: int=1,
        num_SA_layers: int=6,
    ):
        super().__init__()
        self.image_shape = image_shape

        self.input_adapter = ImageInputAdapter(
            image_shape = self.image_shape,
            num_frequency_bands = num_frequency_bands
        )
        self.query_provider = TrainableQueryProvider(
            num_queries = output_query_shape[0],
            num_query_channels = output_query_shape[1]
        )
        self.output_adapter = ClassificationOutputAdapter(
            num_classes = num_classes,
            num_output_query_channels = output_query_shape[1],
        )

        self.encoder = PerceiverEncoder(
            input_adapter = self.input_adapter,
            num_latents = latent_shape[0],
            num_latent_channels = latent_shape[1],
            num_cross_attention_heads = 4,
            num_cross_attention_layers = num_CA_blocks,
            first_cross_attention_layer_shared = False,
            num_self_attention_heads = 4,
            num_self_attention_layers_per_block = num_SA_layers,
            num_self_attention_blocks = num_SA_blocks,
            first_self_attention_block_shared = True,
            dropout = 0.0,
            init_scale = 0.02,
        )
        self.decoder = PerceiverDecoder(
            output_adapter = self.output_adapter,
            output_query_provider = self.query_provider,
            num_latent_channels = latent_shape[1],
            num_cross_attention_heads = 4,
            dropout = 0.0,
            init_scale = 0.02,
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    # main()

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "mnist_perceiver_classification_study1"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction='maximize')
    study.optimize(objective, n_trials=100)