from typing import Tuple, List
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch_optimizer as optim

from einops import rearrange, repeat

import optuna
import logging
import sys

from perceiver.model.core import InputAdapter, OutputAdapter, PerceiverEncoder, PerceiverDecoder, FourierPositionEncoding, TrainableQueryProvider, QueryProvider

from utils import *

dataset_loc = 'MNIST_data/'

def main():
    train_loader, val_loader, test_loader = get_mnist_dataset(dataset_loc=dataset_loc, batch_size=64, num_workers=4)

    example: torch.Tensor = next(iter(train_loader))
    num_classes: int = len(train_loader.dataset.dataset.classes)

    loss_fn = ReconImageLoss(alpha_class=0.1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # initialize model
    model = mnistPerceiverAutoencoder(
        image_shape = example[0].shape[1:],
        num_classes = num_classes,
        num_frequency_bands = 12,
        latent_shape = (16, 16), 
        num_CA_blocks=2,
        num_SA_blocks=2,
        num_SA_layers=4,
        num_CA_out_heads=4,
        )
    
    model.load_state_dict(torch.load('mnist_perceiver_autoencoder.pt'))

    # train model
    optimizer = optim.Lamb(model.parameters(), lr=2e-3)
    model = train_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        num_epochs=10,
        patience=3,
        verbose=True
        )

    # test model
    test_image_loss, test_class_loss, test_net_loss, test_acc = evaluate_autoencoder(model, test_loader, loss_fn, device)
    print(f'--Test--\nrecon loss: {test_image_loss:.4f}, class loss: {test_class_loss:.4f}, accuracy: {test_acc:.4f}, total loss: {test_net_loss:.4f}')

    # save model
    torch.save(model.state_dict(), 'mnist_perceiver_autoencoder.pt')

def conduct_study():
    # basic setup
    train_loader, val_loader, test_loader = get_mnist_dataset(dataset_loc=dataset_loc, batch_size=64, num_workers=4)

    example: torch.Tensor = next(iter(train_loader))
    num_classes: int = len(train_loader.dataset.dataset.classes)

    loss_fn = ReconImageLoss(alpha_class=0.1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def objective(trial: optuna.Trial):
        num_latent_indices = trial.suggest_int('num_latent_indices', 8, 24, step=4)
        num_latent_channels_quart = trial.suggest_int('num_latent_channels+quart', 2, 8, log=True)
        num_frequency_bands = trial.suggest_int('num_frequency_bands', 4, 16, step=4)
        num_CA_out_heads_pow = trial.suggest_int('num_CA_out_heads_pow', 1, 3)
        num_CA_out_heads = 2 ** num_CA_out_heads_pow
        num_latent_channels = num_latent_channels_quart * 4

        # initialize model
        model = mnistPerceiverAutoencoder(
            image_shape = example[0].shape[1:],
            num_classes = num_classes,
            num_frequency_bands = num_frequency_bands,
            latent_shape = (num_latent_indices, num_latent_channels), 
            num_CA_blocks=2,
            num_SA_blocks=2,
            num_SA_layers=4,
            num_CA_out_heads=num_CA_out_heads,
            )
        # train model
        optimizer = optim.Lamb(model.parameters(), lr=2e-3)
        model = train_autoencoder(
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
        test_image_loss, test_class_loss, test_net_loss, test_acc = evaluate_autoencoder(model, test_loader, loss_fn, device)
        print(f'--Test--\nrecon loss: {test_image_loss:.4f}, class loss: {test_class_loss:.4f}, accuracy: {test_acc:.4f}, total loss: {test_net_loss:.4f}')

        return test_net_loss

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "mnist_perceiver_autoencoder_study1"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction='minimize')
    study.optimize(objective, n_trials=100)



class ReconImageLoss(nn.Module):
    def __init__(self, alpha_class=1.):
        super().__init__()
        self.alpha_class = alpha_class
        self.loss_class = nn.CrossEntropyLoss()
        self.loss_mse = nn.MSELoss()
        self.loss_l1 = nn.L1Loss()

    def loss_recon(self, x_pred, x):
        return self.loss_mse(x_pred, x) + 0.5 * self.loss_l1(x_pred, x)

    def get_losses(self, x, y, x_pred, y_pred):
        return self.loss_recon(x_pred, x), self.loss_class(y_pred, y)

    def forward(self, x, y, x_pred, y_pred):
        return self.alpha_class * self.loss_class(y_pred, y) + self.loss_recon(x_pred, x)

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
        x = rearrange(x, "b c h w -> b (h w) c")
        return torch.cat([x, x_enc], dim=-1)

class ReconImageQueryProvider(nn.Module, QueryProvider):
    '''A query provider for reconstruction and classification.'''
    def __init__(self, position_encoding: FourierPositionEncoding):
        super().__init__()
        self.position_encoding = position_encoding
        self._class_query = TrainableQueryProvider(num_queries=1, num_query_channels=self.num_query_channels)
        self._is_image = TrainableQueryProvider(num_queries=1, num_query_channels=self.num_query_channels - self.position_encoding.num_position_encoding_channels())
        self.num_pixels = self.position_encoding(1).shape[1]
        
    @property
    def num_query_channels(self):
        return 8 * (1 + self.position_encoding.num_position_encoding_channels() // 8)

    def forward(self, x=None):
        return torch.cat([
            torch.cat([
                self.position_encoding(1), 
                repeat(self._is_image(), '1 1 c -> 1 p c', p=self.num_pixels)
                ], dim=-1), 
            self._class_query(),
            ], dim=1)

class ReconImageOutputAdapter(OutputAdapter):
    '''An output adapter for classification.'''
    def __init__(self, num_classes, num_output_query_channels, image_shape):
        super().__init__()
        self.num_image_channels, *self.image_spatial_shape = image_shape
        self.pixel_remap = nn.Linear(num_output_query_channels, self.num_image_channels)
        self.class_remap = nn.Linear(num_output_query_channels, num_classes)

    def forward(self, x):
        x_image = x[:, :-1]
        x_class = x[:, -1]

        num_pixels = x_image.shape[1]
        assert self.image_spatial_shape[0] * self.image_spatial_shape[1] == num_pixels, f"Image spatial shape {self.image_spatial_shape} does not match number of pixels {num_pixels}"
        x_image = rearrange(x_image, 'b p c -> (b p) c')
        x_image = self.pixel_remap(x_image)
        x_image = rearrange(x_image, '(b h w) c -> b c h w', h=self.image_spatial_shape[0], w=self.image_spatial_shape[1])

        x_class = self.class_remap(x_class)
        return x_image, x_class

class mnistPerceiverAutoencoder(nn.Module):
    '''A Perceiver for MNIST classification.'''
    def __init__(self, 
        image_shape: Tuple[int, ...], 
        num_classes: int, 
        num_frequency_bands=10, 
        latent_shape: Tuple[int, int]=(8, 8), 
        num_CA_blocks: int=2,
        num_SA_blocks: int=2,
        num_SA_layers: int=4,
        num_CA_out_heads: int=4,
    ):
        super().__init__()
        self.image_shape = image_shape

        self.input_adapter = ImageInputAdapter(
            image_shape = self.image_shape,
            num_frequency_bands = num_frequency_bands
        )
        self.position_encoding: FourierPositionEncoding = self.input_adapter.position_encoding
        self.query_provider = ReconImageQueryProvider(self.position_encoding)
        self.output_adapter = ReconImageOutputAdapter(
            num_classes = num_classes,
            num_output_query_channels = self.query_provider.num_query_channels,
            image_shape=self.image_shape,
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
            num_cross_attention_heads = num_CA_out_heads,
            dropout = 0.0,
            init_scale = 0.02,
        )

    def forward(self, x):
        x_latent = self.encoder(x)
        x_image, x_class = self.decoder(x_latent)
        return x_image, x_class


if __name__ == '__main__':
    main()
    # conduct_study()