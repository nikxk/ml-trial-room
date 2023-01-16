import os
import argparse
import numpy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl
from perceiver.model.core import PerceiverIO, PerceiverEncoder, PerceiverDecoder, TrainableQueryProvider
from perceiver.model.core.classifier import ClassificationOutputAdapter
from perceiver.model.vision import ImageInputAdapter
from perceiver.scripts.lrs import ConstantWithWarmupLR
torch.manual_seed(0)

MNIST_data_dir = 'MNIST_data/'
model_save_dir = 'saved-models/perceiver-pl/'

class mnistPerceiver(pl.LightningModule):
    '''A Perceiver for MNIST classification.'''
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.criterion = nn.NLLLoss()

        self.input_adapter = ImageInputAdapter(
            image_shape = (28, 28, 1),
            num_frequency_bands = 32
        )
        self.encoder = PerceiverEncoder(
            input_adapter = self.input_adapter,
            num_latents = 16,
            num_latent_channels = 4,
            num_cross_attention_layers = 2,
            num_self_attention_layers_per_block = 2,
            num_self_attention_blocks = 3
            )
        self.query_provider = TrainableQueryProvider(
            num_queries = 1,
            num_query_channels = 10
        )
        self.output_adapter = ClassificationOutputAdapter(
            num_classes = 10,
            num_output_query_channels = 10
        )
        self.decoder = PerceiverDecoder(
            output_adapter = self.output_adapter,
            output_query_provider = self.query_provider,
            num_latent_channels = 4,
            num_cross_attention_heads=2
        )
        self.perceiver = PerceiverIO(
            encoder = self.encoder,
            decoder = self.decoder
        )
        
    def forward(self, x):
        x = x.view(x.shape[0], 28, 28, 1)
        x = torch.log_softmax(self.perceiver(x), dim=1)
        return x

    def prepare_data(self):
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        datasets.MNIST(MNIST_data_dir, download=True, train=True)

    def train_dataloader(self):
        mnist_trainset = datasets.MNIST(MNIST_data_dir, train=True, download=True, transform=self.transform)
        mnist_trainloader = DataLoader(mnist_trainset, batch_size=self.batch_size, shuffle=True, num_workers=12)
        return mnist_trainloader

    def val_dataloader(self):
        mnist_valset = datasets.MNIST(MNIST_data_dir, train=False, download=True, transform=self.transform)
        mnist_valloader = DataLoader(mnist_valset, batch_size=self.batch_size, shuffle=False, num_workers=12)
        return mnist_valloader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = ConstantWithWarmupLR(optimizer, warmup_steps=500)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)

def get_args():
    '''Set up the parser and get args.'''
    # set up the parser
    parser = argparse.ArgumentParser(description='Train a Perceiver with MNIST in PyTorch Lightning.')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    # get args
    args = parser.parse_args()
    return args

def get_latest_model_path():
    '''Get the latest model in the saved-models directory.'''
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


if __name__ == '__main__':
    args = get_args()
    get_latest_model_path()

    model = mnistPerceiver(batch_size=args.batch_size)
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=args.epochs, enable_checkpointing=True, default_root_dir=model_save_dir)
    trainer.fit(model)