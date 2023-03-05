from typing import Tuple, List
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch_optimizer as optim

from einops import rearrange, reduce

from perceiver.model.core import InputAdapter, OutputAdapter, PerceiverEncoder, PerceiverDecoder, FourierPositionEncoding, TrainableQueryProvider

dataset_loc = 'MNIST_data'

def main():
    train_loader, test_loader = get_mnist_dataset(batch_size=256)

    example: torch.Tensor = next(iter(train_loader))
    num_classes: int = len(train_loader.dataset.classes)

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

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Lamb(model.parameters(), lr=1e-3)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    num_batches = len(train_loader)

    # train model
    for epoch in range(10):
        print(f'Epoch: {epoch}')
        batch_idx = 0
        pbar = tqdm(train_loader)
        for batch in pbar:
            pbar.set_description(f'Batch {batch_idx}/{num_batches} ')
            batch_idx += 1
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f' Loss: {loss.item()}', end='')

        # test model
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x = x.to(device)
                y = y.to(device)

                pred = model(x)
                _, pred = torch.max(pred, dim=1)
                total += y.size(0)
                correct += (pred == y).sum().item()

        print(f', Accuracy: {correct/total}')

    # save model
    torch.save(model.state_dict(), 'mnist_perceiver.pt')



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


def get_mnist_dataset(batch_size=32, num_workers=4, shuffle=True, pin_memory=True) -> Tuple[DataLoader, DataLoader]:
    MNIST_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root=dataset_loc, train=True, download=True, transform=MNIST_transform)
    test_dataset = datasets.MNIST(root=dataset_loc, train=False, download=True, transform=MNIST_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory)
    return train_loader, test_loader
    
if __name__ == '__main__':
    main()