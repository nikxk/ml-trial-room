import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils import tensorboard

from torchvision import datasets, transforms

from typing import Tuple, List, Callable
from tqdm import tqdm


def get_mnist_dataset(dataset_loc: str, batch_size: int=32, num_workers: int=4, shuffle=True, pin_memory=True) -> Tuple[DataLoader, DataLoader]:
    MNIST_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root=dataset_loc, train=True, download=True, transform=MNIST_transform)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [50000, 10000])
    test_dataset = datasets.MNIST(root=dataset_loc, train=False, download=True, transform=MNIST_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader

def train_classifier(
    model: nn.Module, 
    train_loader: DataLoader, 
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer, 
    loss_fn: Callable, 
    device: torch.device, 
    num_epochs: int=100, 
    patience: int=5, 
    verbose: bool=True
) -> nn.Module:
    model.to(device)
    num_batches = len(train_loader)
    best_loss = float('inf')
    best_model = model
    early_stop = 0

    for epoch in range(num_epochs):
        if verbose:
            print(f'Epoch: {epoch}')

        model.train()
        batch_idx = 0
        total_loss = 0.
        pbar = tqdm(train_loader) if verbose else train_loader
        for batch in pbar:
            if verbose and batch_idx>0:
                pbar.set_description(f'Train loss: {total_loss/batch_idx:.4f}')
            batch_idx += 1
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if verbose:
            print(f'Validating... ', end='')
        val_loss, val_acc = evaluate_classifier(model, val_loader, loss_fn, device)
        if verbose:
            print(f'validation loss: {val_loss:.4f}, validation accuracy: {val_acc:.4f}')
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model
            early_stop = 0
        else:
            early_stop += 1
            if early_stop == patience:
                break
    return best_model
    
def evaluate_classifier(model: nn.Module, loader: DataLoader, loss_fn: Callable, device: torch.device) -> float:
    model.to(device)
    model.eval()
    num_batches = len(loader)
    total_loss = 0.
    correct = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            correct += (pred.argmax(dim=1) == y).sum().item()
    return total_loss/num_batches, correct/len(loader.dataset)

    
def train_autoencoder(
    model: nn.Module, 
    train_loader: DataLoader, 
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer, 
    loss_fn: Callable,
    device: torch.device, 
    num_epochs: int=100, 
    patience: int=5, 
    verbose: bool=True
) -> nn.Module:
    model.to(device)
    num_batches = len(train_loader)
    best_loss = float('inf')
    best_model = model
    early_stop = 0

    for epoch in range(num_epochs):
        if verbose:
            print(f'Epoch: {epoch}')

        model.train()
        batch_idx = 0
        total_loss = 0.
        pbar = tqdm(train_loader) if verbose else train_loader
        for batch in pbar:
            if verbose and batch_idx>0:
                pbar.set_description(f'Train loss: {total_loss/batch_idx:.4f}')
            batch_idx += 1
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            pred_image, pred_class = model(x)
            loss = loss_fn(x, y, pred_image, pred_class) 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if verbose:
            print(f'Validating... ', end='')
        val_image_loss, val_class_loss, val_net_loss, val_acc = evaluate_autoencoder(model, val_loader, loss_fn, device)
        if verbose:
            print(f'recon loss: {val_image_loss:.4f}, class loss: {val_class_loss:.4f}, accuracy: {val_acc:.4f}, total loss: {val_net_loss:.4f}')
        if val_net_loss < best_loss:
            best_loss = val_net_loss
            best_model = model
            early_stop = 0
        else:
            early_stop += 1
            if early_stop == patience:
                break
    return best_model
    
def evaluate_autoencoder(model: nn.Module, loader: DataLoader, loss_fn: Callable, device: torch.device) -> Tuple[float, ...]:
    model.to(device)
    model.eval()
    num_batches = len(loader)
    total_image_loss, total_class_loss, total_loss = 0., 0., 0.
    correct = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            pred_image, pred_class = model(x)
            loss_image, loss_class = loss_fn.get_losses(x, y, pred_image, pred_class)

            total_image_loss += loss_image.item()
            total_class_loss += loss_class.item()
            correct += (pred_class.argmax(dim=1) == y).sum().item()
    total_loss = total_image_loss + loss_fn.alpha_class * total_class_loss
    return total_image_loss/num_batches, total_class_loss/num_batches, total_loss/num_batches, correct/len(loader.dataset)