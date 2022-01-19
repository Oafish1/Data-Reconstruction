from commando import ComManDo
import torch
import torch.nn as nn
from unioncom.UnionCom import UnionCom

from .model_classes import PredictionDataset


USE_COMMANDO = True


def joint_embed(*datasets, **hyperparams):
    """Perform joint embedding on any number of datasets"""
    if USE_COMMANDO:
        constructor_class = ComManDo
    else:
        constructor_class = UnionCom
    return constructor_class(**hyperparams).fit_transform([*datasets])


def create_dataloader(inputs, labels):
    """Create a dataloader for the given inputs and labels"""
    dataset = PredictionDataset(inputs, labels)
    return torch.utils.data.DataLoader(dataset)


def train_model(
    model,
    dataloader,
    criterion=nn.MSELoss(),
    epochs=100,
    log_epoch=50,
    optimizer=torch.optim.AdamW,
):
    """Train a ``data_reconstruct.model_classes.Model`` model"""
    optimizer = optimizer(model.parameters())

    for epoch in range(epochs):
        epoch_loss = 0
        for i, data in enumerate(dataloader):
            inputs, labels = data
            optimizer.zero_grad()

            # Forward
            logits = model(inputs)

            # Backward
            loss = criterion(logits.float(), labels.float())
            loss.backward()
            optimizer.step()

            # Bookkeeping
            epoch_loss += loss

        # CLI Output
        if epoch % log_epoch == log_epoch-1:
            print(f'Epoch: {epoch + 1:>3}    Loss: {epoch_loss / (i + 1): .5f}')


def run_validation(
    model,
    dataloader,
    criterion=nn.MSELoss(),
):
    """Run validation set using a given ``model`` and ``dataloader``"""
    validation_loss = 0
    for i, data in enumerate(dataloader):
        inputs, labels = data
        logits = model(inputs)
        validation_loss += criterion(logits.float(), labels.float())
    print(f'Validation Loss: {validation_loss / (i+1):.5f}')
