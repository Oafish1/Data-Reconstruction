import torch
import torch.nn as nn


class PredictionDataset(torch.utils.data.Dataset):
    """Base dataset class"""
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inputs = self.inputs[idx]
        labels = self.labels[idx]
        return inputs, labels


class Model(nn.Module):
    """Thin, simple NN model"""
    def __init__(self, input_dim, output_dim, internal_dim=10):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, internal_dim)
        self.fc2 = nn.Linear(internal_dim, output_dim)

    def forward(self, x):
        """Forward pass for the model"""
        return self.fc2(self.fc1(x))
