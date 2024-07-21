import random
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, yhat, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(yhat, y) + 1e-6)
        return loss

# MLP model
class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden1,
        hidden2,
        hidden3,
        output_size,
        seed,
    ):
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        super(MLP, self).__init__()

        self.hidden1 = nn.Sequential(
            nn.Linear(int(input_size), int(hidden1)),
            nn.ReLU(),
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(int(hidden1), int(hidden2)),
            nn.ReLU(),
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(int(hidden2), int(hidden3)),
            nn.ReLU(),
        )
        self.output = nn.Linear(int(hidden3), int(output_size))

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.output(x)
        return x


# ResNet model
class ResidualBlock(nn.Module):
    def __init__(self, in_features, dropout=0.5):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(),
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.block(x)

class ResNet(nn.Module):
    def __init__(
        self,
        input_size,
        embedding_size,
        num_residual_blocks,
        dropout,
        output_size,
        seed,
    ):
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        super(ResNet, self).__init__()

        self.embedding = nn.Linear(int(input_size), int(embedding_size))
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(int(embedding_size), dropout) for _ in range(int(num_residual_blocks))]
        )
        self.output = nn.Linear(int(embedding_size), int(output_size))

    def forward(self, x):
        x = self.embedding(x)
        for block in self.residual_blocks:
            x = block(x)
        x = self.output(x)
        return x