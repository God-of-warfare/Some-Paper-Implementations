

import torch



from torch import nn, optim
from torch.utils.data import DataLoader


import torch.nn as nn
import pytorch_lightning as pl
from loss import YoloLoss

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.bn(self.conv(x)))

class NN(pl.LightningModule):
    def __init__(self, input_size=448, num_channels=3, learning_rate=2e-5, **kwargs):
        super().__init__()
        self.loss_fn = YoloLoss()
        self.input_size = input_size
        self.num_channels = num_channels
        self.learning_rate = learning_rate
        self.layers = nn.Sequential(
            ConvBlock(num_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(192, 128, kernel_size=1),
            ConvBlock(128, 256, kernel_size=3, padding=1),
            ConvBlock(256, 256, kernel_size=1),
            ConvBlock(256, 512, kernel_size=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *[nn.Sequential(
                ConvBlock(512, 256, kernel_size=1),
                ConvBlock(256, 512, kernel_size=3, padding=1)
            ) for _ in range(4)],
            ConvBlock(512, 512, kernel_size=1),
            ConvBlock(512, 1024, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *[nn.Sequential(
                ConvBlock(1024, 512, kernel_size=1),
                ConvBlock(512, 1024, kernel_size=3, padding=1)
            ) for _ in range(2)],
            ConvBlock(1024, 1024, kernel_size=3, padding=1),
            ConvBlock(1024, 1024, kernel_size=3, stride=2, padding=1),
            ConvBlock(1024, 1024, kernel_size=3, padding=1),
            ConvBlock(1024, 1024, kernel_size=3, padding=1),
        )

        self.fc1 = nn.Linear(50176, 4096)
        self.fc2 = nn.Linear(4096, 1470)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.leakyrelu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        loss = self.loss_fn(predictions, y)
        self.log_dict({'train_loss': loss},
                      on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss, "scores": predictions, "y": y}

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log('test_loss', loss)
        self.log_dict({'test_loss': loss})
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds

    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)









