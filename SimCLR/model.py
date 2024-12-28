import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch import optim
from torch.nn import functional as F
import torchvision.transforms.functional as func
import torch.utils as U
import torch.optim
from torch.optim import Adam
import numpy as np
import torchvision.transforms as transforms
from typing import Optional
import pytorch_lightning as pl
from pl_bolts.models.self_supervised.resnets import resnet50
from pl_bolts.models.self_supervised.evaluator import Flatten
from torch.utils.data import random_split, DataLoader


class SimclrTrainsetTransforms(object):
    def __init__(self, jitter_strength:float = 1,gaussian_blur:bool = False, input_height:int = 224,normalize: Optional[transforms.Normalize] = None):
        self.jitter_strength = jitter_strength
        self.gaussian_blur = gaussian_blur
        self.input_height = input_height
        self.normalize = normalize
        self.color_jitter = transforms.ColorJitter(0.8 * self.jitter_strength,0.8 * self.jitter_strength,0.8 * self.jitter_strength,0.2 * self.jitter_strength)
        data_transforms = [transforms.RandomResizedCrop(size=self.input_height),transforms.RandomHorizontalFlip(p=0.5),transforms.RandomApply([self.color_jitter], p=0.8),transforms.RandomGrayscale(p=0.2)]

        if self.gaussian_blur:
            data_transforms.append(GaussianBlur(kernel_size=int(0.1 * self.input_height)))

        data_transforms.append(transforms.ToTensor())

        if self.normalize:
            data_transforms.append(normalize)

        self.transform = transforms.Compose(data_transforms)

    def __call__(self, sample):
        transform = self.transform

        xi = transform(sample)
        xj = transform(sample)

        return xi, xj


class SimclrEvalDataTransform(object):
    def __init__(
        self,
        input_height: int = 224,
        normalize: Optional[transforms.Normalize] = None
    ):
        self.input_height = input_height
        self.normalize = normalize

        data_transforms = [
            transforms.Resize(self.input_height),
            transforms.ToTensor()
        ]

        if self.normalize:
            data_transforms.append(normalize)

        self.test_transform = transforms.Compose(data_transforms)

    def __call__(self, sample):
        transform = self.test_transform

        xi = transform(sample)
        xj = transform(sample)

        return xi, xj


class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, p=0.5, min=0.1, max=2.0):
        self.min = min
        self.max = max

        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, sample):
        sample = func.to_tensor(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < self.p:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = func.gaussian_blur(kernel_size=self.kernel_size,sigma=sigma)

        return sample



def nt_xent_loss(out_1, out_2, temperature):

    # out_1 and out_2 are (b, 128)
    out = torch.cat([out_1, out_2], dim=0)  # (2b, 128)
    n_samples = len(out)  # 2b

    # Full similarity matrix
    # (2b, 128) x (128, 2b) -> (2b, 2b)
    cov = torch.mm(out, out.t().contiguous())  # Each element i,j represents the similarity between two samples i,j
    sim = torch.exp(cov / temperature)  # Exponentiate

    mask = ~torch.eye(n_samples, device=sim.device).bool()  # True for the off-diagonal elements. Diagonal elements are similar anyway
    neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)  # (2b,)  # The denominator term of NT-Xent loss

    # Positive similarity                  (b,)
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)  # The numerator term of NT-Xent loss
    pos = torch.cat([pos, pos], dim=0)  # (2b,)

    loss = -torch.log(pos / neg).mean()
    return loss

class Projection(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(self.input_dim, self.hidden_dim, bias=True),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False))

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)



class DataModule(pl.LightningDataModule):
    def __init__(self,data_dir, batch_size, num_workers):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage):
        entire_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            transform=SimclrTrainsetTransforms(),
            download=False,
        )

        self.train_ds, self.val_ds = random_split(entire_dataset, [50000, 10000])

        self.test_ds = datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            transform=SimclrEvalDataTransform(),
            download=False,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True
        )



class SimCLR(pl.LightningModule):
    def __init__(self,lr=1e-4,loss_temperature=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.nt_xent_loss = nt_xent_loss

        self.encoder = self.init_encoder()
        self.projection = Projection()

    def init_encoder(self):
        encoder = resnet50(return_all_feature_maps=False)

        # when using cifar10, replace the first conv so image doesn't shrink away
        encoder.conv1 = nn.Conv2d(
            3, 64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        return encoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.projection(x)
        return x

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)

        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss, on_epoch=True)
        return result

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)

        result = pl.EvalResult(checkpoint_on=loss)
        result.log('avg_val_loss', loss)
        return result

    def shared_step(self, batch, batch_idx):
        (img1, img2), y = batch

        # ENCODE
        # encode -> representations
        # (b, 3, 32, 32) -> (b, 2048, 2, 2)
        h1 = self.encoder(img1)
        h2 = self.encoder(img2)

        # the bolts resnets return a list of feature maps
        if isinstance(h1, list):
            h1 = h1[-1]
            h2 = h2[-1]

        # (b, 2048, 2, 2) -> (b, 128)
        z1 = self.projection(h1)
        z2 = self.projection(h2)

        loss = self.nt_xent_loss(z1, z2, self.hparams.loss_temperature)

        return loss


    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.hparams.lr)





