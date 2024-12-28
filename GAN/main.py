

import torch
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import random_split
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Metric
import multiprocessing


class Discriminator(pl.LightningModule):
    def __init__(self, batch_size, img_dim=784):
        super().__init__()
        self.img_dim = img_dim
        self.batch_size = batch_size

        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),        # (Batchsize,784) -> (Batchsize,128)
            nn.LeakyReLU(0.2),                       # (Batchsize,128)
            nn.BatchNorm1d(128),                    # (Batchsize,128)
            nn.Linear(128,1),    # (Batchsize,1)
            nn.Sigmoid()                             # (Batchsize,1)
        )


    def forward(self,x):
        x = self.disc(x)
        return x


class Generator(pl.LightningModule):
    def __init__(self, batch_size, z_dim, img_dim=784):
        super().__init__()
        self.img_dim = img_dim
        self.batch_size = batch_size
        self.z_dim = z_dim

        self.gen = nn.Sequential(
            nn.Linear(self.z_dim, 256),  # (Batchsize,z) -> (Batchsize,256)
            nn.LeakyReLU(0.2),                       # (Batchsize,256)
            nn.BatchNorm1d(256),              # (Batchsize,256)
            nn.Linear(256,img_dim),    # (Batchsize,img_dim)
            nn.Tanh()
        )

    def forward(self,x):
        x = self.gen(x)
        return x



class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, dir, num_workers):
        super().__init__()
        self.dir = dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5,),(0.5,))])
        self.train_ds = None


    def prepare_data(self):
        datasets.MNIST(self.dir, train=True, download=True)


    def setup(self, stage: str):
        entire_dataset = datasets.MNIST(
            root=self.dir,
            train=True,
            transform=self.transforms,
            download=False,
        )
        self.train_ds = entire_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True
        )




class GAN(pl.LightningModule):
    def __init__(self, batch_size, z_dim, img_dim=784, lr_g=1e-4, lr_d=4e-4):
        super().__init__()
        self.save_hyperparameters()

        self.G = Generator(batch_size=batch_size, z_dim=z_dim, img_dim=img_dim)
        self.D = Discriminator(batch_size=batch_size, img_dim=img_dim)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def forward(self, z):
        return self.G(z)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch
        imgs = imgs.view(imgs.size(0), -1)

        # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.z_dim)
        z = z.type_as(imgs)

        # train generator
        if optimizer_idx == 0:
            # generate images
            self.generated_imgs = self(z)

            # log sampled images
            sample_imgs = self.generated_imgs[:6]
            grid = torchvision.utils.make_grid(sample_imgs.view(-1, 1, 28, 28))
            self.logger.experiment.add_image('generated_images', grid, 0)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.D(self(z)), valid)
            self.log("g_loss", g_loss, prog_bar=True)
            return g_loss

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            real_loss = self.adversarial_loss(self.D(imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)

            fake_loss = self.adversarial_loss(
                self.D(self(z).detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            self.log("d_loss", d_loss, prog_bar=True)
            return d_loss

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.G.parameters(), lr=self.hparams.lr_g)
        opt_d = torch.optim.Adam(self.D.parameters(), lr=self.hparams.lr_d)
        return [opt_g, opt_d], []


def main():
    # Hyperparameters
    BATCH_SIZE = 64
    Z_DIM = 100
    IMG_DIM = 784  # 28x28 pixels
    LR_G = 1e-4
    LR_D = 4e-4
    NUM_WORKERS = 4
    MAX_EPOCHS = 50

    # Data
    dm = MNISTDataModule(batch_size=BATCH_SIZE, dir='./data', num_workers=NUM_WORKERS)

    # Model
    model = GAN(batch_size=BATCH_SIZE, z_dim=Z_DIM, img_dim=IMG_DIM, lr_g=LR_G, lr_d=LR_D)

    # Logger
    logger = TensorBoardLogger("tb_logs", name="mnist_gan")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='gan-{epoch:02d}-{g_loss:.2f}',
        save_top_k=3,
        monitor='g_loss',
        mode='min'
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",  # Uses GPU if available, otherwise CPU
        devices="auto",      # Uses all available GPUs or 1 CPU
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    # Train the model
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()





