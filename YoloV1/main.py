import torch
import multiprocessing
import pytorch_lightning as pl
from datasets import VocDataModule
from model import NN

def main():
    LEARNING_RATE = 2e-5
    DEVICE = "cuda" if torch.cuda.is_available else "cpu"
    BATCH_SIZE = 16
    EPOCHS = 1000
    NUM_WORKERS = 2
    IMG_DIR = "data/images"
    LABEL_DIR = "data/labels"



    model = NN(learning_rate=LEARNING_RATE)
    dm = VocDataModule(train_csv_file="data/train.csv",test_csv_file="data/test.csv",img_dir=IMG_DIR,label_dir=LABEL_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    trainer = pl.Trainer(accelerator="gpu", devices=1, min_epochs=1, max_epochs=EPOCHS, precision=16)
    trainer.fit(model, dm)
    trainer.test(model, dm)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()





