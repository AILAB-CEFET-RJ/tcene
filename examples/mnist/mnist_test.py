import click
import sys
import os
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
from tensorboardX import SummaryWriter
import uuid
# internal files
# Add the parent directory of 'ptdec' and 'ptsdae' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ptdec.dec import DEC
from ptdec.model import train, predict
from ptdec.utils import cluster_accuracy

from ptsdae.sdae import StackedDenoisingAutoEncoder
import ptsdae.model as ae




def main(cuda=False, batch_size=256, pretrain_epochs=300, finetune_epochs=500, testing_mode=False):
    
    
    
    # Define transform (optional: normalization or flattening)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL Image to Tensor and scales to [0, 1]
        transforms.Lambda(lambda x: x.view(-1))  # Flatten from [1, 28, 28] to [784]
    ])

    # Load datasets
    ds_train = MNIST(root="./data", train=True, download=True, transform=transform)
    ds_val = MNIST(root="./data", train=False, download=True, transform=transform)
    
    
    img_train, label_train = ds_train[0]
    print(label_train)
    
    
    autoencoder = StackedDenoisingAutoEncoder(
            [28 * 28, 500, 500, 2000, 10], final_activation=None
    )
    
    ae.pretrain(
        ds_train,
        autoencoder,
        cuda=cuda,
        validation=ds_val,
        epochs=pretrain_epochs,
        batch_size=batch_size,
        optimizer=lambda model: SGD(model.parameters(), lr=0.1, momentum=0.9), #  it means the current update is made of 90% of the previous update (momentum) and 10% of the new gradient
        scheduler=lambda x: StepLR(x, 100, gamma=0.1), # gamma decay rate
        corruption=0.2, # introducing noise or modifying a percentage of the input data
    )




if __name__ == "__main__":
    main()
