
import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader
from examples.empenhos_df import EMPENHOS



dataset = EMPENHOS(
    train=False, val=False, testing_mode=True, sort_by_Elem_Despesa=True
)  # training dataset



autoencoder = torch.load('models/autoencoder_full.pt', map_location=torch.device('cpu'))

print(dataset[1].shape)
    


losses = []
for i in range(len(dataset)):
    encoded = autoencoder.encoder(dataset[i])
    decoded = autoencoder.decoder(encoded)
    mse = torch.nn.functional.mse_loss(decoded, dataset[i])
    losses.append(mse.item())

print(np.mean(losses)) # 0.00833762885367161
