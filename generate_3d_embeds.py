import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data.empenhos_df import EMPENHOS
import torch
import os
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from ptsdae.sdae import StackedDenoisingAutoEncoder
import ptsdae.model as ae

import tensorflow as tf
import os
import numpy as np
from tensorboard.plugins import projector

if not os.path.exists('outputs/embedded_3d.npy'):

    if not os.path.exists('outputs/models/autoencoder_3d.pt'):
        ds_train = EMPENHOS(
            train=True, val=False, testing_mode=False
            )  # training dataset
            
        ds_val = EMPENHOS(
            train=False, val=True, testing_mode=False
        )  # evaluation dataset

        
        
        autoencoder = StackedDenoisingAutoEncoder(
                [387, 300, 300, 1000, 3], final_activation=None
            )
        
        ae.pretrain(
            ds_train,
            autoencoder,
            cuda=False,
            validation=ds_val,
            epochs=20,
            batch_size=256,
            optimizer=lambda model: SGD(model.parameters(), lr=0.1, momentum=0.9), 
            scheduler=lambda x: StepLR(x, 100, gamma=0.1),
            corruption=0.2, 
            silent=True,
            update_freq=1,
        )
        
        print("Training stage - Autoencoder.")
        ae_optimizer = SGD(params=autoencoder.parameters(), lr=0.1, momentum=0.9)
        ae.train( #
            ds_train,
            autoencoder,
            cuda=False,
            validation=ds_val,
            epochs=20,
            batch_size=256,
            optimizer=ae_optimizer,
            scheduler=StepLR(ae_optimizer, 100, gamma=0.1),
            corruption=0.2,
            silent=True,
            update_freq=1,
        )
        
        torch.save(autoencoder, os.path.join("outputs/models", "autoencoder_3d.pt"))

    else:
        autoencoder = torch.load('outputs/models/autoencoder_3d.pt', map_location=torch.device('cpu'), weights_only=False) 
        
    dataset = EMPENHOS(
        train=False, val=False, testing_mode=False
    )
    vectors = []
    for index in range(129):
        sub_ds, _ = dataset.X_by_elem(index)
        
        encoded = autoencoder.encoder(sub_ds) # encoder que faz output de 3 dimensoes
        
        
        # Convert encoded tensor to numpy array if needed
        array = encoded.detach().cpu().numpy()
        array_avg = np.mean(array, axis=0)  # shape: (3,)
        
        vectors.append(array_avg)
        
    # Stack all arrays vertically
    vectors_np = np.vstack(vectors)
    
    # if vectors_np.ndim == 3:
    #     vectors_np = vectors_np.reshape(-1, 3)
    np.save('outputs/embedded_3d.npy', vectors_np)

else:
    data = np.load('outputs/embedded_3d.npy')


if not os.path.exists('outputs/vectors.tsv'):
    np.savetxt("outputs/vectors.tsv", data, delimiter="\t")


vectors = np.loadtxt("outputs/vectors.tsv", delimiter="\t")



weights = tf.Variable(vectors, name='embedding')

# Create log directory
log_dir = "logs/projector"
os.makedirs(log_dir, exist_ok=True)

# Save embeddings as checkpoint
checkpoint = tf.train.Checkpoint(embedding=weights)
checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

# Save metadata
with open(os.path.join(log_dir, 'metadata.tsv'), 'w') as f:
    for i in range(len(vectors)):
        f.write(f"{i}\n")

# Set up config.
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding.metadata_path = "metadata.tsv"

# Save config file
projector.visualize_embeddings(log_dir, config)


# tensorboard --logdir /logs/projector


# troubleshoot:
# https://github.com/tensorflow/tensorboard/blob/master/README.md#my-tensorboard-isnt-showing-any-data-whats-wrong