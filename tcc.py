import click
import sys
import os
import yaml
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
import torch
from tensorboardX import SummaryWriter
import uuid

from data.empenhos_df import EMPENHOS
from ptdec.dec import DEC
from ptdec.model import train, predict
from ptdec.utils import cluster_accuracy
from ptsdae.sdae import StackedDenoisingAutoEncoder
import ptsdae.model as ae
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import silhouette_score

from sklearn.cluster import KMeans

@click.command()
@click.option(
    "--cuda", help="whether to use CUDA (default False).", type=bool, default=False
)

@click.option(
    "--testing-mode",
    help="whether to run in testing mode (default False).",
    type=bool,
    default=False,
)

@click.option(
    "--train-autoencoder",
    help="whether to train autoencoder from scratch or to load an existing one (default True).",
    type=bool,
    default=True,
)

@click.option(
    "--sort-by-elem",
    help="whether to train DEC model for each type of ElemDespesaTCE (default True).",
    type=bool,
    default=False,
)

# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
def main(cuda, testing_mode, train_autoencoder, sort_by_elem):
    writer = SummaryWriter()  # create the TensorBoard object
    # SummaryWriter is a class provided by PyTorch to log data 
    # callback function to call during training, uses writer from the scope
    
    save_dec = True
    save_autoencoder = True
        
    # Open the configuration file and load the different arguments
    print(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    batch_size = config['batch_size_models']
    pretrain_epochs = config['pretrain_epochs']
    finetune_epochs = config['finetune_epochs']
    
    

    def training_callback(epoch, lr, loss, validation_loss): # summary writer
        writer.add_scalars(
            "data/autoencoder",
            {"lr": lr, "loss": loss, "validation_loss": validation_loss,},
            epoch,
        )

    hidden_layer = config['hidden_layer']
    if train_autoencoder:
        
        ds_train = EMPENHOS(
        train=True, val=False, testing_mode=testing_mode
        )  # training dataset
        
        ds_val = EMPENHOS(
            train=False, val=True, testing_mode=testing_mode
        )  # evaluation dataset
    
        ## ---------------------------------------------------------------------------- ##
        # Future work = Hiperparameter-tune hidden_layers
        autoencoder = StackedDenoisingAutoEncoder(
            [config['input_dim'], 300, 300, 1000, hidden_layer], final_activation=None
        )
        ## ---------------------------------------------------------------------------- ##
        
        if cuda:
            torch.cuda.set_device(1)
            autoencoder.cuda()

    
        print("Pretraining stage - Autoencoder.")
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
            update_freq=1,
        )
        
        print("Training stage - Autoencoder.")
        ae_optimizer = SGD(params=autoencoder.parameters(), lr=0.1, momentum=0.9)
        ae.train( #
            ds_train,
            autoencoder,
            cuda=cuda,
            validation=ds_val,
            epochs=finetune_epochs,
            batch_size=batch_size,
            optimizer=ae_optimizer,
            scheduler=StepLR(ae_optimizer, 100, gamma=0.1),
            corruption=0.2,
            update_callback=training_callback,
            update_freq=1,
        )
        
        if (save_autoencoder):
            torch.save(autoencoder, os.path.join(config['saved_models_dir'], "autoencoder_full.pt"))
    else:
        models_dir = config['saved_models_dir']
        # autoencoder = torch.load(f'{models_dir}/autoencoder_full.pt', map_location=torch.device('cpu')) # python version < 3.11
        autoencoder = torch.load(f'{models_dir}/autoencoder_full.pt', map_location=torch.device('cpu'), weights_only=False) 
        
        
    
    print("DEC stage.")
    output_predicted_dir = config['output_predicted_dir']
    lr_dec = config['lr_DEC_opt'] 
    
    dataset = EMPENHOS(
        train=False, val=False, testing_mode=testing_mode
    )
    
    if sort_by_elem:
        
        ss = []
        optimal_ks_by_elem = config['optimal_ks_by_elem']
        
        for i in range(dataset.lenElemDespesaTCE()): # for each elem. despesa tce
            subds, _ = dataset.X_by_elem(i)
            if optimal_ks_by_elem[i] == 1:
                predicted = np.zeros(len(subds)) # if there is just 1 cluster, there is nothing to predict
                np.save(f'{output_predicted_dir}/predicted_elem_{i}.npy', predicted)
            else:
                
                model = DEC(cluster_number=optimal_ks_by_elem[i], hidden_dimension=hidden_layer, encoder=autoencoder.encoder)
                if cuda:
                    model.cuda()
                epochs_dec = config['epochs_dec_elem']
                dec_optimizer = SGD(model.parameters(), lr=0.01, momentum=0.8, weight_decay=1e-4)# regularization to avoid overfitting on small (10D) embeddings
                train(
                    dataset=subds,
                    model=model,
                    epochs=epochs_dec,
                    batch_size=batch_size,
                    optimizer=dec_optimizer,
                    scheduler= StepLR(dec_optimizer, 15, gamma=0.5),
                    stopping_delta=0.000001,
                    cuda=cuda,
                    evaluate_batch_size=512,
                    silent=True ######
                )
                X, predicted = predict( # we don't have actual values, so False
                    subds, model, batch_size=512, silent=True, return_actual=False, cuda=cuda
                )
                np.save(f'{output_predicted_dir}/predicted_elem_{i}.npy', predicted)

                if len(np.unique(predicted)) > 1:
                    score = silhouette_score(subds, predicted)
                    print(f"Index = {i}, Clusters = {optimal_ks_by_elem[i]} ,SS= {score:.4f}")
                    ss.append(score)
                else:
                    print(f"Index = {i}, Clusters = {optimal_ks_by_elem[i]}, SS cannot be computed (only one cluster).")
                    ss.append(None)
                    
                    
                if (save_dec):
                    torch.save(model, os.path.join(config['saved_models_sorted_dir'], f"dec_model_{i}.pt"))
        np.save(f'silhouette_scores{i}.npy', ss)
    else:
        num_clusters = config['num_clusters_testing'] if testing_mode else config['num_clusters']
        
        model = DEC(cluster_number=num_clusters, hidden_dimension=hidden_layer, encoder=autoencoder.encoder)
        if cuda:
            model.cuda()
        
        
        dec_optimizer = SGD(model.parameters(), lr=0.01, momentum=0.8, weight_decay=1e-4)
        # dec_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.0005, weight_decay=1e-5)
        
        epochs_dec = config['epochs_dec']
            
        train(
            dataset=dataset,
            model=model,
            epochs=epochs_dec,
            batch_size=batch_size,
            optimizer=dec_optimizer,
            scheduler= StepLR(dec_optimizer, 20, gamma=0.5),
            stopping_delta=0.000001,
            cuda=cuda,
            evaluate_batch_size=1024,
        )
        
        X, predicted = predict(
            dataset, model, 1024, silent=True, return_actual=False, cuda=cuda
        )
    
          
        score = silhouette_score(X, predicted) # X and Labels
        print(f"Silhouette Score: {score:.4f}")
        
        np.save(f'{output_predicted_dir}/predicted_{score:.4f}.npy', predicted)

        
        # salvando o modelo para testes de inferência
        if (save_dec):
            torch.save(model, os.path.join(config['saved_models_dir'], f"dec_model_{score:.4f}.pt"))
    
    writer.close()

if __name__ == "__main__":
    main()
