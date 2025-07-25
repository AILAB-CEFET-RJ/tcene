import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader, default_collate
from typing import Any, Tuple, Callable, Optional, Union
from tqdm import tqdm

from ptdec.utils import target_distribution, cluster_accuracy


def train(
    dataset: torch.utils.data.Dataset,
    model: torch.nn.Module, # o modelo que transforma lotes de samples em seus respectivos cluster assignments (possui também o encoder)
    epochs: int,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    scheduler: Any = None,
    stopping_delta: Optional[float] = None,
    collate_fn=default_collate,
    cuda: bool = True,
    sampler: Optional[torch.utils.data.sampler.Sampler] = None,
    silent: bool = False,
    update_freq: int = 10,
    evaluate_batch_size: int = 1024,
    update_callback: Optional[Callable[[float, float], None]] = None,
    epoch_callback: Optional[Callable[[int, torch.nn.Module], None]] = None,
    
) -> None:
    """
    Train the DEC model given a dataset, a model instance and various configuration parameters.

    :param dataset: instance of Dataset to use for training
    :param model: instance of DEC model to train
    :param epochs: number of training epochs
    :param batch_size: size of the batch to train with
    :param optimizer: instance of optimizer to use
    :param stopping_delta: label delta as a proportion to use for stopping, None to disable, default None
    :param collate_fn: function to merge a list of samples into mini-batch
    :param cuda: whether to use CUDA, defaults to True
    :param sampler: optional sampler to use in the DataLoader, defaults to None
    :param silent: set to True to prevent printing out summary statistics, defaults to False
    :param update_freq: frequency of batches with which to update counter, None disables, default 10
    :param evaluate_batch_size: batch size for evaluation stage, default 1024
    :param update_callback: optional function of accuracy and loss to update, default None
    :param epoch_callback: optional function of epoch and model, default None
    :return: None
    """
    static_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=False,
        sampler=sampler,
        shuffle=False, # os lotes serão na mesma ordem que no original
    )
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        sampler=sampler,
        shuffle=True,
    )
    current_lr = optimizer.param_groups[0]["lr"]
    data_iterator = tqdm(
        static_dataloader,
        leave=True,
        unit="batch",
        postfix={
            "epo": -1,
            "acc": "%.4f" % 0.0,
            "lss": "%.8f" % 0.0,
            "dlb": "%.4f" % -1,
            "lr": current_lr,
        },
        disable=silent,
    )
    
    kmeans = KMeans(n_clusters=model.cluster_number, n_init=20, random_state=42)
    # kmeans = MiniBatchKMeans(n_clusters=model.cluster_number, batch_size=1024, random_state=42) 
    model.train() # setting the model into training mode.
    features = []
    actual = []
    
    
    # form initial cluster centres
    for index, batch in enumerate(data_iterator):
        if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
            batch, value = batch  # if we have a prediction label, separate it to actual
            actual.append(value) # "y_train"
        if cuda:
            batch = batch.cuda(non_blocking=True)
        features.append(model.encoder(batch).detach().cpu())
        

    if len(actual) > 0: # se existir ground truth labels...
        actual = torch.cat(actual).long()
        print("\n\n\n\Actuals sendo adicionados\n\n\n")
    else:
        actual = torch.empty(0, dtype=torch.long)
        
    
    
    accuracy = 0 # target = target_distribution(predicted).detach()
    #_, accuracy = cluster_accuracy(predicted, actual.cpu().numpy()) # os labels são usados aqui
    predicted = kmeans.fit_predict(torch.cat(features).numpy())
    predicted_previous = torch.tensor(np.copy(predicted), dtype=torch.long)
    cluster_centers = torch.tensor(
        kmeans.cluster_centers_, dtype=torch.float, requires_grad=True
    )
    if cuda:
        cluster_centers = cluster_centers.cuda(non_blocking=True)
    with torch.no_grad():
        # initialise the cluster centers
        model.state_dict()["assignment.cluster_centers"].copy_(cluster_centers)
        
        
    loss_function = nn.KLDivLoss(reduction='sum')
    delta_label = None
    for epoch in range(epochs):
        features = []
        data_iterator = tqdm(
            train_dataloader,
            leave=True,
            unit="batch",
            postfix={
                "epo": epoch,
                "acc": "%.4f" % (accuracy or 0.0),
                "lss": "%.8f" % 0.0,
                "dlb": "%.4f" % (delta_label or 0.0),
                "lr": current_lr,
            },
            disable=silent,
        )
        model.train() # ensure the model is in the correct mode before train loop
        # TRAIN LOOP
        for index, batch in enumerate(data_iterator):
            if (isinstance(batch, tuple) or isinstance(batch, list)) and len(
                batch
            ) == 2:
                batch, _ = batch  # if we have a prediction label, strip it away
            if cuda:
                batch = batch.cuda(non_blocking=True)
            output = model(batch) 
            # model é o Encoder +  (assignment): ClusterAssignment()   
            # output será Q
            # tem shape [len(batch), num_clusters]
            # exemplo: torch.Size([230, 10])  

            
            # Target Distribution P (possui mesmo shape de Q)
            target = target_distribution(output).detach() #  Compute the target distribution p_ij, given the batch (q_ij)
            
            pred_labels = output.argmax(dim=1) # possui len() de len(batch)
            pred_target = target.argmax(dim=1) 
            pred_labels_np = pred_labels.detach().cpu().numpy()  
            true_labels_np = pred_target.detach().cpu().numpy()  
            _, accuracy = cluster_accuracy(pred_labels_np, true_labels_np) # not a good measure
            
            # KL Loss
            loss = loss_function(output.log(), target) / output.shape[0]
            data_iterator.set_postfix(
                epo=epoch,
                acc="%.4f" % (accuracy or 0.0),
                lss="%.8f" % float(loss.item()),
                dlb="%.4f" % (delta_label or 0.0),
                lr= "%.6f" % current_lr,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(closure=None)
            
            
                
            features.append(model.encoder(batch).detach().cpu())
            
            
            if update_freq is not None and index % update_freq == 0:
                loss_value = float(loss.item())
                
                data_iterator.set_postfix(
                    epo=epoch,
                    acc="%.4f" % (accuracy or 0.0),
                    lss="%.8f" % loss_value,
                    dlb="%.4f" % (delta_label or 0.0),
                    lr= "%.6f" % current_lr,
                )
                if update_callback is not None:
                    update_callback(accuracy, loss_value, delta_label)
                
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]     
            
            
        if(len(actual)>0):
            pred = predict(
                dataset,
                model,
                batch_size=evaluate_batch_size,
                collate_fn=collate_fn,
                silent=True,
                return_actual=False, # precisa retornar False já que Empenhos não tem actuals
                cuda=cuda,
            ) # retorna uma tupla (predicted) e um tensor (actual)

                
            delta_label = (
                float((pred != predicted_previous).float().sum().item())
                / predicted_previous.shape[0]
            )
            print("\n\nDelta Label: ", delta_label)
            print("Stopping Delta: ", stopping_delta)
            if stopping_delta is not None and delta_label < stopping_delta:
                print(
                    'Early stopping as label delta "%1.5f" less than "%1.5f".'
                    % (delta_label, stopping_delta)
                )
                break
            predicted_previous = pred
            
            _, accuracy = cluster_accuracy(pred.cpu().numpy(), actual.cpu().numpy())
            data_iterator.set_postfix(
                epo=epoch,
                acc="%.4f" % (accuracy or 0.0),
                lss="%.8f" % 0.0,
                dlb="%.4f" % (delta_label or 0.0),
                lr= "%.6f" % optimizer.param_groups[0]["lr"],
            )
        if epoch_callback is not None:
            epoch_callback(epoch, model)


def predict(
    dataset: torch.utils.data.Dataset,
    model: torch.nn.Module,
    batch_size: int = 1024,
    collate_fn=default_collate,
    cuda: bool = True,
    silent: bool = False,
    return_actual: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Predict clusters for a dataset given a DEC model instance and various configuration parameters.

    :param dataset: instance of Dataset to use for training
    :param model: instance of DEC model to predict
    :param batch_size: size of the batch to predict with, default 1024
    :param collate_fn: function to merge a list of samples into mini-batch
    :param cuda: whether CUDA is used, defaults to True
    :param silent: set to True to prevent printing out summary statistics, defaults to False
    :param return_actual: return actual values, if present in the Dataset
    :return: tuple of prediction and actual if return_actual is True otherwise prediction
    """
    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False
    )
    data_iterator = tqdm(dataloader, leave=True, unit="batch", disable=silent,)
    features = []
    actual = []
    model.eval() # colocar o modelo em formato de evaluation
    for batch in data_iterator:
        if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
            batch, value = batch  # unpack if we have a prediction label
            if return_actual:
                actual.append(value)
        elif return_actual:
            raise ValueError(
                "Dataset has no actual value to unpack, but return_actual is set."
            )
        if cuda:
            batch = batch.cuda(non_blocking=True)
        features.append(
            model(batch).detach().cpu()
        )  # move to the CPU to prevent out of memory on the GPU
    if return_actual:
        return torch.cat(features).max(1)[1], torch.cat(actual).long()
    else: # retorna os vetores X e os labels correspondentes
        return torch.cat(features), torch.cat(features).max(1)[1]
