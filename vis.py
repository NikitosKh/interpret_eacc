import torch
from torch.utils.data import Dataset, DataLoader
from merge import MergedModel
import numpy as np

def get_token_autoencoder_representation(token, merged_model, dataset, device):
    #  token: int, token to investigate
    #  merged_model: MergedModel, model to use for representation
    #  dataset: Dataloader, dataset to avergae the representation over
    #  returns: (List[torch.Tensor], torch.Tensor), (average representation of the token for each model merging,
    #  average representation of the token for the merged model (essentialy the average over the first output))
    res=([], None)
    num_of_appearances_so_far=0
    for batch in dataset:
        with torch.no_grad():
            input_ids = batch.to(device)
            if (input_ids == token).any():
                indices = (input_ids == token).nonzero(as_tuple=True)
                indices_tuples = list(zip(*indices))
                output = merged_model(batch, return_embeddings=True)[1]
                output = [sum([output[0][j][index] for index in indices_tuples])/len(indices_tuples) for j in range(len(merged_model.models)), sum([output[1][index] for index in indices_tuples])/len(indices_tuples)]
                res[0]=(res[0]*num_of_appearances_so_far + output[0]/num_of_appearances_so_far+len(indices_tuples))
                res[1]=(res[1]*num_of_appearances_so_far + output[1]/num_of_appearances_so_far+len(indices_tuples))
                num_of_appearances_so_far+=len(indices_tuples)
                
    return res            
                
                 
