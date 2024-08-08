import torch
import torch.nn as nn
from einops import einsum
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
import wandb
import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os


class Autoencoder(nn.Module):
    def __init__(self, hidden_dim, d_model, cfg, max_len=1024):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.contex_length = max_len

        self.W_encode = nn.Parameter(torch.zeros(d_model, hidden_dim))
        nn.init.xavier_uniform_(self.W_encode)
        self.W_decode = nn.Parameter(torch.zeros(hidden_dim, d_model))
        nn.init.xavier_uniform_(self.W_decode)

        self.l2 = nn.MSELoss(reduction='mean')
        self.cfg = cfg

    def forward(self, x):
        latent_vector = nn.ReLU()(einsum(x, self.W_encode, 'b c d, d h -> b c h'))
        loss_l1 = self.cfg.lambda_l1*torch.abs(latent_vector).mean()

        restored = einsum(latent_vector, self.W_decode, 'b c h, h d -> b c d')
        loss_l2 = self.l2(restored, x)

        return (latent_vector, loss_l1, loss_l2)

class AutoencoderMerged(nn.Module):
    def __init__(self, models, cfg, device, max_len=1024, layer=3):
        super().__init__()
        self.models = models
        self.cfg = cfg
        self.device = device
        self.autoencoders = nn.ModuleList()

        for res_dim, model in zip(self.cfg.d_models, self.models):
            model = model.to(self.device)
            hidden_dim = res_dim * cfg.multiplier
            autoencoder = Autoencoder(hidden_dim, res_dim, self.cfg, max_len).to(self.device)
            self.autoencoders.append(autoencoder)

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, x):
        x = x.to(self.device)

        model_outputs = []
        with torch.no_grad():
            for model in self.models:
                output = model(x, output_hidden_states=True).hidden_states[-6]
                model_outputs.append(output)

        losses = []
        vs = []
        l1_log = []
        l2_log = []
        for i, (x, autoencoder) in enumerate(zip(model_outputs, self.autoencoders)):
            v, l1, l2 = autoencoder(x)
            l1_log.append(l1)
            l2_log.append(l2)
            losses.append(l1+l2)
            vs.append(v.view(-1, self.cfg.multiplier * self.cfg.d_models[i]))

        cos_loss=[]
        for i in range(1, len(self.models)):
            cos_loss.append(self.cfg.lambda_cos*(1-self.cos(vs[i], vs[0])).mean(dim=0))
            losses[i] += cos_loss[-1]

        cos_loss=sum(cos_loss)/len(cos_loss)
        l1_log=sum(l1_log)/len(l1_log)
        l2_log=sum(l2_log)/len(l2_log)
        losses=sum(losses)/len(losses)

        return losses, (l1_log, l2_log, cos_loss)


    def get_transition_matrices(self):
        transition_matrices=[[]*len(self.models) for _ in range(len(self.models))]

        for i in range(len(self.models)):
            for j in range(len(self.models)):
                if i != j:
                    transition_matrices[i][j] = self.autoencoders[j].W_decode @ self.autoencoders[i].W_encode
                    transition_matrices[j][i] = self.autoencoders[i].W_decode @ self.autoencoders[j].W_encode
                else:
                    transition_matrices[i][j] = torch.eye(self.autoencoders[i].hidden_dim)

        return transition_matrices

