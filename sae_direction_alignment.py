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
    def __init__(self, hidden_dim, d_model, max_len=1024):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.contex_length = max_len

        self.W_encode = nn.Parameter(torch.zeros(d_model, hidden_dim))
        self.bias_encode = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.xavier_uniform_(self.W_encode)
        self.W_decode = nn.Parameter(torch.zeros(hidden_dim, d_model))
        self.bias_decode = nn.Parameter(torch.zeros(d_model))
        nn.init.xavier_uniform_(self.W_decode)
        
        self.l2 = nn.MSELoss(reduction='mean')
        
    def encode(self, x):
        return nn.ReLU()(einsum(x, self.W_encode, 'b c d, d h -> b c h') + self.bias_encode) 
      
    def decode(self, x):
        return einsum(x, self.W_decode, 'b c h, h d -> b c d') + self.bias_decode    

    def forward(self, x):
        latent_vector = self.encode(x)
        loss_l1 = torch.abs(latent_vector).mean()

        restored = self.decode(latent_vector)
        loss_l2 = self.l2(restored, x)

        return (latent_vector, loss_l1, loss_l2)

#tokenizers must be the same at the moment
class AutoencoderMerged(nn.Module):
    def __init__(self, 
                 models, 
                 cfg, 
                 device, 
                 max_len=1024, 
                 layer=-2):
        super().__init__()
        self.models = models
        self.cfg = cfg
        self.device = device
        self.autoencoders = nn.ModuleList()
        self.layer=layer

        for res_dim, model in zip(self.cfg.d_models, self.models):
            model = model.to(self.device)
            hidden_dim = res_dim * cfg.multiplier
            autoencoder = Autoencoder(hidden_dim, res_dim, max_len).to(self.device)
            self.autoencoders.append(autoencoder)

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, x):
        x = x.to(self.device)

        model_outputs = []
        with torch.no_grad():
            for model in self.models:
                output = model(x, output_hidden_states=True).hidden_states[self.layer]
                model_outputs.append(output)

        losses = []
        vs = []
        l1_log = []
        l2_log = []
        for i, (x, autoencoder) in enumerate(zip(model_outputs, self.autoencoders)):
            v, l1, l2 = autoencoder(x)
            l1_log.append(l1)
            l2_log.append(l2)
            losses.append(self.cfg.lambda_l1*l1+l2)
            vs.append(v.view(-1, self.cfg.multiplier * self.cfg.d_models[i]))

        cos_loss=[]
        for i in range(1, len(self.models)):
            cos_loss.append((1-self.cos(vs[i], vs[0])).mean(dim=0))
            losses[i] += self.cfg.lambda_cos*cos_loss[-1]

        cos_loss=sum(cos_loss)/len(cos_loss)
        l1_log=sum(l1_log)/len(l1_log)
        l2_log=sum(l2_log)/len(l2_log)
        losses=sum(losses)/len(losses)

        return losses, (l1_log, l2_log, cos_loss)


    def get_transitions(self):
        class Transition(nn.Module):
            def __init__(self, endoder, decoder):
                super(Transition, self).__init__()

                self.encoder = endoder
                self.relu=nn.ReLU()
                self.decoder = decoder

            def forward(self, x):
                latent_vector = nn.ReLU()(einsum(x, self.encoder, 'b c d, d h -> b c h'))

                restored = einsum(latent_vector, self.decoder, 'b c h, h d -> b c d')
                return restored

        transitions=[[None]*len(self.models) for _ in range(len(self.models))]

        for i in range(1, len(self.models)):
            transitions[i][0] = Transition(self.autoencoders[i].W_encode,
                                                           self.autoencoders[0].W_decode)

        return transitions

    def get_encoders(self):
        class Encoder(nn.Module):
            def __init__(self, endoder):
                super(Encoder, self).__init__()

                self.encoder = endoder
                self.relu=nn.ReLU()

            def forward(self, x):
                encoding = nn.ReLU()(einsum(x, self.encoder, 'b c d, d h -> b c h'))
                
                return encoding
        
        encoders=[None]*len(self.models)

        for i in range(0, len(self.models)):
            encoders[i] = Encoder(self.autoencoders[i].W_encode)

        return encoders
      
    def get_decoders_to_0(self):
        class Transition(nn.Module):
            def __init__(self, decoder):
                super(Transition, self).__init__()
                self.decoder = decoder

            def forward(self, x):
                decoding = einsum(x, self.decoder, 'b c h, h d -> b c d')
                return decoding
        
        decoders=[None]*len(self.models)

        for i in range(1, len(self.models)):
            decoders[i] = Transition(self.autoencoders[0].W_decode)

        return decoders  

