from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import numpy as np
from sae_direction_alignment import Autoencoder, AutoencoderMerged
import einops


class MergedModelArguments():
    def __init__(self, same_architecture=True):
        self.same_architecture=same_architecture
        
        
class MergedModel(nn.Module):
    def __init__(self, 
                models, 
                configs, 
                device,
                joint_autoencoders,
                ):  
      super(MergedModel, self).__init__()
      for model in models:
          model=model.to(device)
      self.models=nn.ModuleList(models)
      self.same_architecture = configs.same_architecture
      # self.set_trainable_parameters() 
      for autoencoder in joint_autoencoders:
          autoencoder=autoencoder.to(device)
      self.encoders = {joint_autoencoder.layer: joint_autoencoder.get_encoders() for joint_autoencoder in joint_autoencoders}
      self.decoders = {joint_autoencoder.layer: joint_autoencoder.get_decoders_to_0() for joint_autoencoder in joint_autoencoders}
      self.cfg=configs
      self.device=device

    
    def forward(self, input_ids, return_embeddings=False):        
        # input_ids: tensor of shape (batch_size, max_tokens)
        input_ids=input_ids.to(self.device)
        
        current_slice_of_modules = [None] * len(self.models)
        logits=[input_ids] * len(self.models)
        if return_embeddings:
            embeddings={i: [[], None] for i in self.encoders.keys()}
            
        logits=[input_ids] * len(self.models)
        prev_i=0
        for iter, i in enumerate(self.encoders.keys()):
            for j, model in enumerate(self.models):
              
                if iter == 0:
                    class Embedding(nn.Module):
                        def __init__(self, wte, wpe, drop):
                            super(Embedding, self).__init__()
                            self.wte = wte
                            self.wpe = wpe
                            self.drop = drop

                        def forward(self, x):
                            return (self.drop(self.wte(x) + self.wpe(torch.arange(x.size(1), device=x.device))),)

                    current_slice_of_modules[j] = [Embedding(model.transformer.wte, model.transformer.wpe, model.transformer.drop)] + list(model.transformer.h[prev_i:i+1])   
                else:
                    current_slice_of_modules[j] = model.transformer.h[prev_i:i+1]
                      
                prev_i=i    
                
                if j==0:
                    for module in current_slice_of_modules[j]:
                        logits[j]=module(logits[j])[0]
                else:
                    logits_temp=logits[j]
                    
                    for module in current_slice_of_modules[j]:
                        logits_temp=module(logits_temp)[0]
                        
                    logits[j]=logits_temp
                      
                    
                    logits[0]+=self.decoders[i][j](self.encoders[i][j](logits_temp))
                        
                if return_embeddings:
                        embeddings[i][0].append(self.encoders[i][j](logits[j]).unsqueeze(0))                      
                    
            logits[0]/=len(self.models)  
                    
        last_slice = list(self.models[0].transformer.h[prev_i:])
        
        for layer_module in last_slice:
            logits[0] = layer_module(logits[0])[0]

        logits[0]=self.models[0].lm_head(self.models[0].transformer.ln_f(logits[0]))                 
        
        if return_embeddings:
            embeddings={i: [torch.cat(embeddings[i][0], dim=0), (sum(embeddings[i][0])/len(embeddings[i][0])).squeeze(0)] for i in embeddings.keys()}
            return logits[0], embeddings
        else:
            return logits[0]  
      
    def generate(self, input_ids, max_tokens=256):
      # input_ids: tensor of shape (batch_size, max_tokens)
      current_slice_of_modules = [None] * len(self.models)
      generated_text=[]
      for k in range(max_tokens):
          logits=[input_ids] * len(self.models)
          prev_i=0
          for iter, i in enumerate(self.encoders.keys()):
              for j, model in enumerate(self.models):
                
                  if iter == 0:
                      class Embedding(nn.Module):
                          def __init__(self, wte, wpe, drop):
                              super(Embedding, self).__init__()
                              self.wte = wte
                              self.wpe = wpe
                              self.drop = drop

                          def forward(self, x):
                              return (self.drop(self.wte(x) + self.wpe(torch.arange(x.size(1), device=x.device))),)

                      current_slice_of_modules[j] = [Embedding(model.transformer.wte, model.transformer.wpe, model.transformer.drop)] + list(model.transformer.h[prev_i:i+1])   
                  else:
                      current_slice_of_modules[j] = model.transformer.h[prev_i:i+1]
                        
                  prev_i=i 
                  
                  if j==0:
                      for module in current_slice_of_modules[j]:
                          logits[j]=module(logits[j])[0]
                  else:
                      logits_temp=logits[j]
                      
                      for module in current_slice_of_modules[j]:
                          logits_temp=module(logits_temp)[0]
                          
                      logits[j]=logits_temp  
                      logits[0]+=self.decoders[i][j](self.encoders[i][j](logits_temp))
          
              logits[0]/=len(self.models)            

          last_slice = list(self.models[0].transformer.h[prev_i:])
          
          for layer_module in last_slice:
              logits[0] = layer_module(logits[0])[0]

          logits[0]=self.models[0].lm_head(self.models[0].transformer.ln_f(logits[0]))    
          input_ids=torch.cat([input_ids, logits[0][:, -1, :].argmax(dim=-1).unsqueeze(1)], dim=1)
          generated_text.append(logits[0][:, -1, :].argmax(dim=-1).unsqueeze(1))
          
      return torch.cat(generated_text, dim=1)  