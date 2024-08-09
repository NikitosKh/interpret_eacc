from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import numpy as np
from sae_direction_alignment import Autoencoder, AutoencoderMerged
import einops


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
      self.transitions = {joint_autoencoder.layer: joint_autoencoder.get_transitions() for joint_autoencoder in joint_autoencoders}
      self.cfg=configs
      self.device=device

    
    def forward(self, input_ids):
        input_ids=input_ids.to(self.device)
        # input_ids: tensor of shape (batch_size, max_tokens)
        current_slice_of_modules = [None] * len(self.models)
        logits=[input_ids] * len(self.models)
        ouptut=[]
        for k in range(input_ids.shape[-1]):
            inp=input_ids[:, :k+1]
            logits=[inp] * len(self.models)
            prev_i=0
            for iter, (i, transitions) in enumerate(self.transitions.items()):
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
                        logits[0]+=transitions[j][0](logits_temp) 

            last_slice = list(self.models[0].transformer.h[prev_i:])
            
            for layer_module in last_slice:
                logits[0] = layer_module(logits[0])[0]

            logits[0]=self.models[0].lm_head(self.models[0].transformer.ln_f(logits[0]))    

                  
            ouptut.append(logits[0][:, -1, :].argmax(dim=-1).unsqueeze(1))      
            
        return torch.cat(ouptut, dim=1) 
      
    def generate(self, input_ids, max_tokens=256):
      # input_ids: tensor of shape (batch_size, max_tokens)
      current_slice_of_modules = [None] * len(self.models)
      generated_text=[]
      for k in range(max_tokens):
          print(k)
          logits=[input_ids] * len(self.models)
          prev_i=0
          for iter, (i, transitions) in enumerate(self.transitions.items()):
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
                      logits[0]+=transitions[j][0](logits_temp) 

          last_slice = list(self.models[0].transformer.h[prev_i:])
          
          for layer_module in last_slice:
              logits[0] = layer_module(logits[0])[0]

          logits[0]=self.models[0].lm_head(self.models[0].transformer.ln_f(logits[0]))    

                  
          input_ids=torch.cat([input_ids, logits[0][:, -1, :].argmax(dim=-1).unsqueeze(1)], dim=1)
          generated_text.append(logits[0][:, -1, :].argmax(dim=-1).unsqueeze(1))
          
      return torch.cat(generated_text, dim=1)  
