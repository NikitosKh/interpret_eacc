from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import numpy as np


class MergedModel(nn.Module):
  def __init__(self, models, tokenizer, configs):
    super(MergedModel, self).__init__()
    self.models=nn.ModuleList(models)
    self.tokenizer=tokenizer
    self.same_architecture=configs['same_architecture']
    self.find_transition_matrices()
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.set_trainable_parameters(0)   

  def find_transition_matrices(self):
    n=len(self.models)
    self.W = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
      for j in range(i+1,n):
        embed_i=np.array(self.models[i].get_input_embeddings().weight.detach().cpu())
        embed_j=np.array(self.models[j].get_input_embeddings().weight.detach().cpu())

        print(f"fitting {i} to {j} started")
        self.W[i][j] = torch.tensor(np.linalg.lstsq(embed_i, embed_j, rcond=None)[0])
        print(f"fitting {i} to {j} done")
        self.W[j][i] = torch.tensor(np.linalg.lstsq(embed_j, embed_i, rcond=None)[0])
        print(f"fitting {j} to {i} done")

        # Verify the transformation
        transformed_embeddings = np.dot(embed_i, self.W[i][j].detach().numpy())
        error = np.mean((transformed_embeddings - embed_j) ** 2)
        print("Mean Squared Error:", error)

  def set_trainable_parameters(self, threshold=0.5):
    for model in self.models:
      for name, param in model.named_parameters():
        param.requires_grad = False

      for name, param in model.named_parameters():
        if any(f"{i}" in name for i in range(int(threshold*12), 13)) or ('0' in name):
          param.requires_grad = True
                          
  def forward(self, input_ids, attention_mask=None, labels=None, cutting_threshold=0.5, max_tokens=1):
    # output: generated tokens' labels of shape (max_tokens, vocab_size) probably 
    self.set_trainable_parameters(cutting_threshold)   
    second_half_layers = [model.transformer.h[int(cutting_threshold*len(model.transformer.h)):] for model in self.models]
    
    generated_tokens=[]
    for j in range(max_tokens):
      # run all the models separately and add the hidden states of the middle layer  
      for i in range(len(self.models)):
        if j != 0:
          input_ids += torch.cat(generated_tokens, 0).argmax(-1).view(j).tolist()
        with torch.no_grad():
          all_model_embeds = self.models[i](torch.tensor(input_ids), output_hidden_states=True).hidden_states
          model_embeds=all_model_embeds[int(cutting_threshold*len(self.models[i].transformer.h))]
        if i == 0:
          joint_residual = model_embeds
        else:
          joint_residual = model_embeds + torch.einsum('ij, bsj -> bsi', torch.tensor(self.W[i][0]), model_embeds)

      # the exact architecture is important because modules have different names(not only that its the same)!
      if self.same_architecture:
        # run the rest layers of our models on the joint residual and add the results
        half_model_0_output = joint_residual
        for i, layer_module in enumerate(second_half_layers[0]):
          half_model_0_output = layer_module(half_model_0_output)[0]
        logits = half_model_0_output

        for i in range(1, len(self.models)):
          half_model_i_output = joint_residual
          for layer_module in second_half_layers[i]:
            half_model_i_output = layer_module(half_model_i_output)[0]
          logits += half_model_i_output
        logits = self.models[0].lm_head(self.models[0].transformer.ln_f(logits.squeeze(0)[-1]))

      generated_tokens.append(logits.unsqueeze(0))

    return generated_tokens
  

model_name1 = "Sharathhebbar24/math_gpt2_sft"
tokenizer1 = AutoTokenizer.from_pretrained(model_name1)
model1 = AutoModelForCausalLM.from_pretrained(model_name1)


model_name2 = "yoavgur/gpt2-bash-history-baseline"
tokenizer2 = AutoTokenizer.from_pretrained(model_name2)
model2 = AutoModelForCausalLM.from_pretrained(model_name2)  

configs={'same_architecture': True}
mod=MergedModel([model1, model2], tokenizer1, configs)

print(mod(tokenizer1("I am stupid", return_tensors='pt', padding=True)['input_ids'].squeeze().tolist()))

print(tokenizer1.decode(torch.cat(mod(tokenizer1("I am stupid", return_tensors='pt', padding=True)['input_ids'].squeeze().tolist()), 0).argmax(-1)))