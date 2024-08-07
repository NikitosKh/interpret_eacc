import torch
import torch.nn as nn
from einops import einsum
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
import wandb
from torch.utils.data import Dataset, DataLoader 
from transformers import AutoTokenizer, AutoModelForCausalLM
import tqdm
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Autoencoder(nn.Module):
    def __init__(self, hidden_dim, d_model, cfg, max_len=1024):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.contex_length = max_len

        self.W_encode = nn.Parameter(torch.randn(d_model, hidden_dim))
        self.W_decode = nn.Parameter(torch.randn(hidden_dim, d_model))

    def forward(self, x):
        latent_vector = einsum(x, self.W_encode, 'b c d, d h -> b c h')
        loss_l1 = torch.norm(latent_vector, p=1)/latent_vector.flatten().shape[0]

        restored = einsum(latent_vector, self.W_decode, 'b c h, h d -> b c d')
        loss_l2 = torch.norm(restored - x, p=2)/ restored.flatten().shape[0]

        return latent_vector, loss_l1, loss_l2

class AutoencoderMerged(nn.Module):
    def __init__(self, models, cfg_enc, device, max_len=1024, layer=3):
        super().__init__()
        self.models = models
        self.cfg = cfg_enc
        self.device = device
        self.autoencoders = nn.ModuleList()

        for res_dim, model in zip(self.cfg.d_models, self.models):
            model = model.to(self.device)
            hidden_dim = res_dim * cfg_enc.multiplier
            autoencoder = Autoencoder(hidden_dim, res_dim, cfg_enc, max_len).to(self.device)
            self.autoencoders.append(autoencoder)

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, x):
        x = x.to(self.device)

        model_outputs = []
        with torch.no_grad():
            for model in self.models:
                output = model(x, output_hidden_states=True).hidden_states[-3]
                model_outputs.append(output)

        losses = []
        vs = []
        for i, (x, autoencoder) in enumerate(zip(model_outputs, self.autoencoders)):
            v, l1, l2 = autoencoder(x)
            losses.append(l1 + l2)
            vs.append(v.view(-1, self.cfg.multiplier * self.cfg.d_models[i]))

        for i in range(1, len(self.models)):
            losses[i] -= self.cos(vs[i], vs[0]).sum(dim=0)

        return losses

class Config:
    def __init__(self):
        self.multiplier = 1
        self.d_models = []
if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")
    model_name1 = "Sharathhebbar24/math_gpt2_sft"
    tokenizer1 = AutoTokenizer.from_pretrained(model_name1)
    model1 = AutoModelForCausalLM.from_pretrained(model_name1).to(device)

    model_name2 = "yoavgur/gpt2-bash-history-baseline"
    tokenizer2 = AutoTokenizer.from_pretrained(model_name2)
    model2 = AutoModelForCausalLM.from_pretrained(model_name2).to(device)
    cfg = Config()
    cfg.multiplier = 2
    cfg.d_models = [768, 768]
    merged = AutoencoderMerged([model1, model2], cfg, device).to(device)
    param_count = sum(p.numel() for p in merged.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {param_count}")

    for name, param in merged.named_parameters():
        print(f"{name}: {param.shape}")

   
    test_input = tokenizer1(["I am a test input", "This is another test"], return_tensors='pt', padding=True)['input_ids'].to(device)
    losses = merged(test_input)
    print("Losses:", losses)

    
    losses = [loss.cpu() for loss in losses]
class TextDataset(Dataset):
  
  def __init__(self, file_path, tokenizer, max_len):
    super(TextDataset, self).__init__()
    self.tokenizer = tokenizer
    self.max_len = max_len 
    with open(file_path, 'r', encoding = 'utf-8') as file:
      self.lines = file.readlines()
    self.lines = [line.strip() for line in self.lines if line.strip()]
  def __len__(self):
    return len(self.lines)
  def __getitem__(self, idx) :
    line = self.lines[idx].strip()
    encoding = self.tokenizer(line, truncation = True, max_length = self.max_len, padding = 'max_length', return_tensors = 'pt')
    return encoding['input_ids'].squeeze()
def save_model(model, path):
  torch.save(model.state_dict(), path)
def load_model(model, path):
  model.load_state_dict(torch.load(path))
  return model 


def train(model, dataloader, optimizer, device, epochs):
  model.train()
  for epoch in range(epochs):
    total_loss = 0 
    for batch in dataloader:
      optimizer.zero_grad()
      loss = sum(model(batch.to(device)))
      print(f"Loss:{loss}")
      
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
    average_loss = total_loss / len(dataloader)
    save_model(model, '/Users/nikitakhomich/project/Lie_algebras-interpretability')

    print(f"Epoch {epoch + 1}/{epoch}, Average loss : {average_loss}")
  return model 

dataset = TextDataset('/Users/nikitakhomich/Downloads/t8.shakespeare.txt', tokenizer1, max_len  = 512)
dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)
optimizer = torch.optim.Adam(merged.parameters(), lr = 3 * 10 ** -4)
wandb_config = {
        "learning_rate": 3e-4,
        "epochs": 10,
        "batch_size": 32,
        "model_architecture": "AutoencoderMerged",
        "dataset": "Shakespeare"
    }


wandb.init(project="model_merging", config=wandb_config)


num_epochs = 10
trained_model = train(merged, dataloader, optimizer, device, num_epochs)


wandb.finish()