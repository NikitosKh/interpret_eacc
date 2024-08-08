import torch
import torch.nn as nn
from einops import einsum
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
import wandb
from torch.utils.data import Dataset, DataLoader 
from transformers import AutoTokenizer, AutoModelForCausalLM
import tqdm


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
        latent_vector = einsum(x, self.W_encode, 'b c d, d h -> b c h')
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
                output = model(x, output_hidden_states=True).hidden_states[-3]
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


# training
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


def train(model, dataloader, optimizer, cfg):
  model.train()
  for epoch in range(cfg.num_train_epochs):
    total_loss = 0 
    for i, batch in enumerate(dataloader):
      optimizer.zero_grad()
      loss, (l1, l2, cos) = model(batch.to(device))

      wandb.log({'Training Loss': loss}, step=i)
      wandb.log({'l1 Loss': l1}, step=i)
      wandb.log({'l2 Loss': l2}, step=i)
      wandb.log({'cos_loss Loss': cos}, step=i)

      loss.backward()

      optimizer.step()
      total_loss += loss.item()
    average_loss = total_loss / len(dataloader)
    save_model(model, cfg.output_dir)

    print(f"Epoch {epoch + 1}/{epoch}, Average loss : {average_loss}")
  return model 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from transformers import TrainingArguments

class CustomTrainingArguments(TrainingArguments):
    def __init__(self, 
                 *args, 
                 multiplier=1, 
                 d_models=[], 
                 lambda_l1=1, 
                 lambda_cos=2, 
                 **kwargs):
      
        super().__init__(*args, **kwargs)
        self.multiplier=2                    # latent_dim=res_dim * d_model
        self.d_models=[768, 768]
        self.lambda_l1=lambda_l1
        self.lambda_cos=lambda_cos

training_cfg = CustomTrainingArguments(
    output_dir='./results',          # output directory
    learning_rate=3e-4,              # learning rate
    multiplier=2,                    # katent_dim=res_dim * d_model
    d_models=[768, 768],             # no comments      
    per_device_train_batch_size=32,  # batch size for training
    lambda_l1=1,
    lambda_cos=2,
    # per_device_eval_batch_size=16,    # batch size for evaluation
    num_train_epochs=1,              # total number of training epochs
    # commented but to be done
    # weight_decay=0.01,               # strength of weight decay
)

print(f"Using device: {device}")

model_name1 = "Sharathhebbar24/math_gpt2_sft"
tokenizer1 = AutoTokenizer.from_pretrained(model_name1)
model1 = AutoModelForCausalLM.from_pretrained(model_name1).to(device)

model_name2 = "yoavgur/gpt2-bash-history-baseline"
tokenizer2 = AutoTokenizer.from_pretrained(model_name2)
model2 = AutoModelForCausalLM.from_pretrained(model_name2).to(device)


merged = AutoencoderMerged([model1, model2], training_cfg, device).to(device)
param_count = sum(p.numel() for p in merged.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {param_count}")

for name, param in merged.named_parameters():
    print(f"{name}: {param.shape}")


test_input = tokenizer1(["I am a test input", "This is another test"], return_tensors='pt', padding=True)['input_ids'].to(device)
losses = merged(test_input)
print("Losses:", losses)


losses = [loss for loss in losses]        
dataset = TextDataset('shakespeare.txt', tokenizer1, max_len  = 512)
dataloader = DataLoader(dataset, batch_size = training_cfg.per_device_train_batch_size, shuffle = True)


optimizer = torch.optim.Adam(merged.parameters(), lr = 3 * 10 ** -4)
wandb_config = {
        "learning_rate": training_cfg.learning_rate,
        "epochs": training_cfg.num_train_epochs,
        "batch_size": training_cfg.per_device_train_batch_size,
        "model_architecture": "AutoencoderMerged",
        "dataset": "Shakespeare"
    }


wandb.init(project="model_merging", config=wandb_config)


trained_model = train(merged, dataloader, optimizer, training_cfg)


wandb.finish()