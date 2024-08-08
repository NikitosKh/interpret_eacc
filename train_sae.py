import torch
import torch.nn as nn
from einops import einsum
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
import wandb
import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from sae_direction_alignment import Autoencoder, AutoencoderMerged



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


def load_model(model, path):
  model.load_state_dict(torch.load(path), map_location=torch.device('cpu'))
  return model

class EarlyStopping:
    def __init__(self,
                 patience=3,
                 min_delta=0,
                 scheduler=None):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.epochs_no_improve = 0
        self.early_stop = False
        self.scheduler=scheduler

    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss > self.best_loss - self.min_delta:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience/2:
                self.scheduler.step(loss)
            if self.epochs_no_improve >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.epochs_no_improve = 0

def train(model, train_dataloader, optimizer, cfg):
    model.to(device)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)
    early_stopping = EarlyStopping(patience=50, min_delta=0.001, scheduler=scheduler)

    for epoch in range(cfg.num_train_epochs):
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss, (l1, l2, cos) = model(batch.to(device))

            wandb.log({'Training Loss': loss}, step=i)
            wandb.log({'l1 Loss': l1}, step=i)
            wandb.log({'l2 Loss': l2}, step=i)
            wandb.log({'cos_loss Loss': cos}, step=i)

            loss.backward()
            optimizer.step()
            total_loss += loss
            early_stopping(loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        average_loss = total_loss / len(dataloader)

        print(f"Epoch {epoch + 1}/{epoch}, Average loss : {average_loss}")
        # Save model checkpoint
        save_model(model, cfg.output_dir, epoch, average_loss)

    return model

def save_model(model, output_dir, epoch, loss):
    model_save_path = f"{output_dir}/model_epoch_{epoch+1}_loss_{loss:.4f}.pt"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


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
        self.multiplier=multiplier   # latent_dim=res_dim * d_model
        self.d_models=d_models
        self.lambda_l1=lambda_l1
        self.lambda_cos=lambda_cos

training_cfg = CustomTrainingArguments(
    output_dir='./results',          # output directory
    learning_rate=3e-4,              # learning rate
    multiplier=70,                    # katent_dim=res_dim * d_model
    d_models=[768, 768],             # no comments
    per_device_train_batch_size=32,  # batch size for training
    lambda_l1=0.5,
    lambda_cos=0.5,
    # per_device_eval_batch_size=16,    # batch size for evaluation
    num_train_epochs=3,              # total number of training epochs
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


optimizer = torch.optim.Adam(merged.parameters(), lr = training_cfg.learning_rate)
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


def load_model(model, path):
  model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
  return model

model4=load_model(model=merged,
                 path="./results/model_epoch_2_loss_0.0000.pt")


import einops
import copy

test_input = tokenizer1(["That beauty still may live in thine or thee.", "This is another test"], return_tensors='pt', padding=True)['input_ids'].to(device)

input_gpt2 = tokenizer1(["That beauty still may live in thine or thee.", "This is another test"], return_tensors='pt', padding=True)['input_ids'].to(device)

for i in range(30):
    output = model1(test_input, output_hidden_states=True).hidden_states[-6]
    output2 = model2(test_input, output_hidden_states=True).hidden_states[-6]
    second_half_layers = model1.transformer.h[-6:]
    print(model4.autoencoders[0].W_encode.shape)
    t=nn.ReLU()(einsum(model4.autoencoders[0].W_encode, output, "hui latdim, bsz sqln hui -> bsz sqln latdim"))
    print(t.shape, model4.autoencoders[0].W_decode.shape)
    t=einsum(model4.autoencoders[1].W_decode, t, "latdim hui, bsz sqln latdim -> bsz sqln hui")
    print("dsfsdfsdf", nn.MSELoss(reduction='mean')(t, output2))
    t+=output2

    half_model_0_output = t
    for i, layer_module in enumerate(second_half_layers):
        half_model_0_output = layer_module(half_model_0_output)[0]
        
    res=model1.lm_head(model1.transformer.ln_f(half_model_0_output.squeeze(0))) 
    test_input=torch.cat([test_input, res[:, -1, :].argmax(dim=-1).unsqueeze(1)], dim=1)
    input_gpt2=torch.cat([input_gpt2, model2(input_gpt2).logits[:, -1, :].argmax(dim=-1).unsqueeze(1)], dim=1)
    print(test_input, input_gpt2)


print(tokenizer1.decode(test_input[0]))
print(tokenizer1.decode(input_gpt2[0]))
print(tokenizer1.decode(test_input[1]))
print(tokenizer1.decode(input_gpt2[1]))      

      