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

def train(model, train_dataloader, device, layer, optimizer, cfg):
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

        average_loss = total_loss / len(train_dataloader)

        print(f"Epoch {epoch + 1}/{epoch}, Average loss : {average_loss}")
        # Save model checkpoint
        save_model(model, cfg.output_dir, epoch, layer, average_loss)

    return model

def save_model(model, output_dir, epoch, layer, loss):
    model_save_path = f"{output_dir}/model_epoch_{epoch+1}_layer_{layer}_loss_{loss:.4f}.pt"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

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


import yaml

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    with open('./training_sae.yaml', 'r') as f:
        config = yaml.safe_load(f)
        f.close()
        
    training_cfg = CustomTrainingArguments(
        output_dir=config['output_dir'],        
        learning_rate=config['learning_rate'],             
        multiplier=config['multiplier'],                
        d_models=config['d_models'],  
        per_device_train_batch_size=config['batch_size'],
        lambda_l1=config['lambda_l1'],
        lambda_cos=config['lambda_cos'],
        # per_device_eval_batch_size=16,  
        num_train_epochs=config['num_train_epochs'],       
        # commented but to be done
        # weight_decay=0.01,               # strength of weight decay
    )    
        
    model_name1 = "Sharathhebbar24/math_gpt2_sft"
    tokenizer1 = AutoTokenizer.from_pretrained(model_name1)
    model1 = AutoModelForCausalLM.from_pretrained(model_name1).to(device)

    model_name2 = "yoavgur/gpt2-bash-history-baseline"
    tokenizer2 = AutoTokenizer.from_pretrained(model_name2)
    model2 = AutoModelForCausalLM.from_pretrained(model_name2).to(device)


    merged = AutoencoderMerged([model1, model2], 
                               training_cfg, 
                               layer=config['layer'], 
                               device=device).to(device)
    param_count = sum(p.numel() for p in merged.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {param_count}")

    for name, param in merged.named_parameters():
        print(f"{name}: {param.shape}")


    test_input = tokenizer1(["I am a test input", "This is another test"], return_tensors='pt', padding=True)['input_ids'].to(device)
    losses = merged(test_input)
    print("Losses:", losses)


    losses = [loss for loss in losses]
    dataset = TextDataset(config['train_dataset'], tokenizer1, max_len  = 512)
    dataloader = DataLoader(dataset, batch_size = training_cfg.per_device_train_batch_size, shuffle = True)


    optimizer = torch.optim.Adam(merged.parameters(), lr = training_cfg.learning_rate)
    wandb_config = {
            "learning_rate": training_cfg.learning_rate,
            "epochs": training_cfg.num_train_epochs,
            "batch_size": training_cfg.per_device_train_batch_size,
            "model_architecture": "AutoencoderMerged",
            "dataset": config['dataset_name']
        }


    wandb.init(project="model_merging", config=wandb_config)


    trained_model = train(merged, 
                          dataloader, 
                          device, 
                          layer=config['layer'], 
                          optimizer=optimizer, 
                          training_cfg=training_cfg)


    wandb.finish()
    
    return trained_model
