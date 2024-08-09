import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from tqdm import tqdm
import os
from sae_direction_alignment import Autoencoder, AutoencoderMerged
from merge import MergedModel, MergedModelArguments


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len):
        super(TextDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        with open(file_path, 'r', encoding='utf-8') as file:
            self.lines = file.readlines()
        self.lines = [line.strip() for line in self.lines if line.strip()]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip()
        encoding = self.tokenizer(line, truncation=True, max_length=self.max_len, padding='max_length', return_tensors='pt')
        return encoding['input_ids'].squeeze()

class CustomEvaluateArguments(TrainingArguments):
    def __init__(self, *args, multiplier=1, d_models=[], lambda_l1=1, lambda_cos=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.multiplier = multiplier
        self.d_models = d_models
        self.lambda_l1 = lambda_l1
        self.lambda_cos = lambda_cos

def load_dataset(file_path, tokenizer, max_len, batch_size):
    dataset = TextDataset(file_path, tokenizer, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

def load_merged_model(model_path, cfg, models, layer, device):
    merged_model = AutoencoderMerged(models, cfg, layer=layer, device=device)
    merged_model.load_state_dict(torch.load(model_path, map_location=device))
    merged_model.eval()
    return merged_model

def evaluate_model(model, dataloader, device):
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch.to(device)
            target_ids = input_ids[:, 1:].contiguous()  
            
            logits = model(input_ids)
            print(logits.shape)
            logits = logits[:, :-1, :].contiguous()  
            
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            print(loss)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


eval_file_path = 'shakespeare.txt'
model_weights_path = './results/model_epoch_2_loss_0.0000.pt'
max_len = 32
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name1 = "Sharathhebbar24/math_gpt2_sft"
tokenizer1 = AutoTokenizer.from_pretrained(model_name1)
model1 = AutoModelForCausalLM.from_pretrained(model_name1).to(device)

model_name2 = "yoavgur/gpt2-bash-history-baseline"
tokenizer2 = AutoTokenizer.from_pretrained(model_name2)
model2 = AutoModelForCausalLM.from_pretrained(model_name2).to(device)

eval_dataloader = load_dataset(eval_file_path, tokenizer1, max_len, batch_size)

cfg = CustomEvaluateArguments(
    output_dir='./results',
    multiplier=70,
    d_models=[768, 768],
    lambda_l1=0.5,
    lambda_cos=0.5,
)

merged_model = load_merged_model(model_weights_path, cfg, [model1, model2], layer=-2, device=device)

mergedcfg=MergedModelArguments()

model=MergedModel([model1, model2], mergedcfg, device, [merged_model])

avg_loss = evaluate_model(model, eval_dataloader, device)

print(f"Evaluation Results:")
print(f"Average Cross Entropy Loss: {avg_loss:.4f}")