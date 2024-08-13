import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from tqdm import tqdm
import os
from sae_direction_alignment import Autoencoder, AutoencoderMerged
from merge import MergedModel, MergedModelArguments, PropagatedModel


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

class CustomEvaluateArguments():
    def __init__(self, multiplier=1, d_models=[], lambda_l1=1, lambda_cos=2):
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
            
            if isinstance(model, PropagatedModel) or isinstance(model, MergedModel):
                logits = model(input_ids)
            else:
                logits = model(input_ids).logits
                
            print(logits.shape)
            logits = logits[:, :-1, :].contiguous()  
            
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            print(loss)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    import yaml
    with open('eval_merged.yaml', 'r') as f:
        config = yaml.safe_load(f)
        f.close()

    if config['mode']=='merge':
        model_weights_path = config['checkpoint_to_test']


        model_name1 = "Sharathhebbar24/math_gpt2_sft"
        tokenizer = AutoTokenizer.from_pretrained(model_name1)
        model1 = AutoModelForCausalLM.from_pretrained(model_name1).to(device)

        model_name2 = "yoavgur/gpt2-bash-history-baseline"
        model2 = AutoModelForCausalLM.from_pretrained(model_name2).to(device)


        cfg = CustomEvaluateArguments(
            multiplier=config['multiplier'],
            d_models=config['d_models'],
            lambda_l1=config['lambda_l1'],
            lambda_cos=config['lambda_cos'],
        )

        merged_model = load_merged_model(model_weights_path, 
                                        cfg, 
                                        models=[model1, model2], 
                                        layer=config['layer'], 
                                        device=device)

        mergedcfg=MergedModelArguments()

        model=MergedModel([model1, model2], mergedcfg, device, [merged_model])
    
    if config['mode']=='propogate':   
        model1 = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        print(model1)
        model=PropagatedModel(model=model1,
                              from_to_layers=[(11,10)],
                              device=device) 
        
    tokenizer.pad_token = tokenizer.eos_token         

    eval_file_path = config['eval_dataset']
    max_len = config['max_len']
    batch_size = config['batch_size']
    
    eval_dataloader = load_dataset(eval_file_path, tokenizer, max_len, batch_size)
    
    avg_loss = evaluate_model(model, eval_dataloader, device)

    print(f"Evaluation Results:")
    print(f"Average Cross Entropy Loss: {avg_loss:.4f}")
    
    if config['mode']=='merge':    
        loss1 = evaluate_model(model1, eval_dataloader, device)
        print(f"Average Cross Entropy Loss for model1: {loss1:.4f}")
        loss2 = evaluate_model(model2, eval_dataloader, device)
        print(f"Average Cross Entropy Loss for model2: {loss2:.4f}")
        
    if config['mode']=='propogate':    
        loss1 = evaluate_model(model1, eval_dataloader, device)
        print(f"Average Cross Entropy Loss for model1: {loss1:.4f}") 