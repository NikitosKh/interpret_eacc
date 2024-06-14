from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import matplotlib.pyplot as plt
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

dataset = load_dataset("izumi-lab/open-text-books")


model_directory_math = '/Users/nikxoma/proj/model_weights/finetuned_gpt2'
model_directory_bio = '/Users/nikxoma/proj/model_weights/gpt2_bio'


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

model_math = GPT2LMHeadModel.from_pretrained(model_directory_math)
model_bio = GPT2LMHeadModel.from_pretrained(model_directory_bio)

model_name = 'gpt2'
def generate_response(prompt, max_length=100, num_return_sequences=1, top_p=0.9, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model_bio.generate(
        inputs['input_ids'], 
        max_length=max_length, 
        num_return_sequences=num_return_sequences,
        do_sample=True,  
        top_p=top_p,     
        temperature=temperature,  
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
class MyDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')
        return inputs.input_ids.squeeze(), inputs.attention_mask.squeeze()



dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


def interpolate_weights(model1, model2, t):
    new_model = GPT2LMHeadModel.from_pretrained(model_name)
    model1_state_dict = model1.state_dict()
    model2_state_dict = model2.state_dict()
    
    new_state_dict = {}
    for key in model1_state_dict.keys():
        new_state_dict[key] = t * model1_state_dict[key] + (1 - t) * model2_state_dict[key]
    
    new_model.load_state_dict(new_state_dict)
    return new_model
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)


tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

def evaluate_model(dataset, model):
    total_loss = 0.0
    total_samples = 0

    for batch in tokenized_dataset['train']:
        input_ids = torch.tensor(batch['input_ids'])
        attention_mask = torch.tensor(batch['attention_mask'])

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)

    avg_loss = total_loss / total_samples
    return avg_loss

def linear_interpolation_merge_eval(number_of_combinations):
    ts = [a/number_of_combinations for a in range(number_of_combinations + 1)]
    losses = []
    for t in ts:
        print(t)
        model = interpolate_weights(model_math, model_bio, t)
        loss = evaluate_model(tokenized_dataset['train'], model)
        losses.append(loss)
    print(losses)
    plt.figure(figsize=(8, 6))  
    plt.plot(ts, losses, marker='o') 

    
    plt.title('Plot of Y against X')
    plt.xlabel('lambda')
    plt.ylabel('loss')

linear_interpolation_merge_eval(10)