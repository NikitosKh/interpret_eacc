import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from tqdm import tqdm
import os
from sae_direction_alignment import Autoencoder, AutoencoderMerged
from merge import MergedModel, MergedModelArguments, PropagatedModel
from datasets import load_dataset



class CustomEvaluateArguments():
    def __init__(self, multiplier=1, d_models=[], lambda_l1=1, lambda_cos=2):
        self.multiplier = multiplier
        self.d_models = d_models
        self.lambda_l1 = lambda_l1
        self.lambda_cos = lambda_cos


def load_merged_model(model_path, cfg, models, layer, device):
    merged_model = AutoencoderMerged(models, cfg, layer=layer, device=device)
    merged_model.load_state_dict(torch.load(model_path, map_location=device))
    merged_model.eval()
    return merged_model


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
        # model=model1
        
    # Step 1: Load the LAMBADA dataset
    dataset = load_dataset("EleutherAI/lambada_openai", 
                           split="test",
                           trust_remote_code=True)

    # Ensure the model is in evaluation mode
    model.eval()

    # Function to calculate accuracy on LAMBADA
    def evaluate_lambada(model, tokenizer, dataset):
        model=model.to(device) 
        correct = 0
        total = 0

        for batch in tqdm(dataset):
            passage = batch['text']  # Get the passage
            words = passage.split()  # Split the passage into words
            context = " ".join(words[:-1])  # Join all words except the last one
            target_word = words[-1]  # The word to predict

            # Tokenize the context (excluding the last word)
            input_ids = tokenizer(context, return_tensors="pt")['input_ids'].to(device)
            target_word_tokenized = tokenizer(target_word, return_tensors="pt")['input_ids'].to(device)

            # Generate the output
            with torch.no_grad(): 
                predicted_id = model.generate(input_ids, 
                                              max_new_tokens=target_word_tokenized.shape[-1]+4, 
                                              pad_token_id=tokenizer.eos_token_id) 
           
            # Get the prediction for the last word
            predicted_word = tokenizer.decode(predicted_id[:, input_ids.shape[-1]:].squeeze())

            # Check if the prediction is correct
            import re
            if len(predicted_word.strip().split()) != 0:
                if re.sub(r'[.,“”!?]', '', predicted_word.strip().split()[0]) == target_word:
                    correct += 1
                total += 1

        accuracy = correct / total
        return accuracy

    # Step 3: Evaluate the model on LAMBADA
    accuracy = evaluate_lambada(model, tokenizer, dataset)
    print(f"LAMBADA Accuracy of the target model: {accuracy:.4f}")
    
    if config['mode']=='merge':    
        accuracy = evaluate_lambada(model1, tokenizer, dataset)
        print(f"LAMBADA Accuracy of model1: {accuracy:.4f}")
        accuracy = evaluate_lambada(model2, tokenizer, dataset)
        print(f"LAMBADA Accuracy of model2: {accuracy:.4f}")
        
    if config['mode']=='propogate':    
        accuracy = evaluate_lambada(model1, tokenizer, dataset)
        print(f"LAMBADA Accuracy of model1: {accuracy:.4f}")    