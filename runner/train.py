import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from runner.evaluate import evaluate_model
from tqdm import tqdm
import pickle
import time

from utils.two_statements_evaluation import evaluate_two_statements

def compute_log_prob_sum(model, input_ids, attention_mask, output_ids):
    # Compute total log prob of output_ids conditioned on input_ids
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, -output_ids.size(1)-1:-1, :]
    log_probs = torch.log_softmax(logits, dim=-1)
    selected = torch.gather(log_probs, 2, output_ids.unsqueeze(-1)).squeeze(-1)
    return selected.sum(dim=1)  # [batch_size]

# def compute_preference_loss(idx,adapted_model, original_model, tokenizer, input_ids, attention_mask, labels, logits, args,device="cuda"):
#     """
#     Computes the preference loss between the adapted model and the original model.
#     """
#     #open original_results.pkl
#     original_results = None
#     with open("original_results.pkl", 'rb') as f:
#         original_results = pickle.load(f)
#     with torch.no_grad():
#         adapted_generated_ids = adapted_model.generate(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             max_new_tokens=args.max_new_tokens,
#             do_sample=True, # Consistent with evaluation.py
#             pad_token_id=tokenizer.eos_token_id # Add pad_token_id for open-ended generation
#         )
#     # Ensure decoding handles cases where input is part of the output
#     # For instruct models, often the prompt is not repeated.
#     # If prompt is repeated, use: result = tokenizer.decode(generated_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

#     # for adapted_generated_id, label zip(adapted_generated_ids, labels, input_id):

#     adapted_result = tokenizer.decode(adapted_generated_ids[0], skip_special_tokens=True)

#     input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
#     # A simple way to remove prompt if it's there, more robust methods might be needed
#     if adapted_result.startswith(input_text):
#         adapted_answer = adapted_result[len(input_text):].strip()
#     if original_results[idx].startswith(input_text):
#         original_answer = original_results[idx][len(input_text):].strip()
#     # else:
#     # adapted_answer = adapted_result.strip()
#     input_text = input_text.strip()
#     # original_answer = original_results[idx]
#     # print("Adapted Answer: ", adapted_answer)
#     # print("Input Text: ", input_text)
#     # print("\n---------------------------------------\n\n")
#     # print("Original Answer: ", original_answer)
#     # print("\n---------------------------------------\n\n")
#     # print("Adapted Answer: ", adapted_answer)
#     # print("\n****************************************************\n\n")

#     sample_input = {
#         "query": input_text,
#         "response_A": adapted_answer,
#         "response_B": original_answer
#     }
#     json_judgment = evaluate_two_statements(sample_input)

#     eval_A = json_judgment.get("evaluation_A", {"score": None, "explanation": "Not parsed"})
#     eval_A_score = eval_A.get("score", None)

#     eval_B = json_judgment.get("evaluation_B", {"score": None, "explanation": "Not parsed"})
#     eval_B_score = eval_B.get("score", None)
#     # print(eval_A_score,eval_B_score)  # Print the result for visibility

#     # calculate loss to maximize the score of adapted model

#     # get max log_softmax of the logits 
#     original_answer_ids = tokenizer(original_answer, return_tensors="pt").input_ids.to(device)
#     min_seq_len = min(logits.size(1), original_answer_ids.size(1))
#     original_answer_ids = original_answer_ids[:, :min_seq_len]
#     logits_trimmed = logits[:, :min_seq_len, :]
#     adapted_log_prob = torch.max(logits_trimmed, dim=-1).values
#     # print("Adapted Log Prob: ", adapted_log_prob.shape,logits.shape)

#     #tokenize the original answer
    
#     # print("Original Answer IDs: ", original_answer_ids.shape)
#     # find the log prob of the original answer ids from logits
    
    
#     original_log_prob = logits_trimmed.gather(dim=2, index=original_answer_ids.unsqueeze(-1)).squeeze(-1)
#     # print("Original Log Prob: ", original_log_prob.shape)

#     log_probA = adapted_log_prob.mean()
#     log_probB = original_log_prob.mean()

#     loss = 0
#     if eval_A_score is not None and eval_B_score is not None:
#         # loss = eval_B_score - eval_A_score  # We want adapted to be better than original
#         # advantage = torch.tensor((eval_A_score - eval_B_score), device=device, dtype=torch.float32)

#         # loss = -advantage*(log_probA-log_probB) + torch.tensor(0.00001, device=device, dtype=torch.float32)

#         if eval_A_score > eval_B_score:
#             # If adapted model is better, we want to maximize its log prob
#             loss = -log_probA + log_probB
#         else:
#             # If original model is better, we want to minimize its log prob
#             loss = -log_probB + log_probA
#         # if eval_A_score > eval_B_score:
#         #     advantage = eval_A_score - eval_B_score
#         #     loss = -advantage
#         # else:
#         #     advantage = eval_B_score - eval_A_score



#     # make loss a tensor
#     # loss = torch.tensor(loss, device=device, dtype=torch.float32)
#     # print("Loss: ", loss.item())
#     print("Adapted: ",eval_A_score, "Original: ", eval_B_score)
    
#     return loss
#     # print("Sample Input: ", sample_input)
#     # compare the two strings


# Original - Dulanga
# def compute_preference_loss(idx,adapted_model, original_model, tokenizer, input_ids, attention_mask, labels, logits, args,device="cuda"):
#     """
#     Computes the preference loss between the adapted model and the original model.
#     """
#     #open original_results.pkl
#     original_results = None
#     with open('original_results.pkl', 'rb') as f:
#         original_results = pickle.load(f)
#     with torch.no_grad():
#         adapted_generated_ids = adapted_model.generate(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             max_new_tokens=args.max_new_tokens,
#             do_sample=True, # Consistent with evaluation.py
#             pad_token_id=tokenizer.eos_token_id # Add pad_token_id for open-ended generation
#         )
        
#     # Ensure decoding handles cases where input is part of the output
#     # For instruct models, often the prompt is not repeated.
#     # If prompt is repeated, use: result = tokenizer.decode(generated_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
#     index = idx*args.batch_size
#     # loss = torch.tensor(0.0, device=device, dtype=torch.float32)
#     loss = 0
    
#     for adapted_generated_id, label, input_id in zip(adapted_generated_ids, labels, input_ids):

#         try:

#             adapted_result = tokenizer.decode(adapted_generated_id, skip_special_tokens=True)

#             input_text = tokenizer.decode(input_id, skip_special_tokens=True)

#             # gpt_answer = tokenizer.decode(label, skip_special_tokens=True)
#             # A simple way to remove prompt if it's there, more robust methods might be needed
#             if adapted_result.startswith(input_text):
#                 adapted_answer = adapted_result[len(input_text):].strip()
#             if original_results[index].startswith(input_text):    
#                 original_answer = original_results[index][len(input_text):].strip()
#             # else:
#             # adapted_answer = adapted_result.strip()
#             input_text = input_text.strip()
#             # original_answer = original_results[index]

#             # print("\nInput Text: ", input_text)
#             # print("\n---------------------------------------\n\n")
#             print("Adapted Answer: ", adapted_answer)

#             # print("\n---------------------------------------\n\n")
#             # print("Original Answer: ", original_answer)
#             # print("\n****************************************************\n\n")

#             sample_input = {
#                 "query": input_text,
#                 "response_A": adapted_answer,
#                 "response_B": original_answer
#             }
#             json_judgment = evaluate_two_statements(sample_input)

#             eval_A = json_judgment.get("evaluation_A", {"score": None, "explanation": "Not parsed"})
#             eval_A_score = eval_A.get("score", None)

#             eval_B = json_judgment.get("evaluation_B", {"score": None, "explanation": "Not parsed"})
#             eval_B_score = eval_B.get("score", None)
#             # print(eval_A_score,eval_B_score)  # Print the result for visibility

#             # calculate loss to maximize the score of adapted model

#             # get max log_softmax of the logits 
#             original_answer_ids = tokenizer(original_answer, return_tensors="pt").input_ids.to(device)
#             min_seq_len = min(logits.size(1), original_answer_ids.size(1))
#             original_answer_ids = original_answer_ids[:, :min_seq_len]
#             logits_trimmed = logits[:, :min_seq_len, :]
#             adapted_log_prob = torch.max(logits_trimmed, dim=-1).values
#             # print("Adapted Log Prob: ", adapted_log_prob.shape,logits.shape)

#             #tokenize the original answer

#             # print("Original Answer IDs: ", original_answer_ids.shape)
#             # find the log prob of the original answer ids from logits


#             original_log_prob = logits_trimmed.gather(dim=2, index=original_answer_ids.unsqueeze(-1)).squeeze(-1)
#             # print("Original Log Prob: ", original_log_prob.shape)

#             log_probA = adapted_log_prob.mean()
#             log_probB = original_log_prob.mean()

#             # loss = 0 #bug
#             if eval_A_score == None:
#                 eval_A_score = 0.0
#             if eval_B_score == None:
#                 eval_B_score = 0.0
#             if eval_A_score is not None and eval_B_score is not None:
#                 # loss = eval_B_score - eval_A_score  # We want adapted to be better than original
                
#                 # advantage = torch.tensor((eval_A_score - eval_B_score), device=device, dtype=torch.float32)
#                 # advantage = torch.tanh(advantage)

#                 # loss = -advantage*(log_probA-log_probB) #+ torch.tensor(0.00001, device=device, dtype=torch.float32)

#                 # Working version
#                 if eval_A_score > eval_B_score:
#                     # If adapted model is better, we want to maximize its log prob
#                     loss += -log_probA + log_probB
#                 else:
#                     # If original model is better, we want to minimize its log prob
#                     loss += -log_probB + log_probA





#             print("Adapted: ",eval_A_score, "Original: ", eval_B_score)
#             index+=1
#         except IndexError:
#             print("List Index Out of Error - Preference Loss")
#             pass 

    

#     return torch.tensor(float(loss), requires_grad=True, device=device)
#     # print("Sample Input: ", sample_input)
#     # compare the two strings

# Sasika - I did these changes 
import pickle
import torch

def compute_preference_loss(idx, adapted_model, original_model, tokenizer, input_ids, attention_mask, labels, logits, args, device="cuda"):
    """
    Computes the preference loss between the adapted model and the original model.
    Adds detailed debug logging and error handling.
    """
    # Load original results
    with open('original_results.pkl', 'rb') as f:
        original_results = pickle.load(f)

    # print(f"[DEBUG] Loaded {len(original_results)} original results from 'original_results.pkl'")

    # Calculate index for current batch
    index = idx * args.batch_size
    # print(f"[DEBUG] Starting index for current batch: {index}")

    # Total loss (accumulator)
    total_loss = 0

    # Generate predictions from adapted model
    with torch.no_grad():
        adapted_generated_ids = adapted_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    for i, (adapted_generated_id, label, input_id) in enumerate(zip(adapted_generated_ids, labels, input_ids)):
        current_index = index + i
        # print(f"\n[DEBUG] Processing sample {i} (global index {current_index})")

        if current_index >= len(original_results):
            print(f"[ERROR] Index {current_index} out of bounds for original_results (len={len(original_results)})")
            continue

        try:
            adapted_result = tokenizer.decode(adapted_generated_id, skip_special_tokens=True)
            input_text = tokenizer.decode(input_id, skip_special_tokens=True)
            original_result = original_results[current_index]

            # Remove the prompt if it appears in the generation
            adapted_answer = adapted_result[len(input_text):].strip() if adapted_result.startswith(input_text) else adapted_result.strip()
            original_answer = original_result[len(input_text):].strip() if original_result.startswith(input_text) else original_result.strip()

            # print(f" Input Text: {input_text[:100]}...")  # truncate long text
            # print(f"Adapted Answer: {adapted_answer[:100]}...")
            # print(f"Original Answer: {original_answer[:100]}...")

            sample_input = {
                "query": input_text,
                "response_A": adapted_answer,
                "response_B": original_answer
            }

            json_judgment = evaluate_two_statements(sample_input)
            eval_A = json_judgment.get("evaluation_A", {"score": 0.0})
            eval_B = json_judgment.get("evaluation_B", {"score": 0.0})
            eval_A_score = eval_A.get("score", 0.0) or 0.0
            eval_B_score = eval_B.get("score", 0.0) or 0.0

            print(f" Eval Scores -> Adapted: {eval_A_score}, Original: {eval_B_score}")

            # Convert original answer to token IDs
            original_answer_ids = tokenizer(original_answer, return_tensors="pt").input_ids.to(device)
            min_seq_len = min(logits.size(1), original_answer_ids.size(1))
            original_answer_ids = original_answer_ids[:, :min_seq_len]
            logits_trimmed = logits[:, :min_seq_len, :]

            adapted_log_prob = torch.max(logits_trimmed, dim=-1).values
            original_log_prob = logits_trimmed.gather(dim=2, index=original_answer_ids.unsqueeze(-1)).squeeze(-1)

            log_probA = adapted_log_prob.mean()
            log_probB = original_log_prob.mean()

            if eval_A_score > eval_B_score:
                # Adapted is better → reward adapted
                sample_loss = -log_probA + log_probB
            else:
                # Original is better → penalize adapted
                sample_loss = -log_probB + log_probA

            total_loss += sample_loss

        except IndexError as e:
            print(f"[ERROR] IndexError at sample {i}: {e}")
            import sys
            sys.exit()
        except Exception as e:
            print(f"[ERROR] Unexpected exception at sample {i}: {e}")


    final_loss = torch.tensor(float(total_loss), requires_grad=True, device=device)
    print(f"\n[DEBUG] Total computed loss: {final_loss.item():.4f}")
    return final_loss


def freeze_model_except_adapters(model):
    for name, param in model.named_parameters():
        # print("Adapter? ",name)
        # if "Sequential" in name or "adapter" or "model.layers.27.input_layernorm" in name:
        if "Sequential" in name or "adapter" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
        
    # print number of trainable parameters here.
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    

    all_params = sum(p.numel() for p in model.parameters())

    print(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {trainable_params/all_params*100}")

def train_model(model, config, dataset, processor, preprocess_fn=None, batch_size=8, epochs=3, device="cuda", lr=0.0001):
    if config['models'][0]['name'] == 'qwen':
        train_model_qwen(model, config, dataset, processor, preprocess_fn, batch_size, epochs, device, lr)
    elif config['models'][0]['name'] == 'florence': 
        train_model_florence(model, config, dataset, processor, preprocess_fn, batch_size, epochs, device, lr)
    elif config['models'][0]['name'] == 'florence-large': 
        train_model_florence(model, config, dataset, processor, preprocess_fn, batch_size, epochs, device, lr)
    elif config['models'][0]['name'] == 'mistral-7b-instruct':
        train_model_mistral(model, config, dataset, processor, preprocess_fn, batch_size, epochs, device, lr)


def train_model_florence(model, config, dataset, processor, preprocess_fn=None, batch_size=8, epochs=3, device="cuda", lr=0.0001):
    loader = dataset['train']
    model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    for epoch in range(epochs):  
        
        # linear_layer.train()
        total_train_loss = 0  

        for idx,batch in enumerate(loader):
            # break
            inputs, answers = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}

            image_features = model._encode_image(inputs['pixel_values']).to(device)
            input_embeds = model.get_input_embeddings()(inputs['input_ids']).to(device)
            input_embeds, attention_mask = model._merge_input_ids_with_image_features(input_embeds, image_features)

            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"] 
            labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to('cuda')
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)

            # Compute loss
            # loss = loss_criterion(outputs.logits.view(-1, outputs.logits.size(-1)),labels.view(-1))
            loss = outputs.loss
            # print('loss',loss)
            # print("NAN: ",torch.isnan(outputs.logits).any())
            # print("NAN: ",outputs.logits)  # Should be False
            # print("NAN: ",torch.isnan(labels).any())  # Should be False
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()  
            # wandb.log({
            # "train_loss": loss.item(),
            # "batch_id": idx
            # })
            if idx == 50:
                break
        avg_train_loss = total_train_loss / 50 
        print(avg_train_loss)

        # Validation Loop
        # Adapter validation 
        # linear_layer.eval()
        if epoch % 20 == 0:
            evaluate_model(model, config, dataset,processor)
            # print(f"Accuracy: {acc:.4f}")



def train_model_qwen(model, config, dataset, processor, preprocess_fn=None, batch_size=8, epochs=3, device="cuda", lr=0.0001):
    # loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loader = dataset['train']
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0.0
        idx =1
        pbar = tqdm(loader, desc=f"Training Epoch {epoch+1}")
        for batch in pbar:
            inputs = preprocess_fn(batch) if preprocess_fn else batch
            labels = batch["labels"].to(device)
            # print(labels.shape)

            inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
            # refs = inputs.pop('ref')
            outputs = model(**inputs)
            loss = outputs.loss
            # print(outputs)
            # loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_description(f"Loss: {total_loss/idx:.4f}")
            idx+=1

            if idx==50:
                break

        if epoch%20 == 0:
            acc = evaluate_model(model, config, dataset,processor)
            print(f"Accuracy: {acc:.4f}")


        print(f"[Epoch {epoch+1}] Avg Loss: {total_loss / len(loader):.4f}")

# --------------
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_and_tokenize_dataset(dataset_hf, tokenizer, max_seq_length):
    """
    Formats the conversational history and tokenizes it for Mistral instruction fine-tuning.
    Masks prompt tokens in the labels.

    Args:
        dataset_hf (datasets.Dataset): The Hugging Face dataset.
                                       Expected to have a 'history' column.
        tokenizer: The tokenizer.
        max_seq_length (int): Maximum sequence length for truncation.

    Returns:
        dict: A dictionary containing 'input_ids', 'attention_mask', and 'labels'.
    """
    processed_examples = {'input_ids': [], 'attention_mask': [], 'labels': []}

    # Determine BOS and EOS tokens from the tokenizer
    bos = tokenizer.bos_token if tokenizer.bos_token else "<s>"
    eos = tokenizer.eos_token if tokenizer.eos_token else "</s>"
    inst_open = "[INST]"
    inst_close = "[/INST]"

    print(f"Using BOS: '{bos}', EOS: '{eos}' for formatting.")

    for item_idx, item_history in enumerate(dataset_hf['history']):
        full_concatenated_input_ids = []
        full_concatenated_labels = []

        if not isinstance(item_history, list):
            print(f"Item at index {item_idx} has history of type {type(item_history)}, expected list. Skipping.")
            continue

        for turn_idx, turn in enumerate(item_history):
            if not isinstance(turn, dict) or 'user' not in turn or 'bot' not in turn:
                print(f"Turn {turn_idx} in item {item_idx} is malformed: {turn}. Skipping turn.")
                continue
            
            user_query = str(turn['user'])
            bot_response = str(turn['bot'])

            # Format for Mistral: <s>[INST] User Query [/INST] Bot Response</s>
            # Note: A space is often added before the bot_response if not handled by tokenizer.
            prompt_str = f"{bos}{inst_open} {user_query} {inst_close}"
            answer_str = f" {bot_response}{eos}" # Leading space for the answer part

            # Tokenize prompt and answer parts separately to correctly create labels
            # add_special_tokens=False because we are manually adding BOS/EOS per turn segment
            prompt_tokens = tokenizer.encode(prompt_str, add_special_tokens=False)
            answer_tokens = tokenizer.encode(answer_str, add_special_tokens=False)
            
            current_turn_input_ids = prompt_tokens + answer_tokens
            # For labels, mask the prompt part by setting tokens to -100
            current_turn_labels = [-100] * len(prompt_tokens) + answer_tokens
            
            full_concatenated_input_ids.extend(current_turn_input_ids)
            full_concatenated_labels.extend(current_turn_labels)

        # Truncate if the full concatenated history exceeds max_seq_length
        if len(full_concatenated_input_ids) > max_seq_length:
            full_concatenated_input_ids = full_concatenated_input_ids[:max_seq_length]
            full_concatenated_labels = full_concatenated_labels[:max_seq_length]
        elif len(full_concatenated_input_ids) == 0: # Handle empty history cases
            print(f"Item at index {item_idx} resulted in empty tokenized output. Skipping.")
            continue
            
        # Create attention mask (1 for real tokens, 0 for padding - padding handled by collator)
        attention_mask = [1] * len(full_concatenated_input_ids)

        processed_examples['input_ids'].append(full_concatenated_input_ids)
        processed_examples['attention_mask'].append(attention_mask)
        processed_examples['labels'].append(full_concatenated_labels)
        
    return processed_examples


class ConversationDataset(Dataset):
    """PyTorch Dataset for conversational data."""
    def __init__(self, tokenized_data):
        self.input_ids = tokenized_data['input_ids']
        self.attention_mask = tokenized_data['attention_mask']
        self.labels = tokenized_data['labels']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def collate_fn_conversations(batch, tokenizer):
    """Collate function to pad batch elements to the same length."""
    input_ids_list = [item['input_ids'] for item in batch]
    attention_mask_list = [item['attention_mask'] for item in batch]
    labels_list = [item['labels'] for item in batch]

    # Determine max length in this batch for padding
    max_len = max(len(ids) for ids in input_ids_list)
    if max_len == 0: # Should not happen if empty examples are filtered
        return None 

    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        # Fallback if pad_token_id is not set, common to use eos_token_id
        pad_token_id = tokenizer.eos_token_id 
        print(f"tokenizer.pad_token_id is None. Using eos_token_id ({pad_token_id}) for padding.")
        if pad_token_id is None: # Critical error if no pad token can be determined
             raise ValueError("Tokenizer has no pad_token_id and no eos_token_id to use as fallback for padding.")


    for i in range(len(batch)):
        input_ids = input_ids_list[i]
        attention_mask = attention_mask_list[i]
        labels = labels_list[i]
        
        padding_length = max_len - len(input_ids)
        
        # Pad right
        padded_input_ids.append(torch.cat([input_ids, torch.full((padding_length,), pad_token_id, dtype=torch.long)]))
        padded_attention_mask.append(torch.cat([attention_mask, torch.full((padding_length,), 0, dtype=torch.long)])) # Pad attention mask with 0
        padded_labels.append(torch.cat([labels, torch.full((padding_length,), -100, dtype=torch.long)])) # Pad labels with -100 (ignore index)

    return {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_mask),
        "labels": torch.stack(padded_labels)
    }



def train_model_mistral(adapted_model,original_model, tokenizer, train_dataset_hf, args):
    """
    Trains a Mistral model using the provided dataset and arguments.

    Args:
        original_model: The pre-trained Mistral model (e.g., from AutoModelForCausalLM.from_pretrained).
        tokenizer: The tokenizer for the model (e.g., from AutoTokenizer.from_pretrained).
        train_dataset_hf (datasets.Dataset): The Hugging Face training dataset.
                                            Must contain a 'history' column, where each item is a list of turns,
                                            and each turn is a dict {'user': str, 'bot': str}.
        args: An object or Namespace containing training arguments:
              - num_epochs (int): Number of training epochs.
              - model_save_path (str): Path to save the fine-tuned model and tokenizer.
              - learning_rate (float): Optimizer learning rate (e.g., 2e-5, 5e-5).
              - batch_size (int): Training batch size (e.g., 1, 2, 4, adjust based on GPU memory).
              - max_seq_length (int): Maximum sequence length for tokenization and padding (e.g., 512, 1024, 2048).
              - gradient_accumulation_steps (int, optional): Number of steps to accumulate gradients before an optimizer update. Defaults to 1.
              - warmup_steps (int, optional): Number of warmup steps for the learning rate scheduler. Defaults to 0.
              - logging_steps (int, optional): Log training loss every X steps. Defaults to 10.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    adapted_model.to(device)
    adapted_model.train() # Set model to training mode
    original_model.to(device)
    original_model.eval() # Set model to training mode

    # Ensure tokenizer has a pad token. This is crucial for batching.
    if tokenizer.pad_token is None:
        print("Tokenizer does not have a pad_token. Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        # Important: If you add a new token or change pad_token such that vocab size changes,
        # you might need to resize model token embeddings:
        # original_model.resize_token_embeddings(len(tokenizer))
        # However, just setting pad_token = eos_token usually means using an existing token.
    if tokenizer.pad_token_id is None: # Ensure pad_token_id is also set
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Tokenizer pad token ID: {tokenizer.pad_token_id}")


    # 1. Preprocess and tokenize the dataset
    print("Preprocessing and tokenizing dataset...")
    tokenized_data_dict = format_and_tokenize_dataset(train_dataset_hf, tokenizer, args.max_seq_length)
    
    if not tokenized_data_dict['input_ids']:
        print("Tokenization resulted in an empty dataset. Please check your data and formatting.")
        return None

    # Create a PyTorch Dataset
    pytorch_train_dataset = ConversationDataset(tokenized_data_dict)
    print(f"Created PyTorch Dataset with {len(pytorch_train_dataset)} examples.")


    # 2. Create DataLoader
    print(f"Creating DataLoader with batch size: {args.batch_size}...")
    train_dataloader = DataLoader(
        pytorch_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn_conversations(batch, tokenizer)
    )

    # 3. Set up Optimizer and Scheduler
    print(f"Setting up optimizer with learning rate: {args.learning_rate}...")

    # Freeze all parameters
    for name, param in adapted_model.named_parameters():
        param.requires_grad = False

    # Unfreeze only adapter layers based on name match
    adapter_layer_prefixes = [layer['name'] for layer in args.adapter_layers_json]

    for name, param in adapted_model.named_parameters():
        for prefix in adapter_layer_prefixes:
            if name.startswith(prefix):
                param.requires_grad = True
                
                break

    print("trainable layers: ")
    # Print layers which require gradients (i.e., will be updated during training)
    for name, param in adapted_model.named_parameters():
        if param.requires_grad:
            print(name)

    # Count total and trainable parameters
    total_params = sum(p.numel() for p in adapted_model.parameters())
    trainable_params = sum(p.numel() for p in adapted_model.parameters() if p.requires_grad)

    print(f"\nTrainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")


    # Only pass trainable parameters to the optimizer
    optimizer = AdamW(filter(lambda p: p.requires_grad, adapted_model.parameters()), lr=args.learning_rate, eps=1e-8) # Added eps for stability

    gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
    num_training_steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps
    if len(train_dataloader) % gradient_accumulation_steps != 0:
        num_training_steps_per_epoch +=1 # account for the last partial step

    total_training_steps = num_training_steps_per_epoch * args.num_epochs
    
    num_warmup_steps = getattr(args, 'warmup_steps', 0)
    if isinstance(num_warmup_steps, float): # if warmup_steps is a ratio
        num_warmup_steps = int(total_training_steps * num_warmup_steps)

    print(f"Total training steps: {total_training_steps}, Warmup steps: {num_warmup_steps}")
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps
    )
    
    logging_steps = getattr(args, 'logging_steps', 10)

    # 4. Training Loop
    print(f"Starting training for {args.num_epochs} epochs...")
    adapted_model.zero_grad() # Clear gradients before starting

    for epoch in range(args.num_epochs):
        print(f"--- Epoch {epoch+1}/{args.num_epochs} ---")
        epoch_total_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            if batch is None: # Skip if collate_fn returned None (e.g. empty batch after filtering)
                continue

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = adapted_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps # Normalize loss
            
            # Backward pass
            loss.backward()
            
            # Optimizer step (with gradient accumulation)
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(adapted_model.parameters(), 1.0) # Gradient clipping
                optimizer.step()
                scheduler.step() # Update learning rate
                optimizer.zero_grad() # Clear gradients for the next accumulation

            epoch_total_loss += loss.item() * gradient_accumulation_steps # De-normalize for logging

            

            start_time = time.time()

            if (step + 1) % (logging_steps * gradient_accumulation_steps) == 0:
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                step_time = elapsed * 1000 / logging_steps  # in ms
                tokens_per_second = int(input_ids.numel() * logging_steps / elapsed)

                grad_norm = 0.0
                for p in adapted_model.parameters():
                    if p.requires_grad and p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5

                print(
                    f"step {step+1}/{len(train_dataloader)} | "
                    f"loss {loss.item() * gradient_accumulation_steps:.6f} (+nanz)| "
                    f"norm {grad_norm:.4f} (+nanz)| "
                    f"lr {current_lr:.2e} | "
                    f"{step_time:.2f} ms | "
                    f"{tokens_per_second} tok/s",
                    flush=True
                )

                start_time = time.time()


        avg_epoch_loss = epoch_total_loss / len(train_dataloader)
        print(f"--- End of Epoch {epoch+1}, Average Loss: {avg_epoch_loss:.4f} ---")

    

    return adapted_model
 
def generate_predictions_for_original_model(original_model, tokenizer, train_dataset_hf, args):
    """
    Generates predictions using the original model.

    Args:
        original_model: The pre-trained Mistral model.
        tokenizer: The tokenizer for the model.
        input_ids (torch.Tensor): Input IDs for the model.
        attention_mask (torch.Tensor): Attention mask for the input.
        max_new_tokens (int): Maximum number of new tokens to generate.

    Returns:
        torch.Tensor: Generated token IDs.
    """
    original_results = []
    for step, batch in enumerate(train_dataloader):
        if batch is None: # Skip if collate_fn returned None (e.g. empty batch after filtering)
            continue

        #print all keys in batch
        print("Batch keys: ", batch.keys())

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        with torch.no_grad():
            generated_ids = original_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,  # Use sampling for diversity
                pad_token_id=tokenizer.eos_token_id  # Ensure padding is handled correctly
            )
        result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # A simple way to remove prompt if it's there, more robust methods might be needed
        # if result.startswith(model_input_text):
        #     parsed_answer = result[len(model_input_text):].strip()
        # else:
        answer = result.strip()
        print("Original Answer: ", answer)
        original_results.append(answer)

def train_model_adapted_llama_2(adapted_model, original_model, tokenizer, train_dataset_hf, args, DEBUG=False):
    # TODO: to be completed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    original_model.to(device)
    original_model.eval() # Set model to training mode

    adapted_model.to(device)
    adapted_model.train() # Set model to training mode

    # Ensure tokenizer has a pad token. This is crucial for batching.
    if tokenizer.pad_token is None:
        print("Tokenizer does not have a pad_token. Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        # Important: If you add a new token or change pad_token such that vocab size changes,
        # you might need to resize model token embeddings:
        # original_model.resize_token_embeddings(len(tokenizer))
        # However, just setting pad_token = eos_token usually means using an existing token.
    if tokenizer.pad_token_id is None: # Ensure pad_token_id is also set
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Tokenizer pad token ID: {tokenizer.pad_token_id}")

    # 1. Preprocess and tokenize the dataset
    print("Preprocessing and tokenizing dataset...")
    tokenized_data_dict = format_and_tokenize_dataset(train_dataset_hf, tokenizer, args.max_seq_length)
    
    if not tokenized_data_dict['input_ids']:
        print("Tokenization resulted in an empty dataset. Please check your data and formatting.")
        return None
    
    # Create a PyTorch Dataset
    pytorch_train_dataset = ConversationDataset(tokenized_data_dict)
    print(f"Created PyTorch Dataset with {len(pytorch_train_dataset)} examples.")

    # Create DataLoader
    print(f"Creating DataLoader with batch size: {args.batch_size}...")
    train_dataloader = DataLoader(
        pytorch_train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn_conversations(batch, tokenizer)
    )

    # 3. Set up Optimizer and Scheduler
    print(f"Setting up optimizer with learning rate: {args.learning_rate}...")

    print("trainable layers: ")
    # Print layers which require gradients (i.e., will be updated during training)
    for name, param in adapted_model.named_parameters():
        if param.requires_grad:
            print("✓", name)

    # Count total and trainable parameters
    total_params = sum(p.numel() for p in adapted_model.parameters())
    trainable_params = sum(p.numel() for p in adapted_model.parameters() if p.requires_grad)

    print(f"\nTrainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

    # Only pass trainable parameters to the optimizer
    optimizer = AdamW(filter(lambda p: p.requires_grad, adapted_model.parameters()), lr=args.learning_rate, eps=1e-8) # Added eps for stability

    gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
    num_training_steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps
    if len(train_dataloader) % gradient_accumulation_steps != 0:
        num_training_steps_per_epoch +=1 # account for the last partial step

    total_training_steps = num_training_steps_per_epoch * args.num_epochs
    
    num_warmup_steps = getattr(args, 'warmup_steps', 0)
    if isinstance(num_warmup_steps, float): # if warmup_steps is a ratio
        num_warmup_steps = int(total_training_steps * num_warmup_steps)

    print(f"Total training steps: {total_training_steps}, Warmup steps: {num_warmup_steps}")
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps
    )
    
    logging_steps = getattr(args, 'logging_steps', 10)

    USE_ORIGINAL_MODEL_FOR_COMPARISON = False # Run this before perference optimization to generate original_results.pkl

    if USE_ORIGINAL_MODEL_FOR_COMPARISON:
        print("Generating predictions using the original model before training...")

        original_results = []

        for step, batch in enumerate(train_dataloader):
            if batch is None:  # Skip if collate_fn returned None (e.g., empty batch after filtering)
                continue

            print(f"Step {step}/{len(train_dataloader)} | Batch keys: {batch.keys()}")

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.no_grad():
                generated_ids = original_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,  # Use sampling for diversity
                    pad_token_id=tokenizer.eos_token_id  # Ensure padding is handled correctly
                )

            # Loop through each sample in the batch
            for i in range(generated_ids.size(0)):
                result = tokenizer.decode(generated_ids[i], skip_special_tokens=True)
                answer = result.strip()
                original_results.append(answer)

                print(f"Step {step}, Sample {i}: Original Answer (truncated): {answer[:80]}...")

        total_samples_collected = len(original_results)
        print(f"[DEBUG] Total samples collected in original_results: {total_samples_collected}")

        # Save to pickle
        original_results_path = "original_results.pkl"
        with open(original_results_path, 'wb') as f:
            pickle.dump(original_results, f)

        print(f"[INFO] Original results saved to {original_results_path}")

        # ✅ Sanity Check: Verify the number of results
        try:
            expected_total = len(train_dataloader.dataset)
            print(f"[CHECK] Collected {total_samples_collected} / Expected {expected_total} samples")
            if total_samples_collected != expected_total:
                print("[WARNING] Mismatch between collected and expected sample count!")
        except AttributeError:
            print("[WARNING] Could not determine expected dataset size for sanity check.")


    # 4. Training Loop
    print(f"Starting training for {args.num_epochs} epochs...")
    adapted_model.zero_grad() # Clear gradients before starting
    print("Num Epochs:", args.num_epochs)

    for epoch in range(args.num_epochs):
        print(f"--- Epoch {epoch+1}/{args.num_epochs} ---")
        print("Epoch: ", epoch, " Out of: ", args.num_epochs)
        epoch_total_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            if batch is None: # Skip if collate_fn returned None (e.g. empty batch after filtering)
                continue

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device).to(adapted_model.dtype)
            labels = batch['labels'].to(device)

            # if DEBUG: 
            #     for name, param in adapted_model.named_parameters():
            #         print(f"{name:50} — dtype: {param.dtype}")

            # Forward pass
            outputs = adapted_model(
                input_ids=input_ids,
                attention_mask=attention_mask,  # must use the casted one
                labels=labels
            )


            logits = outputs.logits
            #compute log softmax
            logits = torch.log_softmax(logits, dim=-1)

            # outputs_orig = original_model(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     labels=labels
            # )

            # print("[logits]")

            loss = compute_preference_loss(step,adapted_model, original_model, tokenizer, input_ids, attention_mask,labels, logits, args,device=device)
            
            # Convert loss to a tensor
             
            # TODO: Does this break the compute graph?
            print("[loss]",loss)

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps # Normalize loss
            
            # Backward pass
            loss.backward()
            
            # Optimizer step (with gradient accumulation)
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(adapted_model.parameters(), 1.0) # Gradient clipping
                optimizer.step()
                scheduler.step() # Update learning rate
                optimizer.zero_grad() # Clear gradients for the next accumulation

            epoch_total_loss += loss.item() * gradient_accumulation_steps # De-normalize for logging

            

            start_time = time.time()

            if (step + 1) % (logging_steps * gradient_accumulation_steps) == 0:
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                step_time = elapsed * 1000 / logging_steps  # in ms
                tokens_per_second = int(input_ids.numel() * logging_steps / elapsed)

                grad_norm = 0.0
                for p in adapted_model.parameters():
                    if p.requires_grad and p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5

                print(
                    f"step {step+1}/{len(train_dataloader)} | "
                    f"loss {loss.item() * gradient_accumulation_steps:.6f} (+nanz)| "
                    f"norm {grad_norm:.4f} (+nanz)| "
                    f"lr {current_lr:.2e} | "
                    f"{step_time:.2f} ms | "
                    f"{tokens_per_second} tok/s",
                    flush=True
                )

                start_time = time.time()
            # break # remove this later


        avg_epoch_loss = epoch_total_loss / len(train_dataloader)
        print(f"--- End of Epoch {epoch+1}, Average Loss: {avg_epoch_loss:.4f} ---")
    
    return adapted_model

    
def train_model_adapted_mistral(adapted_model, original_model, tokenizer, train_dataset_hf, args):
    """
    Trains a Mistral model using the provided dataset and arguments.

    Args:
        original_model: The pre-trained Mistral model (e.g., from AutoModelForCausalLM.from_pretrained).
        tokenizer: The tokenizer for the model (e.g., from AutoTokenizer.from_pretrained).
        train_dataset_hf (datasets.Dataset): The Hugging Face training dataset.
                                            Must contain a 'history' column, where each item is a list of turns,
                                            and each turn is a dict {'user': str, 'bot': str}.
        args: An object or Namespace containing training arguments:
              - num_epochs (int): Number of training epochs.
              - model_save_path (str): Path to save the fine-tuned model and tokenizer.
              - learning_rate (float): Optimizer learning rate (e.g., 2e-5, 5e-5).
              - batch_size (int): Training batch size (e.g., 1, 2, 4, adjust based on GPU memory).
              - max_seq_length (int): Maximum sequence length for tokenization and padding (e.g., 512, 1024, 2048).
              - gradient_accumulation_steps (int, optional): Number of steps to accumulate gradients before an optimizer update. Defaults to 1.
              - warmup_steps (int, optional): Number of warmup steps for the learning rate scheduler. Defaults to 0.
              - logging_steps (int, optional): Log training loss every X steps. Defaults to 10.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    original_model.to(device)
    original_model.eval() # Set model to training mode

    adapted_model.to(device)
    adapted_model.train() # Set model to training mode

    # Ensure tokenizer has a pad token. This is crucial for batching.
    if tokenizer.pad_token is None:
        print("Tokenizer does not have a pad_token. Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        # Important: If you add a new token or change pad_token such that vocab size changes,
        # you might need to resize model token embeddings:
        # original_model.resize_token_embeddings(len(tokenizer))
        # However, just setting pad_token = eos_token usually means using an existing token.
    if tokenizer.pad_token_id is None: # Ensure pad_token_id is also set
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Tokenizer pad token ID: {tokenizer.pad_token_id}")


    # 1. Preprocess and tokenize the dataset
    print("Preprocessing and tokenizing dataset...")
    tokenized_data_dict = format_and_tokenize_dataset(train_dataset_hf, tokenizer, args.max_seq_length)
    
    if not tokenized_data_dict['input_ids']:
        print("Tokenization resulted in an empty dataset. Please check your data and formatting.")
        return None

    # Create a PyTorch Dataset
    pytorch_train_dataset = ConversationDataset(tokenized_data_dict)
    print(f"Created PyTorch Dataset with {len(pytorch_train_dataset)} examples.")


    # 2. Create DataLoader
    print(f"Creating DataLoader with batch size: {args.batch_size}...")
    train_dataloader = DataLoader(
        pytorch_train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn_conversations(batch, tokenizer)
    )

    # 3. Set up Optimizer and Scheduler
    print(f"Setting up optimizer with learning rate: {args.learning_rate}...")

    print("trainable layers: ")
    # Print layers which require gradients (i.e., will be updated during training)
    for name, param in adapted_model.named_parameters():
        if param.requires_grad:
            print("✓", name)

    # Count total and trainable parameters
    total_params = sum(p.numel() for p in adapted_model.parameters())
    trainable_params = sum(p.numel() for p in adapted_model.parameters() if p.requires_grad)

    print(f"\nTrainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

    # Only pass trainable parameters to the optimizer
    optimizer = AdamW(filter(lambda p: p.requires_grad, adapted_model.parameters()), lr=args.learning_rate, eps=1e-8) # Added eps for stability

    gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
    num_training_steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps
    if len(train_dataloader) % gradient_accumulation_steps != 0:
        num_training_steps_per_epoch +=1 # account for the last partial step

    total_training_steps = num_training_steps_per_epoch * args.num_epochs
    
    num_warmup_steps = getattr(args, 'warmup_steps', 0)
    if isinstance(num_warmup_steps, float): # if warmup_steps is a ratio
        num_warmup_steps = int(total_training_steps * num_warmup_steps)

    print(f"Total training steps: {total_training_steps}, Warmup steps: {num_warmup_steps}")
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps
    )
    
    logging_steps = getattr(args, 'logging_steps', 10)

    # # Generate predictions using the original model before training
    # print("Generating predictions using the original model before training...")
    # original_results = []
    # for step, batch in enumerate(train_dataloader):
    #     if batch is None: # Skip if collate_fn returned None (e.g. empty batch after filtering)
    #         continue

    #     #print all keys in batch
    #     print("Batch keys: ", batch.keys())

    #     input_ids = batch['input_ids'].to(device)
    #     attention_mask = batch['attention_mask'].to(device)
    #     labels = batch['labels'].to(device)
    #     with torch.no_grad():
    #         generated_ids = original_model.generate(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             max_new_tokens=args.max_new_tokens,
    #             do_sample=True,  # Use sampling for diversity
    #             pad_token_id=tokenizer.eos_token_id  # Ensure padding is handled correctly
    #         )
    #     result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    #     # A simple way to remove prompt if it's there, more robust methods might be needed
    #     # if result.startswith(model_input_text):
    #     #     parsed_answer = result[len(model_input_text):].strip()
    #     # else:
    #     answer = result.strip()
    #     print("Step: ",step,"/",len(train_dataloader), " ","Original Answer: ", answer)
    #     original_results.append(answer)
    
    # #save the original results to a pickle file
    # # original_results_path = os.path.join(args.model_save_path, "original_results.pkl")
    # original_results_path = "original_results.pkl"
    # with open(original_results_path, 'wb') as f:
    #     pickle.dump(original_results, f)
    # print(f"Original results saved to {original_results_path}")


    # 4. Training Loop
    print(f"Starting training for {args.num_epochs} epochs...")
    adapted_model.zero_grad() # Clear gradients before starting
    print("Num Epochs:", args.num_epochs)

    for epoch in range(args.num_epochs):
        print(f"--- Epoch {epoch+1}/{args.num_epochs} ---")
        print("Epoch: ", epoch, " Out of: ", args.num_epochs)
        epoch_total_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            if batch is None: # Skip if collate_fn returned None (e.g. empty batch after filtering)
                continue

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)


            # go through each key in batch and pass to device
            # for key in batch.keys():
            #     if isinstance(batch[key], torch.Tensor):
            #         batch[key] = batch[key].to(device)

            # Forward pass
            outputs = adapted_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            logits = outputs.logits
            #compute log softmax
            logits = torch.log_softmax(logits, dim=-1)

            # outputs_orig = original_model(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     labels=labels
            # )

            loss = compute_preference_loss(step,adapted_model, original_model, tokenizer, input_ids, attention_mask,labels, logits, args,device=device)
            # loss = outputs.loss
            # loss = outputs.loss
            
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps # Normalize loss
            
            # Backward pass
            loss.backward()
            
            # Optimizer step (with gradient accumulation)
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(adapted_model.parameters(), 1.0) # Gradient clipping
                optimizer.step()
                scheduler.step() # Update learning rate
                optimizer.zero_grad() # Clear gradients for the next accumulation

            epoch_total_loss += loss.item() * gradient_accumulation_steps # De-normalize for logging

            

            start_time = time.time()

            if (step + 1) % (logging_steps * gradient_accumulation_steps) == 0:
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                step_time = elapsed * 1000 / logging_steps  # in ms
                tokens_per_second = int(input_ids.numel() * logging_steps / elapsed)

                grad_norm = 0.0
                for p in adapted_model.parameters():
                    if p.requires_grad and p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5

                print(
                    f"step {step+1}/{len(train_dataloader)} | "
                    f"loss {loss.item() * gradient_accumulation_steps:.6f} (+nanz)| "
                    f"norm {grad_norm:.4f} (+nanz)| "
                    f"lr {current_lr:.2e} | "
                    f"{step_time:.2f} ms | "
                    f"{tokens_per_second} tok/s",
                    flush=True
                )

                start_time = time.time()
            # break # remove this later


        avg_epoch_loss = epoch_total_loss / len(train_dataloader)
        print(f"--- End of Epoch {epoch+1}, Average Loss: {avg_epoch_loss:.4f} ---")

    return adapted_model
