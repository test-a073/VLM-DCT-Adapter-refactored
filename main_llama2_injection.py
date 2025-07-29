# TODO: I want to inject the adapter to LLAma 2 - 7B
import argparse
import json
import os
import yaml
import logging
from datasets import Dataset
import sys
import gc


import torch
from prettyprinter import pprint

# from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"  # Set to the GPU you want to use, or leave empty for all available GPUs
# from evaluator.cascade_evaluator import CascadeEvaluator # If needed

# # Import for Adapter
from adapter.adapter import DCTAdapter # Assuming DCTAdapter is in adapter/adapter.py
from adapter.adapter_helper_functions import inject_adapters, get_parent_module
from runner.train import train_model, freeze_model_except_adapters
from runner.train import train_model_mistral , train_model_adapted_mistral, generate_predictions_for_original_model, train_model_adapted_llama_2
    

from utils.helper_functions import load_openai_config, simple_text_summarizer_postprocessor, generate_predictions, run_evaluation_pipeline

# --- Main Script Logic ---
def main(DEBUG=False):

    # Load the layers to train
    import yaml
    model_config_file_path = "config/llama_2_config.yaml"

    with open(model_config_file_path, 'r') as f:
        config_data = yaml.safe_load(f)

    pprint("Config data")
    pprint( config_data)

    parser = argparse.ArgumentParser(description="Evaluate original and adapted LLM models.")
    args = parser.parse_args()
    
    args.model_name = config_data.get("models").get("name")
    args.train_dataset_path = config_data.get("train").get("dataset_path")
    args.eval_dataset_path = config_data.get("eval").get("dataset_path")
    args.num_train_samples = config_data.get("train").get("num_train_samples")
    args.num_eval_samples = config_data.get("eval").get("num_eval_samples")
    args.max_new_tokens = config_data.get("generation").get("max_new_tokens")


    args.eval_output_dir = config_data.get("eval").get("output_dir")
    args.openai_config_path = config_data.get("openai").get("config_path")

    args.evaluator_type = config_data.get("evaluator").get("evaluator_type")
    args.judge_model_name = config_data.get("evaluator").get("judge_model_name")
    
    args.judge_system_prompt = None
    
    args.batch_size = config_data.get("train").get("batch_size")
    args.num_epochs = config_data.get("train").get("num_epochs")

    # Arguments for Adapter Injection
    args.do_adapter_injection = config_data.get("adapter").get("do_adapt")
    args.adapter_layers_json = config_data.get("adapter").get("layers")
    args.adapter_params_json = config_data.get("adapter").get("params")
    args.perform_adapter_training = config_data.get("adapter").get("perform_adapter_training")

    # TODO: Check what following does
    args.adapter_lr = 1e-4
    args.learning_rate = 1e-4
    

    # New argument for full fine-tuning
    args.perform_full_finetune = config_data.get("train").get("perform_full_finetune")
    args.model_save_path = config_data.get("save").get("model_save_path")
    args.max_seq_length = 1024

    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.eval_output_dir, exist_ok=True)

    with open(args.openai_config_path, "r") as file:
        config = yaml.safe_load(file)

    opeanai_api_key = config.get("OPENAI_API_KEY")
    if not opeanai_api_key:
        raise ValueError("API key not found in config.")

    # Set environment variable
    os.environ["OPENAI_API_KEY"] = opeanai_api_key

    # Load and prepare train_dataset
    print(f"Loading training dataset from {args.train_dataset_path}...")
    try:
        train_dataset_list = []
        with open(args.train_dataset_path, 'r') as f:
            for line in f:
                train_dataset_list.append(json.loads(line))
        
        if args.num_train_samples != -1 and args.num_train_samples < len(train_dataset_list): # If only fraction of train dataset is used
            train_dataset_list = train_dataset_list[:args.num_train_samples]

        train_dataset = Dataset.from_list(train_dataset_list)
        if DEBUG:
            print("[Train dataset]", train_dataset)
            print("[One sample from train dataset]")
            print(train_dataset[0])
        print(f"Training dataset loaded. Samples: {len(train_dataset)}")

    except Exception as e:
        print(f"Error loading training dataset: {e}")
        return

    # Load and prepare eval_dataset
    print(f"Loading evaluation dataset from {args.eval_dataset_path}...")
    try:
        eval_dataset_list = []
        with open(args.eval_dataset_path, 'r') as f:
            for line in f:
                eval_dataset_list.append(json.loads(line))

        if args.num_eval_samples != -1 and args.num_eval_samples < len(eval_dataset_list):
            eval_dataset_list = eval_dataset_list[:args.num_eval_samples]

        eval_dataset = Dataset.from_list(eval_dataset_list)

        if DEBUG:
            print("[Eval dataset]", eval_dataset)
            print("[One sample from eval dataset]")
            print(eval_dataset[0])
        print(f"Evaluation dataset loaded. Samples: {len(eval_dataset)}")

    except Exception as e:
        print(f"Error loading evaluation dataset: {e}")
        return

    # 2. Load Tokenizer (shared for both models)
    try:
        if args.model_name in ["meta-llama/Llama-2-7b-chat-hf"]:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side='left')
            if DEBUG:
                print(f"Tokenizer is loaded")
    except Exception as e:
        print(f"Failed to load tokenizer for {args.model_name}: {e}")
        raise e
    
        
    # --- Adapter Injection and Evaluation for Adapted Model ---
    print(f"--- Starting Evaluation for Adapted Model: {args.model_name} + Adapter ---")
    adapted_model = None 
    original_model = None
    model_for_adapted_eval_name = args.model_name 
    
    try:
        if args.do_adapter_injection:
            print(f"Loading base model ({args.model_name}) for adapter injection...")
            # original_model = AutoModelForCausalLM.from_pretrained(args.model_name)

            torch.cuda.empty_cache()
            gc.collect()
            original_model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.float16, 
                device_map="cuda"
                ).eval()
            adapted_model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.float16, 
                device_map="cuda"
                ) 
            if DEBUG: 
                print("> Model loaded")
            model_name_  = args.model_name.split("/")[1] # model name without organization name
            
            model_for_adapted_eval_name = f"{model_name_}_adapted"
            model_arch_path = os.path.join(args.model_save_path, f"{model_name_}_architecture.txt")
            try:
                with open(model_arch_path, 'w') as f:
                    f.write(str(adapted_model))
                if DEBUG:
                    print(f"Model architecture written to {model_arch_path}")
            except Exception as e:
                print(f"Failed to write model architecture: {e}")
            
            try: 
                if DEBUG:
                    print(f"{args.adapter_layers_json=}")
                    print(f"{args.adapter_params_json=}")
                    
                adapted_model = inject_adapters(
                    adapted_model, 
                    DCTAdapter, 
                    args.adapter_params_json, 
                    args.adapter_layers_json
                )
                
                if DEBUG:   
                    print("Adapter injection process finished.")
                freeze_model_except_adapters(adapted_model)
                
                if DEBUG:
                    print(">Non-adapter layers are frozn")
                print("[ADAPTED MODEL ARCHITECTURE]\n")
                print(adapted_model)

                # print trainable parameters number.
                trainable_params = sum(p.numel() for p in adapted_model.parameters() if p.requires_grad)
                print(f"Number of trainable parameters in the adapted model: {trainable_params}")

                trainable_params_original = sum(p.numel() for p in original_model.parameters() if p.requires_grad)
                print(f"Number of trainable parameters in the original model: {trainable_params_original}")

            except Exception as e:
                print(e)

            adapted_model = adapted_model.half() # for the injected layers also to have torch.float16 dtype

            if args.perform_adapter_training:
                if train_dataset is None or len(train_dataset) == 0:
                    print("Adapter training requested, but train_dataset is empty or None. Skipping training.")
                else:
                    print(f"Performing adapter training using {len(train_dataset)} samples...")
                    
                    if args.model_name == "meta-llama/Llama-2-7b-chat-hf":
                        adapted_model = train_model_adapted_llama_2(adapted_model, original_model, tokenizer, train_dataset, args, DEBUG=DEBUG)
                    elif args.model_name == "mistral-model-name # TODO:change this later":
                        adapted_model = train_model_adapted_mistral(adapted_model, original_model, tokenizer, train_dataset, args)

                    print("Adapters training finished.")
                    

                    # TODO: Uncomment this; I got disk space exceeded IO error 
                    
                    # # Save the adapter-injected model (adapter weights)
                    # finetuned_adapter_dir = os.path.join(args.model_save_path, "finetuned_adapter_model")
                    # os.makedirs(finetuned_adapter_dir, exist_ok=True)
                   
                    # # Save the full model (including adapters)
                    # adapted_model.save_pretrained(finetuned_adapter_dir)
                    # tokenizer.save_pretrained(finetuned_adapter_dir)
                    # print(f"Adapter-injected model saved to {finetuned_adapter_dir}")
            else:
                print("Adapter training not requested (perform_adapter_training=False).")
        else:
            print("Adapter injection not requested (do_adapter_injection=False). Evaluating base model as 'adapted' model.")
        
        gc.collect()
        torch.cuda.empty_cache()

        
        if adapted_model: 
            print("Evaluation of adapted model--------------------")
            run_evaluation_pipeline(
                model_name_or_path=model_for_adapted_eval_name, 
                model_to_evaluate=adapted_model,
                tokenizer=tokenizer,
                eval_dataset=eval_dataset,
                args=args,
                output_suffix="adapted" 
            )
        else:
            print("Adapted model was not loaded or created. Skipping evaluation for adapted model.")
        # del adapted_model 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error during adapted model setup or evaluation: {e}")

    print(f"--- Main {args.model_name} Adapter Injection Script Finished ---")

if __name__ == "__main__":
    main(DEBUG=True)