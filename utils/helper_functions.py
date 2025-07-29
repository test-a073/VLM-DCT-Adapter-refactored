import os 
import yaml 
from typing import Dict, Any , List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import torch
import json
from tqdm import tqdm
from datasets import Dataset 
import sys
from evaluator.generic_evaluator import GenericLLMEvaluator
import re

class ConfigDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)
    def __setattr__(self, name, value):
        self[name] = value


# --- Helper Functions (adapted from evaluation.py) ---
def load_openai_config(config_path: str) -> dict:
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    print(f"OpenAI config file not found at {config_path}")
    return {}

def simple_text_summarizer_postprocessor(judge_response_text: str) -> Dict[str, Any]:
    """Postprocessor to extract a score from judge response text using regex."""
    score = None
    lines = judge_response_text.strip().split('\n')
    score_keyword_pattern = r"(?:score|评[分价测]|得分)[:：]?\s*(\d+(?:\.\d+)?)(?:/\d+)?"
    standalone_score_pattern = r"(?<![a-zA-Z0-9\._-])(\b\d+(?:\.\d+)?\b)(?![a-zA-Z0-9\._-])"

    for line in reversed(lines):
        line_cleaned = line.strip()
        if not line_cleaned:
            continue
        match = re.search(score_keyword_pattern, line_cleaned, re.IGNORECASE)
        if match:
            score_str = match.group(1)
            if score_str:
                try:
                    score = float(score_str)
                    break
                except ValueError:
                    print(f"Found score-like text \"{score_str}\" with keyword but failed to parse as float.")
                    pass
        if score is not None:
            break
        if re.fullmatch(r"\d+(?:\.\d+)?", line_cleaned):
            try:
                potential_score = float(line_cleaned)
                if 0 <= potential_score <= 10:
                    score = potential_score
                    break
            except ValueError:
                pass
        if score is not None:
            break
        else:
            all_standalone_matches = list(re.finditer(standalone_score_pattern, line_cleaned))
            if all_standalone_matches:
                last_match_str = all_standalone_matches[-1].group(1)
                try:
                    potential_score = float(last_match_str)
                    if 0 <= potential_score <= 10:
                        score = potential_score
                        break
                except ValueError:
                    pass
        if score is not None:
            break
    return {"score": score, "raw_judge_response": judge_response_text}

def generate_predictions(
    model,
    tokenizer,
    dataset_split,
    device: str,
    max_new_tokens: int = 512
) -> List[Dict[str, Any]]:
    print(f"Generating predictions for {len(dataset_split)} samples...")
    predictions_data = []
    for example in tqdm(dataset_split, desc="Generating Predictions"):
        conv_id = example.get("id", "unknown_id")
        # Assuming dataset has 'query' and 'reference' (optional)
        # For mtbench, 'history' is used. We need the last user turn as query.
        history = example.get("history")
        if not history or not isinstance(history, list) or not history[-1].get("user"):
            current_prompt_text = example.get("query", "") # Fallback if history is not as expected
            if not current_prompt_text:
                 print(f"Skipping item {conv_id} due to missing user prompt in history or query field.")
                 predictions_data.append({
                    "id": conv_id, "task_category": example.get("task_category", "N/A"),
                    "model_input": "Error: Missing prompt", "prediction": "Error: Missing prompt",
                    "reference_answer": example.get("reference", "N/A"), "full_history": history
                 })
                 continue
        else:
            current_prompt_text = history[-1]["user"]

        reference_answer = history[-1].get("bot", example.get("reference", "N/A"))
        
        model_input_text = current_prompt_text

        try:
            
            inputs = tokenizer(model_input_text, return_tensors="pt", truncation=True, max_length=2048).to(device) # Added truncation
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True, # Consistent with evaluation.py
                    pad_token_id=tokenizer.eos_token_id # Add pad_token_id for open-ended generation
                )
            
            
            # Ensure decoding handles cases where input is part of the output
            # For instruct models, often the prompt is not repeated.
            # If prompt is repeated, use: result = tokenizer.decode(generated_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            # A simple way to remove prompt if it's there, more robust methods might be needed
            if result.startswith(model_input_text):
                parsed_answer = result[len(model_input_text):].strip()
            else:
                parsed_answer = result.strip()

            predictions_data.append({
                "id": conv_id,
                "task_category": example.get("task_category", "N/A"),
                "model_input": model_input_text,
                "prediction": parsed_answer,
                "reference_answer": reference_answer,
                "full_history": history
            })
        except Exception as e:
            print("Error: ",e)
            print(f"Error generating prediction for ID {conv_id}: {e}")
            predictions_data.append({
                "id": conv_id, "task_category": example.get("task_category", "N/A"),
                "model_input": model_input_text, "prediction": f"Error: {e}",
                "reference_answer": reference_answer, "full_history": history
            })
    return predictions_data

def run_evaluation_pipeline(
    model_name_or_path: str,
    model_to_evaluate: AutoModelForCausalLM, # Pass the loaded model
    tokenizer: AutoTokenizer, # Pass the loaded tokenizer
    eval_dataset,
    args: argparse.Namespace,
    output_suffix: str = ""
) -> Optional[float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_to_evaluate.to(device)
    model_to_evaluate.eval()

    # 1. Generate Predictions
    predictions_output_filename = f"predictions_{output_suffix}.jsonl"
    predictions_output_filepath = os.path.join(args.eval_output_dir, predictions_output_filename)
    
    generated_preds_list = generate_predictions(model_to_evaluate, tokenizer, eval_dataset, device, args.max_new_tokens)
    
    with open(predictions_output_filepath, 'w') as f:
        for item in generated_preds_list:
            f.write(json.dumps(item) + '\n')
    print(f"Predictions for {output_suffix} saved to {predictions_output_filepath}")

    # 2. Prepare dataset for evaluator
    dataset_for_eval_dict = {
        "query": [], "prediction": [], "reference": [], "id": [], "task_category": []
    }
    valid_predictions_count = 0
    for item in generated_preds_list:
        if not item["prediction"].startswith("Error:"):
            dataset_for_eval_dict["query"].append(item["model_input"])
            dataset_for_eval_dict["prediction"].append(item["prediction"])
            ref = item["reference_answer"]
            if isinstance(ref, list): # Ensure reference is a string
                ref = " ".join(r for r in ref if isinstance(r, str)) if all(isinstance(r, str) for r in ref) else str(ref)
            dataset_for_eval_dict["reference"].append(ref)
            dataset_for_eval_dict["id"].append(item["id"])
            dataset_for_eval_dict["task_category"].append(item.get("task_category", "N/A"))
            valid_predictions_count += 1
    
    if valid_predictions_count == 0:
        print(f"No successful predictions to evaluate for {output_suffix}. Skipping evaluation.")
        final_score_value = "N/A (No valid predictions)"
    else:
        eval_hf_dataset = Dataset.from_dict(dataset_for_eval_dict)
        print(f"Prepared {len(eval_hf_dataset)} samples for the evaluator for {output_suffix}.")

        # 3. Configure and Run Evaluator
        openai_params = load_openai_config(args.openai_config_path)
        judge_cfg_dict = {
            "model": args.judge_model_name,
            "key": openai_params.get("api_key"),
            "openai_api_base": openai_params.get("base_url"),
            "temperature": 0.0, "max_out_len": 1024, "query_per_second": 1,
            "system_prompt_content": args.judge_system_prompt
        }
        prompt_template_dict = {
            "template": "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. "
                         "Your evaluation should consider factors such as helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. "
                         "Begin your evaluation by providing a short explanation. Be as objective as possible. "
                         "After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly outputting a single line with only the score. "
                         "Do not output any other text after the score. "
                         "\n\n[Question]\n{query}\n\n[The Start of Assistant's Answer]\n{prediction}\n[The End of Assistant's Answer]"
                         "\n\n[Reference Answer (if available)]\n{reference}\n[The End of Reference Answer]",
            "input_columns": ["query", "prediction", "reference"],
        }
        evaluator_results_filename = f"{args.evaluator_type}_results_{output_suffix}.json"
        evaluator_output_path = os.path.join(args.eval_output_dir, evaluator_results_filename)

        judge_config = ConfigDict(judge_cfg_dict)
        prompt_template_config = ConfigDict(prompt_template_dict)

        evaluator = GenericLLMEvaluator(
            judge_cfg=judge_config,
            prompt_template=prompt_template_config,
            dict_postprocessor=simple_text_summarizer_postprocessor,
            output_path=evaluator_output_path
        )

        print(f"Running evaluation with {args.evaluator_type} for {output_suffix}...")
        evaluation_results = evaluator.score(
            predictions=list(eval_hf_dataset["prediction"]),
            test_set=eval_hf_dataset
        )
        print(f"Raw Evaluation Results for {output_suffix}:")
        try:
            print(json.dumps(evaluation_results, indent=4))
        except TypeError:
            print(str(evaluation_results))

        final_score_value = "N/A"
        if isinstance(evaluation_results, dict) and "average_score" in evaluation_results:
            final_score_value = evaluation_results["average_score"]
            num_scored = evaluation_results.get('num_scored', 'N/A')
            print(f"Average Judge Score for {output_suffix}: {final_score_value:.2f} (Scored items: {num_scored})")
        else:
            print(f"Could not determine average_score from evaluation results for {output_suffix}.")


    # 4. Save Score File
    score_file_name = f"{model_name_or_path.replace('/', '_')}_{output_suffix}_score.txt"
    score_file_path = os.path.join(args.eval_output_dir, score_file_name)
    try:
        with open(score_file_path, 'w') as f:
            f.write(f"Model: {model_name_or_path} ({output_suffix})\n")
            f.write(f"Training Dataset: {args.train_dataset_path}\n")
            f.write(f"Evaluation Dataset: {args.eval_dataset_path}\n")
            f.write(f"Evaluator: {args.evaluator_type}\n")
            f.write(f"Judge Model: {args.judge_model_name}\n")
            f.write(f"Evaluation Split Size: {len(eval_dataset) if eval_dataset is not None else 'N/A'}\n")
            f.write(f"Final Score: {final_score_value}\n")
        print(f"Evaluation score for {output_suffix} saved to {score_file_path}")
    except Exception as e:
        print(f"Failed to write score to file {score_file_path}: {e}")
    
    if isinstance(final_score_value, (float, int)):
        return float(final_score_value)
    return None
