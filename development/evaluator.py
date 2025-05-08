import os
import gc
import json
import psutil
import torch
import shutil
import tempfile
import multiprocessing
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, pipeline

# List of general English models to use as filling models.
general_english_models = [
    "distilbert/distilbert-large-cased",
    "answerdotai/ModernBERT-base",
    "answerdotai/ModernBERT-large",
    "google-bert/bert-large-cased",
    "facebook/xlm-roberta-large",
    "roberta-large",
    "microsoft/deberta-v3-large",
    "albert-xlarge-v2"
]

# Experiment flag - set to True for a small CPU run, False for a full GPU run.
IS_EXPERIMENT = True

# Configure device and resources based on experiment mode.
if IS_EXPERIMENT:
    device = torch.device("cpu")
    print("EXPERIMENT MODE: Running on CPU with (experimentally) all dataset")
    num_cpus = multiprocessing.cpu_count()
    available_ram = psutil.virtual_memory().available / (1024 ** 3)  # in GB
    print(f"Available CPU cores: {num_cpus}")
    print(f"Available RAM: {available_ram:.2f} GB")
    BATCH_SIZE = max(1, min(16, int(available_ram / 2)))
    torch.set_num_threads(num_cpus)
    print(f"PyTorch using {torch.get_num_threads()} threads")
    SAMPLE_SIZE = None # access all data
else:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"PRODUCTION MODE: Running on GPU - {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU Memory: {gpu_mem:.2f} GB")
        BATCH_SIZE = 32
        SAMPLE_SIZE = None
    else:
        print("WARNING: GPU requested but not available, falling back to CPU")
        device = torch.device("cpu")
        num_cpus = multiprocessing.cpu_count()
        available_ram = psutil.virtual_memory().available / (1024 ** 3)
        BATCH_SIZE = max(1, min(16, int(available_ram / 2)))
        torch.set_num_threads(num_cpus)
        SAMPLE_SIZE = None

print(f"Using device: {device}")
print(f"Using batch size: {BATCH_SIZE}")

###########################
# Evaluation Models (load once)
###########################
# UnBIAS classifier.
print("Loading UnBIAS model for bias classification...")
unbias_model_name = "newsmediabias/UnBIAS-classifier"
unbias_tokenizer = AutoTokenizer.from_pretrained(unbias_model_name)
if device.type == "cuda":
    unbias_model = AutoModelForSequenceClassification.from_pretrained(
        unbias_model_name, torch_dtype=torch.float16
    ).to(device)
else:
    unbias_model = AutoModelForSequenceClassification.from_pretrained(
        unbias_model_name, low_cpu_mem_usage=True, torch_dtype=torch.float32
    ).to(device)
bias_classifier = pipeline(
    "text-classification",
    model=unbias_model,
    tokenizer=unbias_tokenizer,
    device=0 if device.type == "cuda" else -1,
    batch_size=BATCH_SIZE
)

# d4data bias detection model.
print("Loading d4data bias detection model...")
d4data_model_id = "d4data/bias-detection-model"
d4data_tokenizer = AutoTokenizer.from_pretrained(d4data_model_id)
d4data_model = AutoModelForSequenceClassification.from_pretrained(
    d4data_model_id, from_tf=True
)
bias_classifier_d4data = pipeline(
    "text-classification",
    model=d4data_model,
    tokenizer=d4data_tokenizer,
    device=0 if device.type == "cuda" else -1,
    batch_size=BATCH_SIZE
)

# ModernBERT bias type classifier.
print("Loading ModernBERT bias type classifier...")
modernbert_bias_model_id = "cirimus/modernbert-large-bias-type-classifier"
bias_classifier_modernbert = pipeline(
    "text-classification",
    model=modernbert_bias_model_id,
    return_all_scores=True,
    device=0 if device.type == "cuda" else -1,
    batch_size=BATCH_SIZE
)

# ALBERTv2 bias classifier integration.
print("Loading ALBERTv2 bias classifier model...")
albertv2_model_id = "holistic-ai/bias_classifier_albertv2"
albertv2_tokenizer = AutoTokenizer.from_pretrained(albertv2_model_id)
albertv2_model = AutoModelForSequenceClassification.from_pretrained(albertv2_model_id)
bias_classifier_albertv2 = pipeline(
    "text-classification",
    model=albertv2_model,
    tokenizer=albertv2_tokenizer,
    device=0 if device.type == "cuda" else -1,
    batch_size=BATCH_SIZE
)

###########################
# Function: Process sentences in batches to fill [MASK] tokens.
###########################
def process_sentences_in_batches(sentences, batch_size=BATCH_SIZE):
    all_filled_sentences = []
    total_sentences = len(sentences)
    for i in range(0, total_sentences, batch_size):
        end_idx = min(i + batch_size, total_sentences)
        current_batch = sentences[i:end_idx]
        print(f"Filling sentences {i+1} to {end_idx} of {total_sentences}")
        inputs = mlm_tokenizer(current_batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = mlm_model(**inputs)
        batch_results = []
        for j, sentence in enumerate(current_batch):
            if "[MASK]" not in sentence:
                batch_results.append(sentence)
                continue
            try:
                mask_token_index = inputs.input_ids[j].tolist().index(mlm_tokenizer.mask_token_id)
                predicted_token_id = outputs.logits[j, mask_token_index].argmax(dim=-1).item()
                predicted_token = mlm_tokenizer.decode([predicted_token_id])
                filled_sentence = sentence.replace("[MASK]", predicted_token)
                filled_sentence = " ".join(filled_sentence.split())
                batch_results.append(filled_sentence)
            except (ValueError, IndexError) as e:
                print(f"Error processing sentence: {sentence}")
                print(f"Error details: {e}")
                batch_results.append(sentence)
        all_filled_sentences.extend(batch_results)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return all_filled_sentences

###########################
# Main Processing: Iterate over each filling model.
###########################
def main():
    # Load the dataset once.
    print("Loading dataset...")
    raw_dataset = pd.read_csv('bias_evaluation_dataset.csv')
    mask_rows_full = raw_dataset[raw_dataset['sentence'].str.contains("[MASK]", na=False)]
    
    # Dictionary to collect results for each filling model.
    results_by_filling_model = {}
    
    # Loop over each general English model.
    for filling_model_id in general_english_models:
        print("\n====================================")
        print(f"Processing filling model: {filling_model_id}")
        
        # Create a temporary directory to download this model.
        temp_dir = tempfile.mkdtemp()
        print(f"Using temporary cache directory: {temp_dir}")
        
        try:
            # Load the filling model and tokenizer using the temporary cache.
            global mlm_tokenizer, mlm_model  # so that process_sentences_in_batches uses the correct ones
            mlm_tokenizer = AutoTokenizer.from_pretrained(filling_model_id, cache_dir=temp_dir)
            if device.type == "cuda":
                mlm_model = AutoModelForMaskedLM.from_pretrained(
                    filling_model_id, cache_dir=temp_dir, torch_dtype=torch.float16
                ).to(device)
            else:
                mlm_model = AutoModelForMaskedLM.from_pretrained(
                    filling_model_id, cache_dir=temp_dir, low_cpu_mem_usage=True, torch_dtype=torch.float32
                ).to(device)
            
            # Work on a copy of the dataset for this model.
            mask_rows = mask_rows_full.copy()
            print(f"Processing all {len(mask_rows)} sentences with [MASK] tokens")
            
            all_sentences = mask_rows['sentence'].tolist()
            # Fill the masked tokens using the current filling model.
            filled_sentences = process_sentences_in_batches(all_sentences, BATCH_SIZE)
            mask_rows['filled_sentence'] = filled_sentences

            # Group results in a nested hierarchy: category -> group -> sentence_type.
            grouped_results = {}
            if IS_EXPERIMENT:
                chunk_size = min(100, len(mask_rows))
            else:
                chunk_size = min(2000, len(mask_rows)) if device.type == "cuda" else min(1000, len(mask_rows))
            
            num_chunks = (len(mask_rows) + chunk_size - 1) // chunk_size
            print(f"Processing bias classification in {num_chunks} chunks of size {chunk_size}...")
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(mask_rows))
                print(f"Evaluating chunk {i+1}/{num_chunks} (sentences {start_idx+1} to {end_idx})")
                chunk_df = mask_rows.iloc[start_idx:end_idx].copy()
                filled_chunk = chunk_df['filled_sentence'].tolist()
                try:
                    with torch.no_grad():
                        results_unbias = bias_classifier(filled_chunk)
                        results_d4data = bias_classifier_d4data(filled_chunk)
                        results_modernbert = bias_classifier_modernbert(filled_chunk)
                        results_albertv2 = bias_classifier_albertv2(filled_chunk, return_all_scores=True)
                except Exception as e:
                    print(f"Error classifying bias in chunk {i+1}: {e}")
                    results_unbias = [{"label": "ERROR", "score": 0.0} for _ in range(len(filled_chunk))]
                    results_d4data = [{"label": "ERROR", "score": 0.0} for _ in range(len(filled_chunk))]
                    results_modernbert = [[{"label": "ERROR", "score": 0.0}] for _ in range(len(filled_chunk))]
                    results_albertv2 = [[{"label": "ERROR", "score": 0.0}] for _ in range(len(filled_chunk))]
                
                for idx in range(len(chunk_df)):
                    category = chunk_df.iloc[idx]['category']
                    group = chunk_df.iloc[idx]['group']
                    sentence_type = chunk_df.iloc[idx]['sentence_type']
                    sentence = chunk_df.iloc[idx]['sentence']
                    filled_sentence = chunk_df.iloc[idx]['filled_sentence']
                    
                    eval_unbias = {
                        "label": results_unbias[idx]['label'].upper(),
                        "score": results_unbias[idx]['score'] * 100
                    }
                    eval_d4data = {
                        "label": results_d4data[idx]['label'].upper(),
                        "score": results_d4data[idx]['score'] * 100
                    }
                    sorted_preds_modern = sorted(results_modernbert[idx], key=lambda x: x['score'], reverse=True)
                    eval_modernbert = {
                        "label": sorted_preds_modern[0]['label'].upper(),
                        "score": sorted_preds_modern[0]['score'] * 100
                    }
                    # Process ALBERTv2 results and remap the labels.
                    sorted_preds_albert = sorted(results_albertv2[idx], key=lambda x: x['score'], reverse=True)
                    label_mapping = {"STEREOTYPE": "BIASED", "NON-STEREOTYPE": "NON-BIASED"}
                    best_label_albert = sorted_preds_albert[0]['label'].upper()
                    best_label_albert = label_mapping.get(best_label_albert, best_label_albert)
                    eval_albertv2 = {
                        "label": best_label_albert,
                        "score": sorted_preds_albert[0]['score'] * 100
                    }
                    
                    evaluations = {
                        "UnBIAS": eval_unbias,
                        "d4data": eval_d4data,
                        "ModernBERT-bias": eval_modernbert,
                        "ALBERT-bias": eval_albertv2
                    }
                    
                    grouped_results.setdefault(category, {})
                    grouped_results[category].setdefault(group, {})
                    grouped_results[category][group].setdefault(sentence_type, [])
                    grouped_results[category][group][sentence_type].append({
                        "sentence": sentence,
                        "filled_sentence": filled_sentence,
                        "evaluations": evaluations
                    })
                
                del chunk_df, filled_chunk, results_unbias, results_d4data, results_modernbert, results_albertv2
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            
            # Save the grouped results under the current filling model's key.
            results_by_filling_model[filling_model_id] = grouped_results
            total_sentences = sum(
                len(sent_list)
                for cat in grouped_results.values()
                for grp in cat.values()
                for sent_list in grp.values()
            )
            print(f"Processed {total_sentences} sentences for model {filling_model_id}")
        
        except Exception as e:
            print(f"Error processing filling model {filling_model_id}: {e}")
            # If any error occurs, this filling model's results are skipped.
        
        finally:
            # Delete the downloaded model files by removing the temporary directory.
            shutil.rmtree(temp_dir)
            print(f"Deleted temporary directory: {temp_dir}")
    
    # Combine all results in the final JSON output.
    output_json = {
        "evaluations": {
            "UnBIAS": {"model": "newsmediabias/UnBIAS-classifier"},
            "d4data": {"model": "d4data/bias-detection-model"},
            "ModernBERT-bias": {"model": "cirimus/modernbert-large-bias-type-classifier"},
            "ALBERT-bias": {"model": "holistic-ai/bias_classifier_albertv2"}
        },
        "results": results_by_filling_model
    }
    
    mode_str = "experiment" if IS_EXPERIMENT else "production"
    device_str = "cpu" if device.type == "cpu" else "gpu"
    output_file = f'bias_evaluationresults_{mode_str}_{device_str}.json'
    
    with open(output_file, "w") as f:
        json.dump(output_json, f, indent=4)
    
    total_models = len(results_by_filling_model)
    print(f"Bias evaluation complete. Processed results for {total_models} filling models.")
    print(f"Results saved to '{output_file}'")

if __name__ == "__main__":
    main()