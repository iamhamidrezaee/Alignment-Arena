import os
import gc
import json
import psutil
import torch
import shutil
import tempfile
import multiprocessing
import pandas as pd
import time
import logging
import traceback
import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, pipeline
from transformers.utils import logging as transformers_logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_loading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# Reduce verbosity of transformers warnings
transformers_logging.set_verbosity_error()

# List of general English models to use as filling models with corrected identifiers.
general_english_models = [
    "FacebookAI/xlm-roberta-large",  # Corrected from facebook/xlm-roberta-large
    "google-bert/bert-large-cased",
    "answerdotai/ModernBERT-large",
    "albert/albert-xlarge-v2",  # Correctly formatted
    "distilbert/distilbert-base-cased",  # Changed from large to base as large doesn't exist
]

# Model-specific configurations including mask token handling
model_configs = {
    "FacebookAI/xlm-roberta-large": {
        "memory_required": 2.0,  # GB
        "mask_token": "<mask>",  # XLM-RoBERTa uses <mask> instead of [MASK]
        "mask_conversion": True,  # Convert [MASK] to <mask> before tokenizing
    },
    "albert/albert-xlarge-v2": {
        "memory_required": 1.5,
        "mask_token": "[MASK]",
        "mask_conversion": False,
    },
    "distilbert/distilbert-base-cased": {
        "memory_required": 0.5,
        "mask_token": "[MASK]",
        "mask_conversion": False,
    },
    "answerdotai/ModernBERT-large": {
        "memory_required": 2.5,
        "mask_token": "[MASK]",
        "mask_conversion": False,
    },
    "google-bert/bert-large-cased": {
        "memory_required": 1.5,
        "mask_token": "[MASK]",
        "mask_conversion": False,
    }
}

# Experiment flag - set to True for a small CPU run, False for a full GPU run.
IS_EXPERIMENT = True

# Configure device and resources based on experiment mode.
if IS_EXPERIMENT:
    device = torch.device("cpu")
    logger.info("EXPERIMENT MODE: Running on CPU with (experimentally) all dataset")
    num_cpus = multiprocessing.cpu_count()
    available_ram = psutil.virtual_memory().available / (1024 ** 3)  # in GB
    logger.info(f"Available CPU cores: {num_cpus}")
    logger.info(f"Available RAM: {available_ram:.2f} GB")
    BATCH_SIZE = max(1, min(8, int(available_ram / 4)))  # Reduced batch size to prevent OOM
    torch.set_num_threads(num_cpus)
    logger.info(f"PyTorch using {torch.get_num_threads()} threads")
    SAMPLE_SIZE = None  # access all data
else:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"PRODUCTION MODE: Running on GPU - {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"GPU Memory: {gpu_mem:.2f} GB")
        BATCH_SIZE = 32
        SAMPLE_SIZE = None
    else:
        logger.warning("WARNING: GPU requested but not available, falling back to CPU")
        device = torch.device("cpu")
        num_cpus = multiprocessing.cpu_count()
        available_ram = psutil.virtual_memory().available / (1024 ** 3)
        BATCH_SIZE = max(1, min(8, int(available_ram / 4)))
        torch.set_num_threads(num_cpus)
        SAMPLE_SIZE = None

logger.info(f"Using device: {device}")
logger.info(f"Using batch size: {BATCH_SIZE}")

###########################
# Helper Functions
###########################
def check_memory_available():
    """Check if enough memory is available and return available memory in GB."""
    if device.type == "cuda":
        # Get available CUDA memory
        torch.cuda.empty_cache()
        available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        available_memory_gb = available_memory / (1024 ** 3)
    else:
        # Get available system RAM
        available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
    
    return available_memory_gb

def can_load_model(model_id):
    """Check if there's enough memory to load a particular model."""
    required_memory = model_configs.get(model_id, {}).get("memory_required", 2.0)  # Default to 2GB if unknown
    available_memory = check_memory_available()
    
    logger.info(f"Model {model_id} requires ~{required_memory:.2f} GB, {available_memory:.2f} GB available")
    return available_memory >= required_memory * 1.2  # Add 20% buffer

def cleanup_gpu_memory():
    """Force garbage collection and empty CUDA cache if available."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

def safely_remove_temp_dir(temp_dir):
    """Safely remove a temporary directory with retries."""
    max_retries = 5
    retry_delay = 1  # seconds

    for attempt in range(max_retries):
        try:
            if os.path.exists(temp_dir):
                if os.name == 'nt':  # Windows
                    # On Windows, try to remove readonly attributes
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            try:
                                os.chmod(file_path, 0o777)
                            except:
                                pass
                
                shutil.rmtree(temp_dir, ignore_errors=True)
            
            if not os.path.exists(temp_dir):
                logger.info(f"Successfully removed temporary directory: {temp_dir}")
                return True
                
        except Exception as e:
            logger.warning(f"Failed to remove temp directory on attempt {attempt+1}: {e}")
        
        # Wait before retry
        time.sleep(retry_delay)
    
    logger.error(f"Failed to remove temp directory after {max_retries} attempts: {temp_dir}")
    return False

###########################
# Evaluation Models (load once)
###########################
def load_evaluation_models():
    """Load all evaluation models with proper error handling."""
    models = {}
    
    # Load UnBIAS classifier
    try:
        logger.info("Loading UnBIAS model for bias classification...")
        unbias_model_name = "newsmediabias/UnBIAS-classifier"
        unbias_tokenizer = AutoTokenizer.from_pretrained(unbias_model_name)
        logger.info("Device set to use %s", device)
        
        if device.type == "cuda":
            unbias_model = AutoModelForSequenceClassification.from_pretrained(
                unbias_model_name, torch_dtype=torch.float16
            ).to(device)
        else:
            unbias_model = AutoModelForSequenceClassification.from_pretrained(
                unbias_model_name, low_cpu_mem_usage=True, torch_dtype=torch.float32
            ).to(device)
            
        models['unbias_classifier'] = pipeline(
            "text-classification",
            model=unbias_model,
            tokenizer=unbias_tokenizer,
            device=0 if device.type == "cuda" else -1,
            batch_size=BATCH_SIZE
        )
    except Exception as e:
        logger.error(f"Failed to load UnBIAS model: {e}")
        models['unbias_classifier'] = None

    # Load d4data bias detection model
    try:
        logger.info("Loading d4data bias detection model...")
        d4data_model_id = "d4data/bias-detection-model"
        d4data_tokenizer = AutoTokenizer.from_pretrained(d4data_model_id)
        
        # IMPORTANT FIX: Use TFAutoModelForSequenceClassification instead of AutoModelForSequenceClassification
        # Import the TF model class
        from transformers import TFAutoModelForSequenceClassification
        
        # Load the model in its native TensorFlow format
        d4data_model = TFAutoModelForSequenceClassification.from_pretrained(d4data_model_id)
        
        # Create the pipeline with the TF model
        models['bias_classifier_d4data'] = pipeline(
            "text-classification",
            model=d4data_model,
            tokenizer=d4data_tokenizer,
            device=0 if device.type == "cuda" else -1,
            batch_size=BATCH_SIZE
        )
    except Exception as e:
        logger.error(f"Failed to load d4data model: {e}")

    # Load ModernBERT bias type classifier
    try:
        logger.info("Loading ModernBERT bias type classifier...")
        modernbert_bias_model_id = "cirimus/modernbert-large-bias-type-classifier"
        models['bias_classifier_modernbert'] = pipeline(
            "text-classification",
            model=modernbert_bias_model_id,
            # Using top_k instead of deprecated return_all_scores
            top_k=None,
            device=0 if device.type == "cuda" else -1,
            batch_size=BATCH_SIZE
        )
    except Exception as e:
        logger.error(f"Failed to load ModernBERT bias classifier: {e}")
        models['bias_classifier_modernbert'] = None

    # Load ALBERTv2 bias classifier
    try:
        logger.info("Loading ALBERTv2 bias classifier model...")
        albertv2_model_id = "holistic-ai/bias_classifier_albertv2"
        albertv2_tokenizer = AutoTokenizer.from_pretrained(albertv2_model_id)
        albertv2_model = AutoModelForSequenceClassification.from_pretrained(albertv2_model_id)
        models['bias_classifier_albertv2'] = pipeline(
            "text-classification",
            model=albertv2_model,
            tokenizer=albertv2_tokenizer,
            device=0 if device.type == "cuda" else -1,
            batch_size=BATCH_SIZE,
            # Using top_k instead of deprecated return_all_scores
            top_k=None
        )
    except Exception as e:
        logger.error(f"Failed to load ALBERTv2 bias classifier: {e}")
        models['bias_classifier_albertv2'] = None
    
    return models

###########################
# Function: Process sentences in batches to fill [MASK] tokens with model-specific handling
###########################
def process_sentences_in_batches(sentences, mlm_tokenizer, mlm_model, model_id, batch_size=1):
    """Process sentences in batches to fill [MASK] tokens with model-specific handling."""
    all_filled_sentences = []
    total_sentences = len(sentences)
    
    # Get model config for mask token handling
    model_config = model_configs.get(model_id, {})
    mask_token = model_config.get("mask_token", "[MASK]")
    needs_conversion = model_config.get("mask_conversion", False)
    
    # Process one sentence at a time to avoid batch issues
    for i in range(total_sentences):
        logger.info(f"Filling sentences {i+1} to {i+1} of {total_sentences}")
        
        sentence = sentences[i]
        original_sentence = sentence
        
        # Skip if no mask token
        if "[MASK]" not in sentence and "<mask>" not in sentence:
            all_filled_sentences.append(sentence)
            continue
        
        try:
            # Convert [MASK] to the model's mask token if needed
            if needs_conversion:
                sentence = sentence.replace("[MASK]", mask_token)
            
            # XLM-RoBERTa specific handling - find the actual mask token position
            if "FacebookAI/xlm-roberta" in model_id:
                # For XLM-RoBERTa, tokenize with return_offsets_mapping to find the mask token
                encoding = mlm_tokenizer(sentence, return_tensors="pt", return_offsets_mapping=True)
                
                # Find where <mask> is in the original sentence
                mask_pos = sentence.find(mask_token)
                if mask_pos != -1:
                    # Find which token contains this position
                    offsets = encoding.offset_mapping[0].tolist()
                    mask_token_index = None
                    for idx, (start, end) in enumerate(offsets):
                        if start <= mask_pos < end:
                            mask_token_index = idx
                            break
                    
                    if mask_token_index is not None:
                        # Get predictions
                        with torch.no_grad():
                            outputs = mlm_model(**{k: v for k, v in encoding.items() if k != 'offset_mapping'})
                        
                        # Get the predicted token
                        predicted_token_id = outputs.logits[0, mask_token_index].argmax(dim=-1).item()
                        predicted_token = mlm_tokenizer.decode([predicted_token_id])
                        
                        # Replace <mask> with predicted token
                        filled_sentence = sentence.replace(mask_token, predicted_token)
                        filled_sentence = " ".join(filled_sentence.split())
                        all_filled_sentences.append(filled_sentence)
                    else:
                        logger.error(f"Could not find mask token position in offsets for: {sentence}")
                        all_filled_sentences.append(original_sentence)
                else:
                    logger.error(f"Could not find mask token in sentence: {sentence}")
                    all_filled_sentences.append(original_sentence)
            
            # Standard processing for other models
            else:
                inputs = mlm_tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)
                
                with torch.no_grad():
                    outputs = mlm_model(**inputs)
                
                # Find the mask token index - handle different mask tokens
                input_ids = inputs.input_ids[0].tolist()
                mask_token_id = mlm_tokenizer.mask_token_id
                
                if mask_token_id in input_ids:
                    mask_token_index = input_ids.index(mask_token_id)
                    
                    # Get prediction
                    predicted_token_id = outputs.logits[0, mask_token_index].argmax(dim=-1).item()
                    predicted_token = mlm_tokenizer.decode([predicted_token_id])
                    
                    # Replace the appropriate mask token
                    if needs_conversion:
                        filled_sentence = sentence.replace(mask_token, predicted_token)
                    else:
                        filled_sentence = original_sentence.replace("[MASK]", predicted_token)
                    
                    filled_sentence = " ".join(filled_sentence.split())
                    all_filled_sentences.append(filled_sentence)
                else:
                    logger.error(f"Mask token ID {mask_token_id} not found in: {input_ids}")
                    all_filled_sentences.append(original_sentence)
        
        except Exception as e:
            logger.error(f"Error processing sentence: {sentence}")
            logger.error(f"Error details: {e}")
            logger.error(traceback.format_exc())
            all_filled_sentences.append(original_sentence)
        
        # Clean up memory after each sentence for larger models
        if i % 10 == 0:
            cleanup_gpu_memory()
    
    return all_filled_sentences

###########################
# Main Processing: Iterate over each filling model.
###########################
def process_model(filling_model_id, mask_rows_full, evaluation_models, results_by_filling_model):
    """Process a single model with error handling and resource management."""
    logger.info("\n====================================")
    logger.info(f"Processing filling model: {filling_model_id}")
    
    # Check if we have enough memory to load this model
    if not can_load_model(filling_model_id):
        logger.warning(f"Insufficient memory to load {filling_model_id}. Skipping.")
        results_by_filling_model[filling_model_id] = {
            "status": "skipped",
            "reason": "insufficient_memory",
            "results": {}
        }
        return
    
    # Create a dedicated temporary directory for this model
    temp_dir = None
    try:
        # Create a temporary directory with a unique path in a system-appropriate location
        temp_dir = tempfile.mkdtemp(prefix=f"hf_model_{filling_model_id.replace('/', '_')}_")
        logger.info(f"Using temporary cache directory: {temp_dir}")
        
        # Try loading the model with multiple fallback options
        mlm_tokenizer = None
        mlm_model = None
        
        # First try the preferred method with trusted models
        try:
            logger.info(f"Loading tokenizer for {filling_model_id}")
            mlm_tokenizer = AutoTokenizer.from_pretrained(filling_model_id, cache_dir=temp_dir)
            
            logger.info(f"Loading model for {filling_model_id}")
            loading_args = {
                "cache_dir": temp_dir,
                "low_cpu_mem_usage": True,
            }
            
            if device.type == "cuda":
                loading_args["torch_dtype"] = torch.float16
            else:
                loading_args["torch_dtype"] = torch.float32
            
            mlm_model = AutoModelForMaskedLM.from_pretrained(filling_model_id, **loading_args).to(device)
                
        except Exception as e:
            # Try with trust_remote_code for custom models
            logger.warning(f"First loading attempt failed for {filling_model_id}: {e}")
            logger.info(f"Trying with trust_remote_code=True for {filling_model_id}")
            
            try:
                mlm_tokenizer = AutoTokenizer.from_pretrained(
                    filling_model_id, 
                    cache_dir=temp_dir, 
                    trust_remote_code=True
                )
                mlm_model = AutoModelForMaskedLM.from_pretrained(
                    filling_model_id,
                    cache_dir=temp_dir,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float32 if device.type == "cpu" else torch.float16,
                    trust_remote_code=True
                ).to(device)
                
            except Exception as inner_e:
                # Last attempt - try with revision="main" for some models
                logger.warning(f"Second loading attempt failed for {filling_model_id}: {inner_e}")
                logger.info(f"Trying with revision='main' for {filling_model_id}")
                
                try:
                    mlm_tokenizer = AutoTokenizer.from_pretrained(
                        filling_model_id, 
                        cache_dir=temp_dir,
                        revision="main"
                    )
                    mlm_model = AutoModelForMaskedLM.from_pretrained(
                        filling_model_id,
                        cache_dir=temp_dir,
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.float32 if device.type == "cpu" else torch.float16,
                        revision="main"
                    ).to(device)
                except Exception as third_e:
                    # All attempts failed
                    logger.error(f"Failed to load model {filling_model_id} after multiple attempts: {third_e}")
                    results_by_filling_model[filling_model_id] = {
                        "status": "failed",
                        "reason": str(third_e),
                        "results": {}
                    }
                    return
        
        # Check if we got both tokenizer and model
        if mlm_tokenizer is None or mlm_model is None:
            logger.error(f"Failed to load either tokenizer or model for {filling_model_id}")
            results_by_filling_model[filling_model_id] = {
                "status": "failed",
                "reason": "Failed to load tokenizer or model",
                "results": {}
            }
            return
        
        # Special case for XLM-RoBERTa - verify mask token
        if "FacebookAI/xlm-roberta" in filling_model_id:
            logger.info(f"XLM-RoBERTa detected. Mask token: {mlm_tokenizer.mask_token}")
            logger.info(f"Mask token ID: {mlm_tokenizer.mask_token_id}")
        
        # Work on a copy of the dataset for this model
        mask_rows = mask_rows_full.copy()
        logger.info(f"Processing all {len(mask_rows)} sentences with [MASK] tokens")
        
        # Fill the masked tokens using the current filling model - process one at a time for robustness
        all_sentences = mask_rows['sentence'].tolist()
        filled_sentences = process_sentences_in_batches(all_sentences, mlm_tokenizer, mlm_model, filling_model_id, 1)
        mask_rows['filled_sentence'] = filled_sentences
        
        # Free up memory from the model
        del mlm_model, mlm_tokenizer
        cleanup_gpu_memory()
        
        # Group results in a nested hierarchy: category -> group -> sentence_type
        grouped_results = {}
        
        # Determine chunk size based on available resources
        if IS_EXPERIMENT:
            chunk_size = min(50, len(mask_rows))  # Use smaller chunks in experiment mode
        else:
            chunk_size = min(500, len(mask_rows)) if device.type == "cuda" else min(200, len(mask_rows))
        
        num_chunks = (len(mask_rows) + chunk_size - 1) // chunk_size
        logger.info(f"Processing bias classification in {num_chunks} chunks of size {chunk_size}...")
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(mask_rows))
            logger.info(f"Evaluating chunk {i+1}/{num_chunks} (sentences {start_idx+1} to {end_idx})")
            
            chunk_df = mask_rows.iloc[start_idx:end_idx].copy()
            filled_chunk = chunk_df['filled_sentence'].tolist()
            
            # Evaluate sentences with available evaluation models
            try:
                with torch.no_grad():
                    results_unbias = (evaluation_models.get('unbias_classifier', None)(filled_chunk) 
                                      if evaluation_models.get('unbias_classifier') else 
                                      [{"label": "NO_MODEL", "score": 0.0} for _ in range(len(filled_chunk))])
                    
                    results_d4data = (evaluation_models.get('bias_classifier_d4data', None)(filled_chunk) 
                                      if evaluation_models.get('bias_classifier_d4data') else 
                                      [{"label": "NO_MODEL", "score": 0.0} for _ in range(len(filled_chunk))])
                    
                    results_modernbert = (evaluation_models.get('bias_classifier_modernbert', None)(filled_chunk) 
                                         if evaluation_models.get('bias_classifier_modernbert') else 
                                         [[{"label": "NO_MODEL", "score": 0.0}] for _ in range(len(filled_chunk))])
                    
                    results_albertv2 = (evaluation_models.get('bias_classifier_albertv2', None)(filled_chunk, top_k=None) 
                                       if evaluation_models.get('bias_classifier_albertv2') else 
                                       [[{"label": "NO_MODEL", "score": 0.0}] for _ in range(len(filled_chunk))])
            except Exception as e:
                logger.error(f"Error classifying bias in chunk {i+1}: {e}")
                results_unbias = [{"label": "ERROR", "score": 0.0} for _ in range(len(filled_chunk))]
                results_d4data = [{"label": "ERROR", "score": 0.0} for _ in range(len(filled_chunk))]
                results_modernbert = [[{"label": "ERROR", "score": 0.0}] for _ in range(len(filled_chunk))]
                results_albertv2 = [[{"label": "ERROR", "score": 0.0}] for _ in range(len(filled_chunk))]
            
            # Process each sentence in the chunk
            for idx in range(len(chunk_df)):
                category = chunk_df.iloc[idx]['category']
                group = chunk_df.iloc[idx]['group']
                sentence_type = chunk_df.iloc[idx]['sentence_type']
                sentence = chunk_df.iloc[idx]['sentence']
                filled_sentence = chunk_df.iloc[idx]['filled_sentence']
                
                # Process evaluation results
                eval_unbias = {
                    "label": results_unbias[idx]['label'].upper(),
                    "score": results_unbias[idx]['score'] * 100
                }
                
                eval_d4data = {
                    "label": results_d4data[idx]['label'].upper(),
                    "score": results_d4data[idx]['score'] * 100
                }
                
                # Handle ModernBERT results
                try:
                    sorted_preds_modern = sorted(results_modernbert[idx], key=lambda x: x['score'], reverse=True)
                    eval_modernbert = {
                        "label": sorted_preds_modern[0]['label'].upper(),
                        "score": sorted_preds_modern[0]['score'] * 100
                    }
                except (IndexError, KeyError) as e:
                    logger.warning(f"Error processing ModernBERT results: {e}")
                    eval_modernbert = {"label": "ERROR", "score": 0.0}
                
                # Process ALBERTv2 results and remap the labels
                try:
                    sorted_preds_albert = sorted(results_albertv2[idx], key=lambda x: x['score'], reverse=True)
                    label_mapping = {"STEREOTYPE": "BIASED", "NON-STEREOTYPE": "NON-BIASED"}
                    best_label_albert = sorted_preds_albert[0]['label'].upper()
                    best_label_albert = label_mapping.get(best_label_albert, best_label_albert)
                    eval_albertv2 = {
                        "label": best_label_albert,
                        "score": sorted_preds_albert[0]['score'] * 100
                    }
                except (IndexError, KeyError) as e:
                    logger.warning(f"Error processing ALBERTv2 results: {e}")
                    eval_albertv2 = {"label": "ERROR", "score": 0.0}
                
                # Collect all evaluations
                evaluations = {
                    "UnBIAS": eval_unbias,
                    "d4data": eval_d4data,
                    "ModernBERT-bias": eval_modernbert,
                    "ALBERT-bias": eval_albertv2
                }
                
                # Add to grouped results
                grouped_results.setdefault(category, {})
                grouped_results[category].setdefault(group, {})
                grouped_results[category][group].setdefault(sentence_type, [])
                grouped_results[category][group][sentence_type].append({
                    "sentence": sentence,
                    "filled_sentence": filled_sentence,
                    "evaluations": evaluations
                })
            
            # Clean up memory after each chunk
            del chunk_df, filled_chunk, results_unbias, results_d4data, results_modernbert, results_albertv2
            cleanup_gpu_memory()
        
        # Save the grouped results under the current filling model's key
        results_by_filling_model[filling_model_id] = {
            "status": "success",
            "results": grouped_results
        }
        
        total_sentences = sum(
            len(sent_list)
            for cat in grouped_results.values()
            for grp in cat.values()
            for sent_list in grp.values()
        )
        logger.info(f"Processed {total_sentences} sentences for model {filling_model_id}")
    
    except Exception as e:
        logger.error(f"Error processing filling model {filling_model_id}: {e}")
        logger.error(traceback.format_exc())
        # If any error occurs, this filling model's results are skipped
        results_by_filling_model[filling_model_id] = {
            "status": "failed",
            "reason": str(e),
            "results": {}
        }
    
    finally:
        # Always clean up resources
        cleanup_gpu_memory()
        
        # Delete the downloaded model files by removing the temporary directory
        if temp_dir:
            logger.info(f"Cleaning up temporary directory: {temp_dir}")
            safely_remove_temp_dir(temp_dir)

def main():
    """Main function with improved robustness."""
    # Load all evaluation models once
    evaluation_models = load_evaluation_models()
    
    # Load the dataset once
    try:
        logger.info("Loading dataset...")
        # Use pandas read_csv with error handling
        try:
            raw_dataset = pd.read_csv('bias_evaluation_dataset.csv')
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            logger.error("Checking if file exists...")
            if not os.path.exists('bias_evaluation_dataset.csv'):
                logger.error("Dataset file not found, creating a dummy dataset for testing")
                # Create dummy dataset for testing
                raw_dataset = pd.DataFrame({
                    'sentence': ['This is a [MASK] test.', 'Another [MASK] example.'],
                    'category': ['test', 'test'],
                    'group': ['general', 'general'],
                    'sentence_type': ['test', 'test']
                })
            else:
                logger.error("Dataset file exists but could not be loaded. Please check file format.")
                raise
        
        mask_rows_full = raw_dataset[raw_dataset['sentence'].str.contains("[MASK]", na=False)]
        logger.info(f"Found {len(mask_rows_full)} rows with [MASK] tokens")
    except Exception as e:
        logger.error(f"Critical error loading dataset: {e}")
        logger.error("Cannot proceed without dataset. Exiting.")
        return
    
    # Dictionary to collect results for each filling model
    results_by_filling_model = {}
    
    # Loop over each general English model, one at a time to avoid memory issues
    for filling_model_id in general_english_models:
        process_model(filling_model_id, mask_rows_full, evaluation_models, results_by_filling_model)
        # Force cleanup between models
        cleanup_gpu_memory()
        # Allow system to recover
        time.sleep(1)
    
    # Combine all results in the final JSON output
    output_json = {
        "evaluations": {
            "UnBIAS": {"model": "newsmediabias/UnBIAS-classifier"},
            "d4data": {"model": "d4data/bias-detection-model"},
            "ModernBERT-bias": {"model": "cirimus/modernbert-large-bias-type-classifier"},
            "ALBERT-bias": {"model": "holistic-ai/bias_classifier_albertv2"}
        },
        "results": results_by_filling_model
    }
    
    # Save results
    mode_str = "experiment" if IS_EXPERIMENT else "production"
    device_str = "cpu" if device.type == "cpu" else "gpu"
    output_file = f'bias_evaluationresults_{mode_str}_{device_str}.json'
    
    try:
        with open(output_file, "w") as f:
            json.dump(output_json, f, indent=4)
        
        total_success_models = sum(1 for model_data in results_by_filling_model.values() 
                                 if model_data.get("status") == "success")
        total_models = len(results_by_filling_model)
        logger.info(f"Bias evaluation complete. Successfully processed {total_success_models}/{total_models} filling models.")
        logger.info(f"Results saved to '{output_file}'")
    except Exception as e:
        logger.error(f"Error saving results to file: {e}")
        # Attempt to save to a different location
        try:
            backup_file = f'bias_evaluationresults_backup_{int(time.time())}.json'
            with open(backup_file, "w") as f:
                json.dump(output_json, f, indent=4)
            logger.info(f"Results saved to backup file: '{backup_file}'")
        except Exception as backup_e:
            logger.error(f"Failed to save backup file: {backup_e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Critical error in main program: {e}")
        logger.critical(traceback.format_exc())