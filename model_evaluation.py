import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset
import torch
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import matplotlib.pyplot as plt

# Download NLTK data
nltk.download('punkt')

# Define paths
processed_data_dir = "/Users/vishnups/Documents/legal/processed_data/"
model_dir = "/Users/vishnups/Documents/legal/fine_tuned_model/final_model"
results_dir = "/Users/vishnups/Documents/legal/evaluation_results/"

# Create results directory if it doesn't exist
os.makedirs(results_dir, exist_ok=True)

# Load test data
test_df = pd.read_csv(os.path.join(processed_data_dir, 'test.csv'))
test_dataset = Dataset.from_pandas(test_df)

# Load model and tokenizer
try:
    # First try loading from local path with trust_remote_code=True
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
except Exception as e:
    print(f"Could not load model from local path: {e}")
    print("Using pre-trained model facebook/bart-large-cnn")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize ROUGE scorer with additional metrics
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)

# Function to generate summaries
def generate_summary(text):
    inputs = tokenizer(text, max_length=1024, truncation=True, padding="max_length", return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate summary
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=512,
        num_beams=4,
        early_stopping=True
    )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Evaluation metrics
def calculate_metrics(reference, prediction):
    # ROUGE scores
    rouge_scores = scorer.score(reference, prediction)
    
    # BLEU score
    smoothie = SmoothingFunction().method1
    
    # Use simple split instead of nltk.word_tokenize to avoid punkt_tab dependency
    reference_tokens = reference.lower().split()
    prediction_tokens = prediction.lower().split()
    
    bleu_score = sentence_bleu([reference_tokens], prediction_tokens, smoothing_function=smoothie)
    
    return {
        'rouge1': rouge_scores['rouge1'].fmeasure,
        'rouge2': rouge_scores['rouge2'].fmeasure,
        'rougeL': rouge_scores['rougeL'].fmeasure,
        'rougeLsum': rouge_scores['rougeLsum'].fmeasure if 'rougeLsum' in rouge_scores else 0,
        'bleu': bleu_score
    }

# Evaluate on test set
results = []
for i, example in enumerate(test_dataset):
    original_text = example['summary']
    
    # Generate summary
    generated_summary = generate_summary(original_text)
    
    # Calculate metrics
    metrics = calculate_metrics(original_text, generated_summary)
    
    results.append({
        'case_id': example['case_id'],
        'original_text': original_text,
        'generated_summary': generated_summary,
        'rouge1': metrics['rouge1'],
        'rouge2': metrics['rouge2'],
        'rougeL': metrics['rougeL'],
        'rougeLsum': metrics['rougeLsum'],
        'bleu': metrics['bleu']
    })
    
    # Print progress
    if (i + 1) % 5 == 0:
        print(f"Processed {i + 1}/{len(test_dataset)} examples")

# Convert to DataFrame and save results
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(results_dir, 'evaluation_results.csv'), index=False)

# Calculate and print average metrics
avg_metrics = {
    'rouge1': np.mean(results_df['rouge1']),
    'rouge2': np.mean(results_df['rouge2']),
    'rougeL': np.mean(results_df['rougeL']),
    'rougeLsum': np.mean(results_df['rougeLsum']),
    'bleu': np.mean(results_df['bleu'])
}

print("\nAverage Metrics:")
for metric, value in avg_metrics.items():
    print(f"{metric}: {value:.4f}")

# Save average metrics
with open(os.path.join(results_dir, 'average_metrics.txt'), 'w') as f:
    for metric, value in avg_metrics.items():
        f.write(f"{metric}: {value:.4f}\n")

# Add visualization of metrics
plt.figure(figsize=(10, 6))
plt.bar(avg_metrics.keys(), avg_metrics.values())
plt.title('Average Evaluation Metrics')
plt.ylabel('Score')
plt.ylim(0, 1.0)  # ROUGE scores are between 0 and 1
plt.savefig(os.path.join(results_dir, 'metrics_chart.png'))
plt.close()

print(f"Metrics visualization saved to {os.path.join(results_dir, 'metrics_chart.png')}")
print(f"\nEvaluation completed. Results saved to {results_dir}")