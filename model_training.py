import os
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import torch
import nltk
from datasets import Dataset
from nltk.tokenize import sent_tokenize

# Download NLTK data
nltk.download('punkt')

# Define paths
processed_data_dir = "/Users/vishnups/Documents/legal/processed_data/"
model_output_dir = "/Users/vishnups/Documents/legal/fine_tuned_model/"

# Create output directory if it doesn't exist
os.makedirs(model_output_dir, exist_ok=True)

# Load data
train_df = pd.read_csv(os.path.join(processed_data_dir, 'train.csv'))
val_df = pd.read_csv(os.path.join(processed_data_dir, 'validation.csv'))

# Convert to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Initialize tokenizer and model
model_name = "facebook/bart-large-cnn"  # A good base model for summarization
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Tokenization function
def preprocess_function(examples):
    # Tokenize inputs
    inputs = examples["summary"]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length")
    
    # For training a summarization model, we can use the same text as labels
    # In a real scenario, you might have separate full text and summary fields
    labels = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing
tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_val = val_dataset.map(preprocess_function, batched=True)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=model_output_dir,
    eval_strategy="epoch",  # Changed from evaluation_strategy to eval_strategy
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    # fp16=True is already commented out in your current code
    push_to_hub=False,
)

# Initialize trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained(os.path.join(model_output_dir, "final_model"))
tokenizer.save_pretrained(os.path.join(model_output_dir, "final_model"))

print("Model training completed and saved!")