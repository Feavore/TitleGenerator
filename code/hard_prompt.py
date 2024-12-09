import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, AdamW, get_linear_schedule_with_warmup, TrainerCallback
import evaluate
from datasets import Dataset
import wandb
import torch
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
torch.cuda.empty_cache()

# Set the WANDB_WATCH environment variable to "false" to disable system monitoring
os.environ["WANDB_WATCH"] = "false"

# Initialize wandb
wandb.init(project="title-generation", name="hard-tuning-tfidf", config={"learning_rate": 2e-5, "batch_size": 4, "epochs": 15})



# Load the dataset
### Load the dataset
all_data = []
for field in os.listdir('/workspace/aiclub02/huytq/TitleGenerator/data'):
    print(field)
    for domain in os.listdir(os.path.join('/workspace/aiclub02/huytq/TitleGenerator/data', field)):
      if ".xlsx" in domain or "popular_physics" in domain :
        continue
      domain_name = domain.split('.')[0]
      data = pd.read_csv(f'/workspace/aiclub02/huytq/TitleGenerator/data/{field}/{domain}')
      data['Abstract'] = data['Abstract'].apply(lambda x: x.replace('\n', ' '))
      data['Abstract'] = data['Abstract'].apply(lambda x: f"{x} @{domain_name}")
      data = data[['Title', 'Abstract']]
      data.columns = ['target_text', 'input_text']
      all_data.append(data)
# Concatenate all data
all_data = pd.concat(all_data, ignore_index=True)

# Split data into train, validation, and test sets
train_df, temp_df = train_test_split(all_data, test_size=0.25, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.75, random_state=42)

# Convert DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Initialize the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')
model.gradient_checkpointing_enable()
# Explicitly set use_cache to False
model.config.use_cache = False

# Uncomment for the hard prompt without tfidf
# def preprocess_function(examples):
#     inputs = [f"Generate a concise, descriptive title for an academic paper based on the abstract below (limit: 10-15 words): {ex}" for ex in examples['input_text']]
#     model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
#     labels = tokenizer(examples['target_text'], max_length=512, truncation=True, padding="max_length")
#     model_inputs['labels'] = labels['input_ids']
#     return model_inputs
# Define the TF-IDF common keyword function
def find_common_keywords_tfidf(text1, text2):
    # Ensure the inputs are strings (text1 and text2 should not be lists)
    if isinstance(text1, list):
        text1 = ' '.join(text1)
    if isinstance(text2, list):
        text2 = ' '.join(text2)
    
    # Combine the texts into a corpus
    corpus = [text1, text2]

    # Compute TF-IDF scores
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Find words with non-zero values in both texts
    feature_names = vectorizer.get_feature_names_out()
    text1_words = set(tfidf_matrix[0].nonzero()[1])
    text2_words = set(tfidf_matrix[1].nonzero()[1])
    common_indices = text1_words & text2_words

    # Map indices back to words
    common_keywords = [feature_names[index] for index in common_indices]
    return common_keywords


# Example preprocessing function for dataset (including common keywords in prompt)
def preprocess_function(examples):
    inputs = []
    for ex in examples['input_text']:
        # Ensure input_text is a single string
        ex = ' '.join(ex) if isinstance(ex, list) else ex
        
        # Extract common keywords between the abstract and the current title (if available)
        common_keywords = find_common_keywords_tfidf(ex, examples['target_text'])  # Modify as needed
        input_text = f"Generate a concise, descriptive title for an academic paper based on the abstract below. Keywords to include: {', '.join(common_keywords)} (limit: 10-15 words): {ex}"
        inputs.append(input_text)

    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(examples['target_text'], max_length=512, truncation=True, padding="max_length")
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True)

# Load the ROUGE metric using the evaluate library
rouge_metric = evaluate.load("rouge")

def compute_metrics(eval_pred):
    with torch.no_grad():
        # Extract predictions and labels from the EvalPrediction object
        predictions, labels = eval_pred.predictions, eval_pred.label_ids

        # Check if predictions is a tuple, and extract the logits (first element)
        if isinstance(predictions, tuple):
            predictions = predictions[0]  # Logits are in the first element of the tuple

        # Check shapes (for debugging purposes)
        print(f"Predictions shape: {predictions.shape}")
        print(f"Labels shape: {labels.shape}")

        # If predictions are logits (tensor), apply argmax to get token IDs
        # Convert predictions to PyTorch tensor if it is a numpy array
        if isinstance(predictions, np.ndarray):
            predictions = torch.tensor(predictions)

        # Now apply argmax to get the predicted token IDs
        predicted_ids = torch.argmax(predictions, dim=-1)

        # Convert predicted_ids to numpy (no need for cpu() as labels are numpy)
        predicted_ids = predicted_ids.numpy()

        # Decode the predictions and labels
        decoded_preds = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Compute ROUGE scores
        result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, rouge_types=["rougeL"])

        # Check if the result contains 'rougeL' scores directly as a float
        if isinstance(result, dict) and "rougeL" in result:
            result = {"rougeL": result["rougeL"] * 100}  # Directly multiply fmeasure by 100

        # Log ROUGE scores to wandb
        wandb.log({"eval_rougeL": result.get("rougeL", 0)})

    return result

# Define training arguments
training_args = TrainingArguments(
    output_dir='/workspace/aiclub02/huytq/SOTitle/result/Base_t5',
    per_device_train_batch_size=10,
    per_device_eval_batch_size=1,
    eval_strategy='epoch',  # Updated to 'eval_strategy'
    eval_accumulation_steps=100, 
    learning_rate=2e-5,
    num_train_epochs=15,
    weight_decay=0.01,
    report_to="wandb",  # Add this line to report to wandb
    logging_dir="/workspace/aiclub02/huytq/SOTitle/result/Base_t5/logs",  # Specify log directory
    fp16=True,
)

# Initialize the optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)  # Use PyTorch's AdamW
num_train_steps = len(train_dataset) * training_args.num_train_epochs // training_args.per_device_train_batch_size
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

# Define a custom callback to log training and eval loss to wandb
class LogLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Log the training and evaluation loss
        if "loss" in logs:
            wandb.log({"train_loss": logs["loss"]})
        if "eval_loss" in logs:
            wandb.log({"eval_loss": logs["eval_loss"]})

# Initialize the Trainer with the custom callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, lr_scheduler),  # Pass the optimizer and scheduler here
    callbacks=[LogLossCallback()]  # Correctly instantiate the callback
)

# Train the model
trainer.train()
