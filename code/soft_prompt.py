import wandb
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PrefixTuningConfig, TaskType
import pandas as pd
import os
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm import tqdm
import evaluate

# Disable parallelism warnings from the tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize wandb
wandb.init(project="title-generation", name="peft-tuning", config={
    "epochs": 30,
    "batch_size": 8,
    "learning_rate": 1e-4,
    "optimizer": "AdamW",
    "model": "T5",
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_and_tokenizer_name_or_path = "t5-base"

peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, num_virtual_tokens=200)

model = AutoModelForSeq2SeqLM.from_pretrained(model_and_tokenizer_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
tokenizer = AutoTokenizer.from_pretrained("t5-base")
epochs = 30

### Load the dataset
all_data = []

for domain in os.listdir('/workspace/aiclub02/huytq/TitleGenerator/data/scrawled'):
    domain_name = domain.split('.')[0]
    data = pd.read_csv(f'/workspace/aiclub02/huytq/TitleGenerator/data/scrawled/{domain}')
    data['Abstract'] = data['Abstract'].apply(lambda x: x.replace('\n', ' '))
    data['Abstract'] = data['Abstract'].apply(lambda x: f"{x} @{domain_name}")
    data = data[['Title', 'Abstract']]
    data.columns = ['target_text', 'input_text']
    data['prefix'] = 'Generate title:'
    all_data.append(data)

# Concatenate all data
all_data = pd.concat(all_data, ignore_index=True)

# Split data into train, validation, and test sets
train_df, temp_df = train_test_split(all_data, test_size=0.25, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.75, random_state=42)

# Convert DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Preprocess the dataset
def preprocess_function(examples):
    inputs = [f"Generate a concise, descriptive title for an academic paper based on the abstract below (limit: 10-15 words): {ex}" for ex in examples['input_text']]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(examples['target_text'], max_length=512, truncation=True, padding="max_length")
    model_inputs['labels'] = labels['input_ids']
    return model_inputs
# uncomment for soft prompt with tfidf
# # Define the TF-IDF common keyword function
# def find_common_keywords_tfidf(text1, text2):
#     # Ensure the inputs are strings (text1 and text2 should not be lists)
#     if isinstance(text1, list):
#         text1 = ' '.join(text1)
#     if isinstance(text2, list):
#         text2 = ' '.join(text2)
    
#     # Combine the texts into a corpus
#     corpus = [text1, text2]

#     # Compute TF-IDF scores
#     vectorizer = TfidfVectorizer(stop_words="english")
#     tfidf_matrix = vectorizer.fit_transform(corpus)

#     # Find words with non-zero values in both texts
#     feature_names = vectorizer.get_feature_names_out()
#     text1_words = set(tfidf_matrix[0].nonzero()[1])
#     text2_words = set(tfidf_matrix[1].nonzero()[1])
#     common_indices = text1_words & text2_words

#     # Map indices back to words
#     common_keywords = [feature_names[index] for index in common_indices]
#     return common_keywords


# # Example preprocessing function for dataset (including common keywords in prompt)
# def preprocess_function(examples):
#     inputs = []
#     for ex in examples['input_text']:
#         # Ensure input_text is a single string
#         ex = ' '.join(ex) if isinstance(ex, list) else ex
        
#         # Extract common keywords between the abstract and the current title (if available)
#         common_keywords = find_common_keywords_tfidf(ex, examples['target_text'])  # Modify as needed
#         input_text = f"Generate a concise, descriptive title for an academic paper based on the abstract below. Keywords to include: {', '.join(common_keywords)} (limit: 10-15 words): {ex}"
#         inputs.append(input_text)

#     model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
#     labels = tokenizer(examples['target_text'], max_length=512, truncation=True, padding="max_length")
#     model_inputs['labels'] = labels['input_ids']
#     return model_inputs
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)

train_dataloader = DataLoader(
    tokenized_train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=8, pin_memory=True
)
eval_dataloader = DataLoader(tokenized_val_dataset, collate_fn=default_data_collator, batch_size=8, pin_memory=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * epochs),
)

model = model.to(device)
rouge = evaluate.load("rouge")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    eval_loss = 0
    eval_preds = []
    eval_labels = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        eval_preds.extend(
            tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
        )
        eval_labels.extend(
            tokenizer.batch_decode(batch["labels"].detach().cpu().numpy(), skip_special_tokens=True)
        )

    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    
    # Log training metrics to wandb
    wandb.log({
        "train_ppl": train_ppl.item(),
        "train_epoch_loss": train_epoch_loss.item(),
        "eval_ppl": eval_ppl.item(),
        "eval_epoch_loss": eval_epoch_loss.item(),
    })
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

    # Compute ROUGE-L
    rouge_result = rouge.compute(predictions=eval_preds, references=eval_labels, rouge_types=["rougeL"])
    if isinstance(rouge_result, dict) and "rougeL" in rouge_result:
        result = {"rougeL": rouge_result["rougeL"] * 100}  # Directly multiply fmeasure by 100
    print(f"ROUGE-L Score: {rouge_result}")

    # Log ROUGE-L score to wandb
    wandb.log({
        "rougeL": rouge_result["rougeL"] * 100
    })

    # Save the model after each epoch
    model.save_pretrained(f"/workspace/aiclub02/huytq/SOTitle/models/epoch_{epoch}")
    tokenizer.save_pretrained(f"/workspace/aiclub02/huytq/SOTitle/models/epoch_{epoch}")
