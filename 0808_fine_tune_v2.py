from glob import glob
import os
import pandas as pd
from tqdm import tqdm
import json
import copy
import re
import datasets
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2TokenizerFast
from transformers import GPT2Tokenizer
from datasets import load_dataset
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import  DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch
from datasets import load_from_disk
from transformers import PreTrainedTokenizerFast
from transformers import DataCollatorForLanguageModeling
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn as nn
#from sklearn.model_selection import train_test_split

class GPT2ForRegression(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.regression_head = nn.Linear(config.n_embd, 1)

    def forward(self, input_ids, attention_mask):
        outputs = super().forward(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_layer_hidden_states = outputs.hidden_states[-1]
        
        # Max pooling
        pooled_output, _ = torch.max(last_layer_hidden_states, dim=1)
        
        # Regression head
        regression_output = self.regression_head(pooled_output)
        
        return regression_output.squeeze()


config = GPT2Config.from_pretrained('/mnt/prj/AJ/dock_bert/0521_GPT2/cerebras/final_ckpt/runs_BPE_nospace/gpt2medium/params_train_to_hf.json')
model_state_dict = torch.load('/mnt/prj/AJ/dock_bert/0521_GPT2/cerebras/final_ckpt/runs_BPE_nospace/gpt2medium/checkpoint_29964_to_hf.bin')

# Load the base GPT-2 model
base_model = GPT2LMHeadModel(config)
base_model.load_state_dict(model_state_dict)

# Create the custom GPT2ForRegression model
model = GPT2ForRegression(config)

# Copy the weights from the base model to the custom model
model.transformer.load_state_dict(base_model.transformer.state_dict())

# Initialize the regression head (optional, as PyTorch initializes layers by default)
model.regression_head.apply(model._init_weights)

tokenizer = PreTrainedTokenizerFast(tokenizer_file="/mnt/prj/AJ/dock_bert/0521_GPT2/cerebras/final_ckpt/SB_BPE_tokenizer_char.json")
tokenizer._tokenizer.enable_padding(length=1024)  # set max length to 1024 if desired

if not tokenizer.pad_token:
    tokenizer.add_tokens(["[PAD]"])
    tokenizer.pad_token = "[PAD]"
    
df = pd.read_csv('/mnt/prj/AJ/dock_bert/pdbbind_finetune/fine_tune_dataset/refined-set_word_p_affinity_(1).tsv', sep='\t')

from torch.utils.data import Dataset

# Tokenizing the 'wordset' column with padding to a specific max length
max_length = 1024  # You can set this to an appropriate value for your data
tokenized_data = tokenizer(df['wordset'].tolist(), padding='max_length', truncation=True, max_length=max_length)


# Creating a PyTorch Dataset
class RegressionDataset(Dataset):
    def __init__(self, tokenized_data, targets):
        self.input_ids = tokenized_data['input_ids']
        self.attention_mask = tokenized_data['attention_mask']
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_mask[idx], dtype=torch.long),
            'target': torch.tensor(self.targets[idx], dtype=torch.float)
        }

# Creating an instance of the dataset
targets = df['p_affinity'].tolist()
regression_dataset = RegressionDataset(tokenized_data, targets)


import torch

# Check if a GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the device (GPU if available)
model = model.to(device)

from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
import torch.nn.functional as F

# Splitting dataset into training and validation sets (e.g., 90% training, 10% validation)
train_size = int(0.9 * len(regression_dataset))
val_size = len(regression_dataset) - train_size
train_dataset, val_dataset = random_split(regression_dataset, [train_size, val_size])

# Creating DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# Loss function and optimizer
loss_function = torch.nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)


# Training loop
epochs = 10
train_loss_history = [] # To store training loss
val_loss_history = [] # To store validation loss
best_val_loss = float('inf') # Initialize with a high value

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch in train_loader:
        # Move the batch data to the device (GPU if available)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        target = batch['target'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_function(outputs, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Average training loss
    train_loss /= len(train_loader)
    train_loss_history.append(train_loss)
    print(f"Training Loss: {train_loss}")

    # Optional: Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            # Move the batch data to the device (GPU if available)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target = batch['target'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            val_loss += loss_function(outputs, target).item()

    val_loss /= len(val_loader)
    val_loss_history.append(val_loss)
    print(f"Validation Loss: {val_loss}")

    # Save the model if validation loss decreases
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), '/mnt/prj/AJ/dock_bert/pdbbind_finetune/fine_tuned_model_naver_v2.bin')
        print("Model saved with improved validation loss.")

# Optionally save training and validation history
import json
with open('/mnt/prj/AJ/dock_bert/pdbbind_finetune/training_history_v2.json', 'w') as f:
    json.dump({'train_loss': train_loss_history, 'val_loss': val_loss_history}, f)  