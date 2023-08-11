## import libraries
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2TokenizerFast
from transformers import GPT2Tokenizer
from datasets import load_dataset
import json
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import  DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch
from datasets import load_from_disk
import pandas as pd
from glob import glob
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
import os
import sys

## Load model
# load GPT-2 small configuration
config = GPT2Config.from_pretrained('/mnt/prj/AJ/dock_bert/0521_GPT2/gpt2_small/checkpoint-700000/config.json')
# create GPT-2 small model
model = GPT2LMHeadModel(config)

model_state_dict = torch.load('/mnt/prj/AJ/dock_bert/0521_GPT2/gpt2_small/checkpoint-700000/pytorch_model.bin')
model.load_state_dict(model_state_dict)

# Check if a GPU is available and if not, use a CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Move the model to the GPU
model = model.to(device)

## load tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file="/mnt/prj/AJ/dock_bert/0521_GPT2/BPE_data_all_tokenizer/save_bpe.json")
tokenizer._tokenizer.enable_padding(length=None)  # set max length to 1024 if desired


data_path = '/mnt/prj/AJ/dock_bert/0521_GPT2/scripts/eval_target/*'
pc1_data_path = glob(data_path)

pc1_data_all=[]

for path in pc1_data_path:

    single_sen = pd.read_csv(path, sep='\t')

    one_sen = ''
    for word in single_sen['word']:
        one_sen += word + ' '

    one_sen = one_sen[:-1]

    pc1_data_all.append(one_sen)
    
    
pc1_ind_total=[]
pc1_prompt_total = []
pc1_generated_str_lst =[]
pc1_original_str_lst = []

for data_ind in tqdm(range(len(pc1_data_all))):

    the_data = pc1_data_all[data_ind]
    model.eval()
    prompt = ''
    for i in range(int(len(the_data.split()) * 0.5)):
        prompt += the_data.split()[i] + ' '
    prompt = prompt[:-1]
    pc1_prompt_total.append(prompt)

    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    input_ids = input_ids.to(device)

    output = model.generate(input_ids, max_length=int(len(original_str.split())*2), temperature=0.7, do_sample=True, num_return_sequences=100)

    generated_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output]
    
    for gen_txt in generated_texts:
        pc1_generated_str_lst.append(gen_txt[len(prompt):])
        
    original_str = the_data[len(prompt):].strip()

    pc1_original_str_lst.append(original_str)
    pc1_ind_total.append(data_ind)
    
    
txt_name_num = 0
num = 0
save_txt = []

for sentence in pc1_generated_str_lst:
    one_sen = ''
    for word in sentence.split()[:int(len(pc1_original_str_lst[txt_name_num].split()) * 2)]:
        one_sen += word + ' '
    one_sen = one_sen[:-1]

    save_txt.append(one_sen)
    num += 1

    if num == 100:  
        with open('model2_0602_pc1_gen/gen_'+pc1_test_only_path[txt_name_num].split('/')[-1].split('.')[0]+'.txt', 'w') as f:
            for item in save_txt:
                f.write("%s\n" % item)
        save_txt = []  # Reset save_txt for the next 100 sentences
        num = 0
        txt_name_num += 1