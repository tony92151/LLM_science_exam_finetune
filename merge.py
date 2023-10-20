from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import StoppingCriteria, StoppingCriteriaList
from random import sample 
import argparse
import pandas as pd
import torch
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('-m','--lmodel', type = str, required = True)
parser.add_argument('-m','--output', type = str, required = True)
args = parser.parse_args()

print(f"Model path: {args.lmodel}")

print("Merge and save...")
time.sleep(1)


config = PeftConfig.from_pretrained(peft_model_id)

model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, args.lmodel, torch_dtype=torch.bfloat16)


tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

model = model.merge_and_unload()

merged_model_path = args.output
model.save_pretrained(merged_model_path, max_shard_size="2GB")

if True:
    tokenizer.save_pretrained(merged_model_path)
