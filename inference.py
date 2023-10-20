
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import StoppingCriteria, StoppingCriteriaList
from random import sample 
import argparse
import pandas as pd
import torch
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('-m','--model', type = str, required = True)
parser.add_argument('-d','--data', type = str, required = True)
parser.add_argument('-s','--sample', type = int, default = 3)
args = parser.parse_args()

print(f"Model path: {args.model}")


def generate_prompt(data_point):
    return f"""### Comment: You are a helpful bot that answer a multiple choice question with options [A, B, C, D, E] if background provided. 
Please determine whether the background knowledge is useful first, if it is useful, please answer the question based on the background knowledge, otherwise, please answer the question and ignore the background knowledge. **You only allowed reply in following format, for example "useful:True||thought:Because...||answer:A" . The "useful" is a boolean value, "thought" is your reasoning thought about the question and the answer which should be one of [A, B, C, D, E].**
Background knowledge: {data_point["context"]}
### Human:
Question: {data_point["prompt"]}
A {data_point["A"]}
B {data_point["B"]}
C {data_point["C"]}
D {data_point["D"]}
E {data_point["E"]}
### REPLY:
"""



df = pd.read_csv("./stem_1k_v1.csv")
sample_ = sample(df.to_dict(orient='records'), args.sample)


model_name = args.model

tokenizer = AutoTokenizer.from_pretrained(model_name)
    
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
    
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    quantization_config=bnb_config
)

model.config.use_cache = True

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids:list):
        self.keywords = keywords_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for k in self.keywords:
            if np.array_equal(input_ids[0].clone().detach().cpu().numpy()[-len(k):], np.array(k)):
                return True
#             print(input_ids[0][-len(k):], k)
#             exit()
#         if input_ids[0][-1] in self.keywords:
#             return True
        return False

stop_words = ['[</END>]']
stop_criteria = KeywordsStoppingCriteria([tokenizer.encode(w)[1:] for w in stop_words])

# print(tokenizer.encode('[</END>]')[1:])

print(f"{model.config.use_cache=}")

for i,s in enumerate(sample_):
    print("="*20, f" sample {i+1}/{args.sample} ", "="*20)
    text = generate_prompt(s)

    example_inputs = tokenizer(text , return_tensors="pt").to(f"cuda:{model.device.index}")
    
    
    with torch.no_grad():
        output = model.generate(input_ids=example_inputs["input_ids"], attention_mask=example_inputs["attention_mask"], max_new_tokens=400, 
                                return_dict_in_generate=True, output_scores=True,pad_token_id=tokenizer.eos_token_id,
                                stopping_criteria=StoppingCriteriaList([stop_criteria]))

    print(tokenizer.decode(output.sequences[0]))
    print("\n\n")