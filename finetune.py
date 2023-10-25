
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# df1 = pd.read_csv("./llm-science-exam-data-w-thought/train.csv")
# df2 = pd.read_csv("./llm-science-exam-data-w-thought/extra_train_set.csv")

# df = pd.concat([df1, df2], join="inner")

# def make_instruction(data):
#     return f"""You are a genius bot that always correctly answer a multiple choice question with options [A, B, C, D, E] base on
#  knowledge if provided. Please determine whether the background knowledge is useful first, if it is useful, pleas
# e answer the question based on the knowledge, otherwise, please answer the question and ignore the knowledge. **Y
# ou are only allowed to reply in the following format, for example, "useful:True||though:Because...||answer:A" . T
# he "useful" is a boolean value, and "thought" is your reasoning thought about the question and the answer which s
# hould be one of [A, B, C, D, E].**\nKnowledge: {data['context']}"""

# def make_input(data):
#     return f"{data['prompt']}\nA {data['A']}\nB {data['B']}\nC {data['C']}\nD {data['D']}\nE {data['E']}"

# def make_output(data):
#     if data['thought'] == "fail":
#         return f"useful:{data['useful']}||though:<empty>||answer:{data['answer']}### END."
#     else:
#         return f"useful:{data['useful']}||though:{data['thought']}||answer:{data['answer']}### END."


# df["instruction"] = df.apply(make_instruction, axis=1)
# df["input"] = df.apply(make_input, axis=1)
# df["output"] = df.apply(make_output, axis=1)


# df.to_csv("./llm-science-exam-data-w-thought/train_.csv",  index=False)



# limitations under the License.
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from datasets import Dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments

from trl import SFTTrainer


tqdm.pandas()


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    model_name: Optional[str] = field(default="facebook/opt-350m", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco", metadata={"help": "the dataset name"}
    )
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    log_with: Optional[str] = field(default="none", metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=64, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the number of logging steps"})
    use_auth_token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(
        default=100, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    save_total_limit: Optional[int] = field(default=10, metadata={"help": "Limits total number of checkpoints."})
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    hub_model_id: Optional[str] = field(default=None, metadata={"help": "The name of the model on HF Hub"})
    use_flash_attention_2: Optional[bool] = field(default=False, metadata={"help": "Wether to use flash attention 2 (flash-attn==2.2.4)"})

# trainer = transformers.Trainer(
#     model=model,
#     train_dataset=data,
#     args=transformers.TrainingArguments(
#         per_device_train_batch_size=1,
#         gradient_accumulation_steps=8,
#         # warmup_steps=2,
#         warmup_ratio=0.03,
#         # max_steps=20,
#         num_train_epochs=2,
#         learning_rate=2e-4,
#         fp16=True,
#         logging_steps=10,
#         save_strategy="steps",
#         output_dir=lora_output_dir,
#         optim=

# args = {
#     'model_name': '/kaggle/input/mistral-7b-instruct-v0-1/Mistral-7B-Instruct-v0.1',
#     'dataset_name': './llm-science-exam-data-w-thought/train_.csv',
#     'dataset_text_field': 'text',
#     'log_with': 'none',
#     'learning_rate': 2e-4,
#     'batch_size': 64,
#     'seq_length': 2048,
#     'gradient_accumulation_steps': 8,
#     'load_in_8bit': False,
#     'load_in_4bit': True,
#     'use_peft': True,
#     'trust_remote_code': False,
#     'output_dir': 'output',
#     'peft_lora_r': 8,
#     'peft_lora_alpha': 16,
#     'logging_steps': 1,
#     'use_auth_token': False,
#     'num_train_epochs': 1,
#     'max_steps': -1,
#     'save_steps': 200,
#     'save_total_limit': 2,
#     'push_to_hub': False,
#     'hub_model_id': None
# }


# parser = HfArgumentParser(ScriptArguments)
# script_args = parser.parse_dict(args)[0]
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


Accelerator().local_process_index

# %% [code] {"execution":{"iopub.status.busy":"2023-10-16T13:11:41.208797Z","iopub.execute_input":"2023-10-16T13:11:41.209856Z"}}
# Step 1: Load the model
if script_args.load_in_8bit and script_args.load_in_4bit:
    raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
elif script_args.load_in_8bit or script_args.load_in_4bit:
    print("Use quantization.")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=script_args.load_in_4bit,
        load_in_8bit=script_args.load_in_8bit,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    

    # Copy the model to each device
    device_map = {"": Accelerator().local_process_index}
    # device_map="auto"
#     torch_dtype = torch.bfloat16
    torch_dtype = torch.float16
else:
    device_map = None
    quantization_config = None
    torch_dtype = None

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
    use_auth_token=script_args.use_auth_token,
    use_flash_attention_2=script_args.use_flash_attention_2
)

# %% [code]
# Step 2: Load the dataset
def get_training_prompt(data_point):
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
useful:{data_point['useful']}||thought:{data_point['thought']}||answer:{data_point['answer']}[</END>] """


data_list = []

df = pd.read_csv(script_args.dataset_name)

for row in df.to_dict(orient='records'):
    data_list.append({"text": get_training_prompt(row)})

    
dataset = Dataset.from_pandas(pd.DataFrame.from_dict(data_list))
# dataset = Dataset.from_list(data_list)
# dataset = load_dataset(script_args.dataset_name, split="train")



training_args = TrainingArguments(
    bf16=False,
    bf16_full_eval=False,
    fp16=True,
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    num_train_epochs=script_args.num_train_epochs,
    max_steps=script_args.max_steps,
    report_to=script_args.log_with,
    save_steps=script_args.save_steps,
    save_total_limit=script_args.save_total_limit,
    push_to_hub=script_args.push_to_hub,
    hub_model_id=script_args.hub_model_id,
    optim="paged_adamw_8bit",
)


# Step 4: Define the LoraConfig
import bitsandbytes as bnb

def find_linear_layers(model):
    """find linear layers in given transformer model"""
    lora_module_names = set()
    for name, module in model.named_modules():
        # 4 bits for qlora
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    print(f"LoRA module names: {list(lora_module_names)}")
    return list(lora_module_names)

target_modules = find_linear_layers(model)

if script_args.use_peft:
    print("Use peft")
    peft_config = LoraConfig(
        r=script_args.peft_lora_r,
        lora_alpha=script_args.peft_lora_alpha,
        target_modules=target_modules,
        modules_to_save=["embed_tokens", "lm_head"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
else:
    peft_config = None


# Step 5: Define the Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    max_seq_length=script_args.seq_length,
    train_dataset=dataset,
    dataset_text_field=script_args.dataset_text_field,
    peft_config=peft_config,
)


trainer.train()


# Step 6: Save the model
trainer.save_model(script_args.output_dir)


