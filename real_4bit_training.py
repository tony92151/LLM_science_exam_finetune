import os
import time
import bitsandbytes as bnb
import pandas as pd
import torch
import transformers
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM, PeftConfig, PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

# model_id = "EleutherAI/gpt-neox-20b"
# model_id = "elinas/llama-7b-hf-transformers-4.29"
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
# model_id = "Open-Orca/Mistral-7B-OpenOrca"
# model_id = "Open-Orca/OpenOrca-Platypus2-13B"


lora_output_dir = "./output20"
PREFIX_CHECKPOINT_DIR = "checkpoint"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

# Load model directly
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, device_map="auto"
)


model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


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
# target_modules=["q_proj","v_proj","k_proj","o_proj","gate_proj","down_proj","up_proj"]

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=target_modules,
    modules_to_save=["embed_tokens", "lm_head"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)


train_df = pd.read_csv(
    "./llm-science-exam-data-w-thought/train_.csv", dtype={"useful": bool}
)
# extra_train_df = pd.read_csv(
#     "./llm-science-exam-data-w-thought/extra_train_set.csv", dtype={"useful": bool}
# )


# concat_df = pd.concat([train_df, extra_train_df], axis=0)
concat_df = train_df


concat_df.shape


concat_df.keys()


def apply_empty(r):
    if r["thought"] == "fail":
        return "<empty>"
    return r["thought"]


concat_df["thought"] = concat_df.apply(apply_empty, axis=1)


concat_df.head()


data = Dataset.from_pandas(concat_df)

# data = load_dataset("Abirate/english_quotes")
# data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)


CUTOFF_LEN = 4096


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
useful:{data_point['useful']}||thought:{data_point['thought']}||answer:{data_point['answer']}[</END>] """


# EXAMPLE
context = """Eunice Fay McKenzie (February 19, 1918 – April 16, 2019) was an American actress and singer. She also entertained the troops with her former screen partner, Gene Autry. ===Later career=== After World War II, McKenzie retired from films to raise her two children. She was briefly billed as Fay Shannon. ==Biography== ===Early life and silent film=== McKenzie was born on February 19, 1918, in Hollywood, California, to show business parents, film actor Eva (née Heazlitt) and Irish American actor/director Robert McKenzie.Mike Fitzgerald, "An Interview with ... She starred in silent films as a child, and then sound films as an adult, but perhaps she is best known for her leading roles opposite Gene Autry in the early 1940s in five horse opera features. Fay\'s sister Ida Mae McKenzie, cousin Ella McKenzie, and brother-in-law Billy Gilbert, were also actors."""

generate_prompt(
    dict(
        context=context,
        prompt="""What is the main subject of the documentary film "Black White + Gray: A Portrait of Sam Wagstaff and Robert Mapplethorpe"?""",
        A="""The American museum curator Sam Wagstaff.""",
        B="""The debate about public funding for the arts.""",
        C="""The New York art world of the 1970s.""",
        D="""The relationship between Sam Wagstaff and Robert Mapplethorpe.""",
        E="""The controversial fine art images by Robert Mapplethorpe.""",
        useful=True,
        answer="A",
        thought="...",
    )
)


generate_prompt(data[0])


def tokenize(prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < CUTOFF_LEN
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


# print(f"{tokenizer.eos_token_id=}")
# print(f"{tokenizer.sos_token_id=}")

def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt, add_eos_token=True)
#     print(tokenizer.decode(tokenized_full_prompt["input_ids"]))
#     exit()
    return tokenized_full_prompt


# generate_and_tokenize_prompt(data[0])


data.shuffle()


data = data.map(lambda samples: generate_and_tokenize_prompt(samples))




class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(
                state.best_model_checkpoint, "sft_lora_model"
            )
        else:
            checkpoint_folder = os.path.join(
                output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )

        peft_model_path = os.path.join(checkpoint_folder, "sft_lora_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        peft_model_path = os.path.join(args.output_dir, "sft_lora_model")
        kwargs["model"].save_pretrained(peft_model_path)
        if "tokenizer" in kwargs and kwargs["tokenizer"] is not None: 
            kwargs["tokenizer"].save_pretrained(peft_model_path)
        else:
            print("Tokenizer not provided in kwargs!")


# needed for gpt-neo-x tokenizer
tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        # warmup_steps=2,
        warmup_ratio=0.03,
        # max_steps=20,
        num_train_epochs=0.1,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="steps",
        output_dir=lora_output_dir,
        optim = "paged_adamw_8bit",
        lr_scheduler_type = "cosine",
        report_to=None,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.add_callback(SavePeftModelCallback)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

trainer.train()

model.cpu()
del model

##
print("Merge and save...")
time.sleep(1)

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

peft_model_id = os.path.join(lora_output_dir, "sft_lora_model")

config = PeftConfig.from_pretrained(peft_model_id)
model = PeftModel.from_pretrained(model, peft_model_id, torch_dtype=torch.bfloat16)

model = model.merge_and_unload()

merged_model_path = os.path.join(lora_output_dir, "merged_model")
model.save_pretrained(merged_model_path, max_shard_size="2GB")

if True:
    tokenizer.save_pretrained(merged_model_path)
