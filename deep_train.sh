accelerate launch --config_file=./deepspeed/deepspeed_zero2.yaml --num_processes 2 finetune.py \
    --model_name Open-Orca/Mistral-7B-OpenOrca \
    --dataset_name ./llm-science-exam-data-w-thought/train_.csv \
    --dataset_text_field text \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --batch_size 8 \
    --seq_length 2048 \
    --load_in_4bit True \
    --use_peft True \
    --output_dir ./output \
    --peft_lora_r 8 \
    --peft_lora_alpha 16 \
    --num_train_epochs 1 \
    --save_total_limit 2 \
    --save_steps 200

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