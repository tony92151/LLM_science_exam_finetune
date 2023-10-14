# LLM_science_exam_finetune


## train
```bash
accelerate launch -m axolotl.cli.train ./llm_lora.yml --deepspeed deepspeed/zero1.json
```
