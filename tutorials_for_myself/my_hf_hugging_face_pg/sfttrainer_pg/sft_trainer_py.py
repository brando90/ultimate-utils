# %%
from datasets import load_dataset
from trl import SFTTrainer

dataset = load_dataset("imdb", split="train")

trainer = SFTTrainer(
    "facebook/opt-350m",
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
)
trainer.train()

#%%
from transformers import AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTTrainer

dataset = load_dataset("imdb", split="train")

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
)

trainer.train()

#%%
"""
if you're training a model that requires prompts e.g. AF, chat bots: https://huggingface.co/docs/trl/main/en/sft_trainer#advanced-usage
"""

"""
Below is an instruction ...

### Instruction
{prompt}

### Response:
{completion}
"""

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['question'])):
        text = f"### Question: {example['question'][i]}\n ### Answer: {example['answer'][i]}"
        output_texts.append(text)
    return output_texts

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
)

trainer.train()

#%%
"""
https://huggingface.co/docs/trl/main/en/sft_trainer#training-adapters
"""

from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig

dataset = load_dataset("imdb", split="train")

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    "EleutherAI/gpt-neo-125m",
    train_dataset=dataset,
    dataset_text_field="text",
    peft_config=peft_config
)

trainer.train()