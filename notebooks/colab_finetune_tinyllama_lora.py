# Colab Cell 1
# !pip -q install transformers datasets peft accelerate bitsandbytes trl

# Colab Cell 2
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
dataset_path = "/content/indian_stock_mentor_dataset.jsonl"
output_dir = "/content/tinyllama-indian-stock-lora"

# Colab Cell 3

tokenizer = AutoTokenizer.from_pretrained(base_model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Colab Cell 4
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

# Colab Cell 5
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# Colab Cell 6
dataset = load_dataset("json", data_files=dataset_path, split="train")

# Colab Cell 7

def format_example(example):
    return {
        "text": (
            "### Instruction:\n"
            f"{example['instruction']}\n\n"
            "### Response:\n"
            f"{example['response']}"
        )
    }

formatted_dataset = dataset.map(format_example)

# Colab Cell 8
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=2,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
    bf16=False,
    fp16=True,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="none",
)

# Colab Cell 9
trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
)

# Colab Cell 10
trainer.train()

# Colab Cell 11
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Saved adapter at: {output_dir}")
