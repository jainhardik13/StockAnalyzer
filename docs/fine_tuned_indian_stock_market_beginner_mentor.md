# Fine-Tuned Indian Stock Market Beginner Mentor (Beginner Guide)

## STEP 1: What is LoRA fine-tuning (super simple)?

### Full training vs LoRA
- **Full training** means changing almost every model weight. This needs a lot of GPU memory and time.
- **LoRA** (Low-Rank Adaptation) adds a **small set of trainable layers** on top of the base model.
- In LoRA, we freeze the original model and train only these tiny add-on layers.

### Why we use LoRA for this student project
1. It works on **Google Colab free T4 GPU**.
2. It is much faster than full fine-tuning.
3. It is cheaper and easier for beginners.
4. We can share only a small adapter file instead of the full model.

### Why 4-bit quantization
- 4-bit loads model weights in smaller memory form.
- This allows TinyLlama to fit better on Colab free GPU.
- Together, **4-bit + LoRA** is a practical beginner setup.

---

## STEP 2: Dataset creation

- Dataset path: `data/indian_stock_mentor_dataset.jsonl`
- Total pairs: **220 instruction-response samples**
- Format: JSONL (`{"instruction": "...", "response": "...", "category": "..."}`)
- Topics covered:
  - Indian stock basics
  - Risk management
  - Trading vs investing
  - Long-term investing
  - Mutual funds
  - SIP
  - Beginner mistakes
  - Indian terminology
  - Scenario practice

### Dataset quality rules followed
- Every answer is beginner-friendly.
- Every answer uses headings.
- Every answer ends with:
  - `This is for educational purposes only and not financial advice.`
- No direct buy/sell recommendation language.

---

## STEP 3: FULL Google Colab notebook code (cell-by-cell)

> Copy each cell into Colab in the same order.

### Cell 1: Install libraries
```python
!pip -q install transformers datasets peft accelerate bitsandbytes trl
```

### Cell 2: Imports and config
```python
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
dataset_path = "/content/indian_stock_mentor_dataset.jsonl"  # upload this file in Colab
output_dir = "/content/tinyllama-indian-stock-lora"

print(torch.cuda.get_device_name(0))
```

### Cell 3: Load tokenizer
```python
tokenizer = AutoTokenizer.from_pretrained(base_model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

### Cell 4: Load model in 4-bit
```python
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
```

### Cell 5: LoRA config
```python
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
```

### Cell 6: Load JSONL dataset
```python
dataset = load_dataset("json", data_files=dataset_path, split="train")
print(dataset)
print(dataset[0])
```

### Cell 7: Convert to training text prompt
```python
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
print(formatted_dataset[0]["text"][:500])
```

### Cell 8: Training arguments
```python
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
```

### Cell 9: Trainer setup
```python
trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
)
```

### Cell 10: Start training
```python
trainer.train()
```

### Cell 11: Save LoRA adapter + tokenizer
```python
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Saved adapter at: {output_dir}")
```

---

## STEP 4: Inference code (load base model + LoRA adapter)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = "/content/tinyllama-indian-stock-lora"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

question = "What is SIP and why is it useful for beginners in India?"
prompt = (
    "You are a beginner-friendly Indian stock market mentor. "
    "Never give buy/sell advice. Use headings and simple language. "
    "Always end with: This is for educational purposes only and not financial advice.\n\n"
    f"Question: {question}\n\nAnswer:"
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=220,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## STEP 5: Simple FastAPI backend (`/ask`)

File already added in this project: `backend/main.py`

### Run locally
```bash
pip install fastapi uvicorn transformers peft bitsandbytes accelerate torch
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### API request example
```bash
curl -X POST "http://127.0.0.1:8000/ask" \
-H "Content-Type: application/json" \
-d '{"question":"Explain Nifty 50 for a beginner"}'
```

---

## STEP 6: How to present this in class (easy speaking points)

### A) What to say about fine-tuning
- “I started from a pre-trained open-source model (TinyLlama).”
- “Then I gave it a focused education dataset about Indian stock basics.”
- “This makes the model better for one domain, instead of general chat.”

### B) What to say about LoRA
- “LoRA trains small add-on layers, not the full model.”
- “That is why it works on free Colab GPU and is beginner-friendly.”
- “I saved only the adapter and reused the original base model.”

### C) What to say about ethics and safety
- “The model gives educational content, not personal financial advice.”
- “I added a mandatory disclaimer in every training answer.”
- “I avoided prompts that push buy/sell calls.”
- “Users should still consult certified financial professionals for decisions.”

### D) Demo flow in class (2 minutes)
1. Show one dataset row in JSONL.
2. Show LoRA training cell in Colab.
3. Show inference question and response.
4. Show FastAPI `/ask` endpoint.
5. End with disclaimer and ethical note.

---

## Beginner checklist before submission
- [ ] Colab runtime set to **GPU (T4)**.
- [ ] Dataset uploaded to `/content`.
- [ ] Training completed at least 1 epoch.
- [ ] Adapter saved.
- [ ] Inference runs without error.
- [ ] FastAPI endpoint returns formatted answer.
- [ ] Output includes educational disclaimer.
