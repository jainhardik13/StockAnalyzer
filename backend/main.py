from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "./tinyllama-indian-stock-lora"

app = FastAPI(title="Indian Stock Beginner Mentor API")


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    return tokenizer, model


tokenizer, model = load_model()


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ask_question(payload: AskRequest):
    prompt = (
        "You are a beginner-friendly Indian stock market mentor. "
        "Never give buy/sell advice. Use headings and simple language. "
        "Always end with: This is for educational purposes only and not financial advice.\n\n"
        f"Question: {payload.question}\n\nAnswer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    answer = text.split("Answer:", 1)[-1].strip()
    return {"response": answer}
