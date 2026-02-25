# StockAnalyzer

## Fine-Tuned Indian Stock Market Beginner Mentor

This repository now includes a beginner-friendly class project for building an Indian stock market educational assistant using TinyLlama + LoRA on Google Colab free GPU.

### Key files
- `docs/fine_tuned_indian_stock_market_beginner_mentor.md` — full step-by-step guide (LoRA explanation, dataset details, Colab training cells, inference, FastAPI, presentation notes).
- `data/indian_stock_mentor_dataset.jsonl` — 220 instruction-response pairs in JSONL.
- `notebooks/colab_finetune_tinyllama_lora.py` — Colab cell-by-cell training script.
- `backend/main.py` — FastAPI app with `/ask` endpoint.
