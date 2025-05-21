# train_dpo.py
from trl import DPOTrainer, DPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch

# ── Config ──────────────────────────────────
MODEL_NAME = "mistralai/Mistral-7B-v0.1"  # "EleutherAI/pythia-6.9b" "mistralai/Mistral-7B-v0.1" "meta-llama/Meta-Llama-3.1-8B"
DATA_PATH = "data/dpo_train.jsonl"
OUTPUT_DIR = "outputs/mistral_dpo_lora"
BATCH_SIZE = 4
EPOCHS = 2
LR = 5e-5
MAX_LEN = 128

# ── Load tokenizer & base model (4-bit) ─────
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", quantization_config=bnb, torch_dtype=torch.float16
)

# ── Add LoRA adapter ───────────────────────
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=8,
    # Llama 3.1 and Mistral specific
    target_modules=["q_proj", "v_proj"],
    # Pythia specific
    # target_modules=["query_key_value", "dense"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(base, lora_cfg)

# ── Load DPO dataset ───────────────────────
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# ── DPO and Training args ─────────────────
dpo_cfg = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    max_prompt_length=MAX_LEN,
    max_length=MAX_LEN,
    max_completion_length=MAX_LEN,
    beta=0.1,
    save_strategy="no",
    logging_steps=10,
    bf16=False,
    fp16=True,
    report_to="none",
    label_names=["chosen", "rejected"],
)

trainer = DPOTrainer(
    model=model, args=dpo_cfg, train_dataset=dataset, processing_class=tokenizer
)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ DPO-LoRA training complete.")
