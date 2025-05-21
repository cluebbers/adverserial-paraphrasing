from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch

# --- Config ---
model_name = "mistralai/Mistral-7B-v0.1" # "EleutherAI/pythia-6.9b" "mistralai/Mistral-7B-v0.1" "meta-llama/Meta-Llama-3.1-8B"
dataset_path = "data/sft_train.jsonl"
output_dir = "outputs/mistral_sft_lora"
max_length = 128
num_train_epochs = 3
batch_size = 4

# --- Tokenizer & Model ---
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

base_model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", quantization_config=bnb, torch_dtype=torch.float16
)

# --- Add LoRA ---
lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    # Llama 3.1 and Mistral specific
    target_modules=["q_proj", "v_proj"],
    # Pythia specific
    # target_modules=["query_key_value", "dense"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, lora_config)

# --- Dataset ---
def preprocess(example):
    combined = example["prompt"] + "\n" + example["completion"]
    tokens = tokenizer(combined, truncation=True, max_length=max_length, padding="max_length")
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = load_dataset("json", data_files=dataset_path, split="train").map(preprocess)

# --- Training ---
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    num_train_epochs=num_train_epochs,
    learning_rate=5e-5,
    logging_steps=10,
    save_strategy="no",
    bf16=False,
    fp16=True,
    report_to="none",
    label_names=["chosen", "rejected"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✅ SFT-LoRA saved to {output_dir}")
