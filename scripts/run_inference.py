import json
import torch
import argparse
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)

# ---------- Configuration ----------
INPUT_FILE = Path("data/paraphrased_prompts.json")
MAX_NEW_TOKENS = 100
BATCH_SIZE = 8  # Adjust based on your available VRAM


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with specified model")
    parser.add_argument(
        "--model",
        type=str,
        choices=["pythia", "mistral", "llama"],
        required=True,
        help="Model to use for inference",
    )
    return parser.parse_args()


def get_model_path(model):
    model_paths = {
        "pythia": "EleutherAI/pythia-6.9b",
        "mistral": "mistralai/Mistral-7B-v0.1",
        "llama": "meta-llama/Meta-Llama-3.1-8B",
    }
    return model_paths[model]


def main():
    args = parse_args()
    MODEL_ID = get_model_path(args.model)
    OUTPUT_FILE = Path(f"outputs/raw_outputs_{args.model}.json")

    # ---------- Load Model ----------
    print("🔄 Loading model and tokenizer...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto", torch_dtype="auto"
    )

    # ---------- Create Generator Pipeline ----------
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,  # Prevents infinite loops
    )

    # ---------- Load Prompts ----------
    print(f"📂 Reading prompts from {INPUT_FILE}")
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        prompts = json.load(f)

    prompt_texts = [item["paraphrase"] for item in prompts]

    # ---------- Run Batched Inference ----------
    print(f"🚀 Generating completions for {len(prompts)} prompts...")
    results = generator(
        prompt_texts,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.2,
        batch_size=BATCH_SIZE,
    )

    # ---------- Match Completions to Prompts ----------
    outputs = []
    for item, result_group in zip(prompts, results):
        result = result_group[0]  # take the first (and only) generated result
        outputs.append(
            {
                "prompt_id": item["prompt_id"],
                "paraphrase_type": item["variant_type"],
                "prompt": item["paraphrase"],
                "completion": result["generated_text"],
            }
        )

    # ---------- Save Outputs ----------
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)

    print(f"\n✅ Saved {len(outputs)} completions to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
