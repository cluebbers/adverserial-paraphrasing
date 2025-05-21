import json
import torch
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


INPUT_FILE = Path("data/paraphrased_prompts.json")

# Settings
BATCH_SIZE = 8
MAX_NEW_TOKENS = 100


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with specified model")
    parser.add_argument(
        "--model",
        type=str,
        choices=["pythia", "mistral", "llama"],
        required=True,
        help="Model to use for inference",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        choices=["sft", "dpo"],
        required=True,
        help="Adapter to use for inference",
    )
    return parser.parse_args()


def get_model_path(model):
    model_paths = {
        "pythia": "EleutherAI/pythia-6.9b",
        "mistral": "mistralai/Mistral-7B-v0.1",
        "llama": "meta-llama/Meta-Llama-3.1-8B",
    }
    return model_paths[model]


def get_lora_path(model, adapter):
    lora_paths = {
        "llama": {
            "dpo": "cluebbers/Llama-3.1-8B-adverserial-paraphrasing-dpo",
            "sft": "cluebbers/Llama-3.1-8B-adverserial-paraphrasing-sft",
        },
        "mistral": {
            "dpo": "cluebbers/Mistral-7B-v0.1-adverserial-paraphrasing-dpo",
            "sft": "cluebbers/Mistral-7B-v0.1-adverserial-paraphrasing-sft",
        },
        "pythia": {
            "dpo": "cluebbers/pythia-6.9b-adverserial-paraphrasing-dpo",
            "sft": "cluebbers/pythia-6.9b-adverserial-paraphrasing-sft",
        },
    }
    return lora_paths[model][adapter]


def main():
    args = parse_args()
    MODEL_NAME = get_model_path(args.model)
    LORA_DIR = get_lora_path(args.model, args.adapter)
    OUTPUT_FILE = Path(f"outputs/raw_outputs_{args.model}_{args.adapter}.json")

    # Load model
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto", quantization_config=bnb, torch_dtype="auto"
    )
    model = PeftModel.from_pretrained(base, LORA_DIR)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load prompts
    prompts = json.load(INPUT_FILE.open())
    prompt_texts = [p["paraphrase"] for p in prompts]

    # Move model to eval
    model.eval()

    # Outputs
    outputs = []

    # Batched generation
    for i in range(0, len(prompts), BATCH_SIZE):
        batch_prompts = prompt_texts[i : i + BATCH_SIZE]
        batch_metadata = prompts[i : i + BATCH_SIZE]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to("cuda")

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded_outputs = tokenizer.batch_decode(generated, skip_special_tokens=True)

        for meta, text in zip(batch_metadata, decoded_outputs):
            outputs.append(
                {
                    "prompt_id": meta["prompt_id"],
                    "paraphrase_type": meta["variant_type"],
                    "prompt": meta["paraphrase"],
                    "completion": text,
                }
            )

    # Save
    Path(OUTPUT_FILE.parent).mkdir(exist_ok=True)
    json.dump(outputs, OUTPUT_FILE.open("w"), indent=2)
    print(f"✅ Saved {len(outputs)} completions to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
