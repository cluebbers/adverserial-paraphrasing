# Adversarial Paraphrasing Red-Teaming for LLaMA, Mistral & Pythia

This repository delivers a reproducible pipeline to evaluate and improve “refusal” behavior in three open-weight LLMs—LLaMA-3.1-8B, Mistral-7B-v0.1, and Pythia-6.9B—under adversarial paraphrasing. [Full technical report (PDF)](2025-05-09_Luebbers_report.pdf)

## 🚀 Key Features

- **Prompt set**: 64 harmful base prompts × 4 variants (canonical, lexical, syntactic, semantic), including six real-world case studies (e.g. Tokyo sarin, Unit 731, Unabomber).
- **Evaluation scripts**:
  - `run_inference.py` — batch-runs all prompts through any base model/pipeline.
  - `run_inference_lora.py`- batch-run with lora adapters
  - `annotate_outputs.py` — interactive refusal/harm labeling.
  - `evaluation.ipynb` — computes refusal and harmfulness rates, generates publication-quality bar charts.
- **Alignment adapters**: LoRA rank-8 checkpoints for both
  - **SFT** on 580 prompt→refusal pairs, and
  - **DPO** on 580 prompt­–chosen_vs_rejected triples.
- **Results**:
  - **Baseline** refusal: 2–14 \%; harmful: up to 62 \%.
  - **DPO** gains: modest (+4–38 \% refusal; –24–40 \% harm).
  - **SFT** gains: dramatic (+60–96 \% refusal; harmful ≤ 16 \%).

## 📂 Repository Structure

```text
.
├── data/
│   ├── base_prompts.json           # 64 prompts
│   ├── paraphrased_prompts.json    # 64 prompts × 4 variants
│   ├── dpo_train.jsonl             # 580 DPO triples
│   └── sft_train.jsonl             # 580 SFT doubles
├── scripts/
│   ├── run_inference.py
│   ├── run_inference_lora.py
│   ├── annotate_outputs.py
│   ├── evaluation.ipynb
│   ├── train_dpo.py
│   └── train_sft.py
├── figures/
│   ├── refusal_harmful_rates.pdf
│   └── paraphrase_types.pdf
├── 2025-05-09_Luebbers_report.pdf
├── requirements.txt
└── README.md
```

## 🛠️ Quickstart

Tested on

```bash
torch==2.6.0,
transformers==4.51.3
datasets==3.5.0
accelerate==1.6.0
bitsandbytes==0.45.5
matplotlib==3.10.1
trl==0.17.0
peft==0.15.2
```

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Get model access:**

   <https://huggingface.co/meta-llama/Llama-3.1-8B>
   <https://huggingface.co/mistralai/Mistral-7B-v0.1>

3. **Run inference:**

   possible models:
   "pythia": "EleutherAI/pythia-6.9b"
   "mistral": "mistralai/Mistral-7B-v0.1"
   "llama": "meta-llama/Meta-Llama-3.1-8B"

   ```bash
   python scripts/run_inference.py \
     --model llama
   ```

   and adapters either "sft" or "dpo"

   ```bash
   python scripts/run_inference_lora.py \
      --model llama \
      --adapter dpo
   ```

4. **Annotate outputs:**
   You need to specify the input and output files in the script

   ```bash
   python scripts/annotate_outputs.py
   ```

5. **Inspect results** with scripts/evaluation.ipynb

## 📑 Key Findings

Paraphrase-aware SFT yields the largest safety gains with minimal compute.
Even with only 580 examples, SFT yields near-perfect refusal on all three models.

|  Method  | Avg. Refusal ↑ | Avg. Harm ↓ |
| :------: | :------------: | :---------: |
| Baseline |      6 \%      |    41 \%    |
|   DPO    |     17 \%      |    22 \%    |
|   SFT    |     89 \%      |    8 \%     |

![Model Alignment Results](figures/refusal_harmful_rates.pdf)

## 📖 Citing This Work

```bibtex
@article{lubbers2025refusal,
  title={Evaluating Refusal Robustness under Adversarial Paraphrasing},
  author={Luebbers, Christopher L.},
  year={2025},
  howpublished={\url{https://github.com/cluebbers/adverserial-paraphrasing}}
}
```

---

Feel free to explore, adapt, or extend this toolkit for your own red-teaming and alignment research!
