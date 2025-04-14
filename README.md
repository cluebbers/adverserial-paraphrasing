# Evaluating Refusal Robustness in LLaMA 3.1 8B under Adversarial Paraphrasing

## 1. Research Goal

This project investigates how paraphrased versions of harmful prompts affect the refusal behavior of LLaMA 3.1 8B. Models are often evaluated using canonical unsafe prompts, yet real-world use involves diverse linguistic variations that could bypass these defenses. We will analyze whether small language models generalize refusal behavior across lexical, syntactic, and semantic paraphrase types. This research aims to expose brittleness in current alignment techniques and provide a lightweight testbed for safety benchmarking. It is relevant for red-teamers, alignment researchers, and AI safety engineers focused on robustness and adversarial evaluation.

---

## 2. Outline of the Final Report

### Title  
**Evaluating Refusal Robustness in LLaMA 3.1 8B under Adversarial Paraphrasing**

---

### Introduction  
- Motivation: Why refusal behavior is critical to LLM safety  
- Problem: Canonical prompts do not reflect realistic attack surface  
- Goal: Test whether reworded harmful prompts reliably bypass refusal in LLaMA 3.1 8B  
- Relevance: Links to robustness evaluation, red-teaming, and surface-level alignment concerns

---

### Research Question  
Can LLaMA 3.1 8B consistently refuse to respond to harmful instructions when they are rephrased? Which types of paraphrasing are most effective at bypassing refusal?

---

### Methodology  
- **Model**: LLaMA 3.1 8B in quantized format (or with LoRA adapters), running locally  
- **Prompt Set**: ~10 harmful base prompts (e.g., jailbreaks, unethical advice)  
- **Paraphrasing Types**:  
  - Lexical (synonyms, rewording)  
  - Syntactic (structure manipulation)  
  - Semantic drift (obfuscated intent)  
- **Evaluation**:  
  - Manual labeling for `refusal`, `harmfulness`, and optional notes  
  - Category-wise breakdown of failure rates  
  - Example outputs and qualitative observations

---

### Results (To be added in Week 10)
- Table of refusal/harmfulness by paraphrase type  
- Notable failure examples  
- Trends in model behavior and alignment brittleness

---

### Discussion (To be added in Week 10)
- Which paraphrase strategies are most dangerous?  
- What does this tell us about how models learn refusal?  
- Implications for safety alignment, red-teaming tools, and evaluation practices

---

### Conclusion (To be added in Week 11)
- Summary of findings  
- Methodological contributions (e.g., paraphrase eval pipeline)  
- Suggestions for improving paraphrase-robust alignment in small models

---

### Repository Contents
- 🗃 `data/`: Base prompts + paraphrased variants  
- ⚙️ `scripts/`: Inference + scoring tools (planned)  
- 📑 `README.md`: Full report and methodology  
- 📈 `results/`: Output logs and plots (planned)

---

### Status (Week 9)

- [ ] Base prompts: Not yet created  
- [ ] Paraphrases: Not yet written  
- [ ] Model setup: Not yet started  
- [ ] Inference: Not started  
- [ ] Annotation: Not started  
