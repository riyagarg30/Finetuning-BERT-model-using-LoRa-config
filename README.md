# 📚 LoRA Fine-Tuning of RoBERTa for AGNEWS Text Classification

This repository contains the code and report for our **Deep Learning Spring 2025** project: a **parameter-efficient fine-tuning of RoBERTa using LoRA (Low-Rank Adaptation)** on the AGNEWS dataset. The primary objective is to adapt a large pre-trained language model for downstream classification using under **1 million trainable parameters**, while achieving competitive accuracy with minimal computational cost.

> 🔗 **Project Members:**  
> - Pranav Bhatt  
> - Kevin Mai  
> - Riya Garg  
> 📄 [Project Report](./report.pdf)

---

## 🧠 Motivation

Transformer-based language models like BERT and RoBERTa excel in NLP tasks but are resource-intensive to fine-tune. Our goals:
- Reduce training cost by adapting only a small subset of parameters using LoRA.
- Retain generalization and prevent overfitting.
- Stay within a **strict <1M trainable parameters** constraint.
- Benchmark optimizers and layerwise learning-rate strategies.

---

## 🚀 Overview

| Task                | Description                                     |
|---------------------|-------------------------------------------------|
| Model               | `roberta-base` (125M parameters, frozen)        |
| Fine-Tuning Method  | LoRA adapters (r=8, α=32, dropout=0.05)         |
| Target Layers       | Attention query & value in layers 8–11          |
| Dataset             | AGNEWS (4-class news topic classification)      |
| Optimizers Compared | AdamW, RMSprop                                  |
| Trainable Params    | ~691,722 (~0.55% of model)                      |

---

## 🛠️ Methodology

### 🔧 LoRA Adapter Configuration
- **Rank (r):** 8  
- **Alpha (α):** 32  
- **Dropout:** 0.05  
- **Target Modules:** `roberta.encoder.layer.[8–11].attention.self.{query,value}`

LoRA introduces trainable low-rank matrices that perturb frozen weights, enabling task-specific fine-tuning without forgetting pre-trained knowledge.

### 🔍 Layerwise Learning Rates

| Layer Range       | Learning Rate |
|-------------------|---------------|
| Layers 0–3        | 1e-5          |
| Layers 4–7        | 3e-4          |
| Layers 8–11       | 5e-4          |
| Classifier Head   | 1e-3          |

This schedule helps preserve low-level semantic understanding while enabling high-level adaptation.

### ⚙️ Optimizer & Scheduler
- **Best performing optimizer:** `RMSprop` with `weight_decay = 0.01`
- **Scheduler:** Linear warmup (10% of steps) + linear decay
- **Gradient Clipping:** `max_grad_norm = 1.0`
- **Precision:** FP16

---

## 📊 Results

| Metric               | AdamW         | RMSprop       |
|----------------------|---------------|---------------|
| Validation Accuracy  | **94.6%**     | 91.3%         |
| Validation Loss      | 0.185         | 0.266         |
| Kaggle Test Accuracy | 91.7% (est.)  | **92.4%**     |

Although AdamW excelled in validation, it overfit the dataset. RMSprop generalized better on unseen test data.

---

## 📁 Repository Contents

```plaintext
.
├── rmsprop.ipynb             # Full training and evaluation pipeline
├── report.pdf                 # Final project report
├── submission_RMSprop.csv     # Final predictions for Kaggle
├── test_unlabelled.pkl        # Unlabeled test dataset (from competition)
├── README.md                  # This file
├── results/                   # Saved LoRA adapter weights (optional)
