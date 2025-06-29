# 📚 LoRA Fine-Tuning of RoBERTa for AGNEWS Text Classification

This repository contains the code and report for our Deep Learning Spring 2025 project: **parameter-efficient fine-tuning of RoBERTa using LoRA (Low-Rank Adaptation)** on the AGNEWS dataset. The primary goal is to adapt a large pre-trained language model for downstream classification using under **1 million trainable parameters**, achieving competitive accuracy with minimal compute.

> 🔗 **Project Members:**  
> - Pranav Bhatt  
> - Kevin Mai  
> - Riya Garg  
> 📄 [Project Report](./report.pdf)

---

## 🧠 Motivation

Transformer-based language models like BERT and RoBERTa have achieved state-of-the-art results in NLP tasks, but their size makes fine-tuning resource-intensive. Our goal is to:
- Reduce training cost by adapting only a small subset of parameters using LoRA.
- Retain model generalization with minimal overfitting.
- Operate under the strict constraint of **< 1M trainable parameters**.
- Benchmark different optimizers and learning-rate strategies.

---

## 🚀 Overview

| Task              | Description                                    |
|-------------------|------------------------------------------------|
| Model             | `roberta-base` (125M parameters, frozen)      |
| Fine-Tuning Method| LoRA adapters (r=8, α=32, dropout=0.05)       |
| Target Layers     | Attention query & value layers in layers 8–11 |
| Dataset           | AGNEWS (4-class news topic classification)    |
| Optimizers Tried  | AdamW, RMSProp                                |
| Max Trainable Params | ~691,722                                   |

---

## 🛠️ Methodology

### 🔧 LoRA Adapter Configuration
- **Rank (r):** 8
- **Alpha (α):** 32
- **Dropout:** 0.05
- **Target Modules:** `roberta.encoder.layer.[8-11].attention.self.{query,value}`

LoRA introduces trainable low-rank matrices to perturb frozen weights, enabling task-specific adaptation without overwriting pretrained knowledge.

### 🔍 Layerwise Learning Rates
| Layer Range        | Learning Rate |
|--------------------|---------------|
| Layers 0–3         | 1e-5          |
| Layers 4–7         | 3e-4          |
| Layers 8–11        | 5e-4          |
| Classifier Head    | 1e-3          |

This configuration preserves low-level semantic features while allowing higher layers to specialize on the classification task.

### ⚙️ Optimizer and Scheduler
- **Best optimizer:** `RMSprop` with weight decay `0.01`
- **Scheduler:** Linear warmup (10% steps) + linear decay
- **Gradient Clipping:** Max norm = 1.0
- **Precision:** FP16

---

## 📊 Results

| Metric              | AdamW        | RMSprop     |
|---------------------|--------------|-------------|
| Validation Accuracy | **94.6%**    | 91.3%       |
| Validation Loss     | 0.185        | 0.266       |
| Kaggle Test Accuracy| 91.7% (est.) | **92.4%**   |

Although AdamW performed better on validation, it **overfitted**. RMSprop generalized better on **unseen Kaggle test data**.

---

## 📁 Repository Contents

```

.
├── notebook.ipynb            # Full training and evaluation pipeline
├── report.pdf                # Final project report
├── submission\_RMSprop.csv    # Final test predictions
├── test\_unlabelled.pkl       # Provided test dataset
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── results/                  # Saved LoRA adapter weights (if included)

````

---

## 📈 Training Curves

The following graphs compare **AdamW** and **RMSprop** across 5 epochs.

| Training Loss | Validation Loss | Validation Accuracy |
|---------------|------------------|----------------------|
| ![](./assets/train_loss.png) | ![](./assets/val_loss.png) | ![](./assets/val_acc.png) |

> You can generate these graphs using the code in the notebook or use the saved `matplotlib` plot.

---

## 📦 Setup Instructions

1. **Clone this repository**
   ```bash
   git clone https://github.com/<your-username>/lora-roberta-agnews.git
   cd lora-roberta-agnews
  ````

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**
   The training code is available in `notebook.ipynb`. Run all cells to fine-tune and evaluate.

4. **Generate Kaggle Submission**

   ```python
   create_submission()
   ```

---

## 🔍 Key Takeaways

* LoRA enables scalable fine-tuning of massive models with <1% trainable parameters.
* RMSprop outperformed AdamW on unseen test data due to better generalization.
* Layer-wise learning rates are **crucial** to balance adaptation and preservation of knowledge.
* Gradient clipping, batch sizing, and dropout help regularize and stabilize LoRA fine-tuning.

---

## 📚 References

* Hu, Edward J., et al. "**LoRA: Low-Rank Adaptation of Large Language Models**." arXiv preprint arXiv:2106.09685 (2021).
* HuggingFace Transformers: [https://huggingface.co/docs/transformers/index](https://huggingface.co/docs/transformers/index)
* PEFT Library (LoRA): [https://github.com/huggingface/peft](https://github.com/huggingface/peft)

---

## 🏁 Acknowledgements

This project was conducted as part of **Deep Learning Spring 2025**, under the guidance of our course instructors and the Kaggle community.
