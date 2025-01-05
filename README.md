# 📜 Text Summarization with Large Language Models

## 🚀 Introduction

This repository implements **text summarization** using cutting-edge transformer-based approaches, tailored for both **abstractive** and **extractive** tasks. It includes:

1. **LLaMA-based Abstractive Summarization**
2. **GPT-2 with LoRA for Abstractive Summarization**
3. **BERT with Sentence Transformer for Extractive Summarization**

By leveraging these techniques, the project showcases efficient summarization on conversational datasets, providing robust evaluations with industry-standard metrics.

---

## 📦 Installation

To set up this project locally, follow these steps:

### Prerequisites
Ensure the following are installed:
- Python 3.x
- Pip
- (Optional) Virtual Environment Tools (e.g., `venv` or `conda`)

### Steps
1. **Clone the repository:**
   ```bash
   git clone <https://github.com/saadsohail05/Text-Summarization-Using-Large-Language-Models.git>
   cd text-summarization-transformers
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify dataset availability:** The Samsum Dataset is used for training and evaluation. It will be automatically downloaded via the 🤗 Datasets library.

---


## ✨ Features

- 🔍 **Abstractive Summarization:** Generates concise summaries with models like LLaMA and GPT-2.
- 📋 **Extractive Summarization:** Identifies key sentences from dialogues using BERT with Sentence Transformers.
- 📈 **Evaluation Metrics:** Measures summary quality with ROUGE-1, ROUGE-2, and ROUGE-L.
- 🔧 **LoRA for GPT-2:** Implements parameter-efficient fine-tuning using Low-Rank Adaptation.
- 🤗 **Hugging Face Integration:** Utilizes state-of-the-art libraries for model training and evaluation.

---

## 📂 Dataset

The project utilizes the Samsum Dataset, which contains conversation dialogues and their corresponding human-written summaries. The dataset is automatically loaded as follows:

```python
from datasets import load_dataset
dataset = load_dataset("samsum")
```

Preprocessing steps include tokenization, padding, and truncation for transformer input formats.

---

## 🧠 Methodology

### 1. LLaMA-based Abstractive Summarization
This approach fine-tunes a LLaMA model to generate summaries for conversational datasets. Key features include:

- **Preprocessing:** Formatting dialogues for tokenization.
- **Fine-Tuning:** Training the LLaMA model using Hugging Face's Trainer.
- **Inference:** Employing text generation pipelines for summary generation.

### 2. GPT-2 with LoRA for Abstractive Summarization
Leverages Low-Rank Adaptation (LoRA) to fine-tune GPT-2 efficiently. Key features include:

- **LoRA Configuration:** Reduces training complexity while improving performance.
- **Dataset Preparation:** Structured prompts for summarization tasks.
- **ROUGE Evaluation:** Accurate comparison of generated and reference summaries.

### 3. BERT with Sentence Transformer for Extractive Summarization
Employs BERT for sequence classification and Sentence Transformers for similarity computation. Key features:

- **Sentence Tokenization:** Splits dialogues into meaningful sentences.
- **Cosine Similarity:** Ranks sentences based on relevance to the dialogue.
- **Thresholding:** Selects the most informative sentences for extractive summarization.

---

## 📊 Results

The models are evaluated using ROUGE metrics:

- **ROUGE-1:** Unigram overlap between generated and reference summaries.
- **ROUGE-2:** Bigram overlap for more context.
- **ROUGE-L:** Longest common subsequence for structural coherence.

---


