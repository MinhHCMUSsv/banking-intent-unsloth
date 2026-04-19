# 🏦 Banking Intent Classification with Llama 3.1 & Unsloth

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Unsloth-green)](https://github.com/unslothai/unsloth)
[![Model](https://img.shields.io/badge/Model-Llama--3.1--8B-orange)](https://huggingface.co/meta-llama/Llama-3.1-8B)

This project focuses on fine-tuning the **Llama-3.1-8B** model using **LoRA** (Low-Rank Adaptation) via the **Unsloth** library. The goal is to build a high-performance intent detection system for banking customer service queries based on the `PolyAI/banking77` dataset.

---

## 📺 Video Demonstration
> **IMPORTANT:** This video demonstrates the execution of the inference script, accuracy results on the test set, and real-time interaction.
> 
> 🔗 **[Click here to watch the Video Demo](DÁN_LINK_GOOGLE_DRIVE_CỦA_BẠN_VÀO_ĐÂY)**

---

## 📁 Project Structure
The repository follows the mandatory organizational format:

```text
banking-intent-unsloth/
├── configs/
│   ├── train.yaml          # Hyperparameters & data config for training
│   └── inference.yaml      # Model path & config for inference
├── scripts/
│   ├── preprocess_data.py  # Data downloading and splitting logic
│   ├── train.py            # LoRA fine-tuning implementation
│   └── inference.py        # IntentClassification class & evaluation
├── sample_data/
│   ├── train.csv           # Training set
|   ├── val.csv             # Validation set
│   └── test.csv            # Testing set (100 samples)
├── train.sh                # Shortcut for data prep + training
├── inference.sh            # Shortcut for running evaluation/demo
├── requirements.txt        # Environment dependencies
└── README.md               # Documentation
```
## 🛠️ 1. Environment Setup
It is highly recommended to use Google Colab (T4 GPU) or a Linux environment with an NVIDIA GPU and CUDA 12+ installed.
```bash
!git clone https://github.com/MinhHCMUSsv/banking-intent-unsloth.git
%cd banking-intent-unsloth
pip install -r requirements.txt
```
## 🚀 2. Data Preparation & Training
The process uses the PolyAI/banking77 dataset and the configurations specified in configs/train.yaml (utilizing 4-bit quantization).
To fully automate the data downloading, preprocessing, and model fine-tuning, simply run:
```bash
!bash train.sh
```
(Note: This single command sequentially executes preprocess_data.py to generate the CSV files in sample_data/, and then runs train.py to fine-tune the model. The final adapter weights will be saved in the llama_lora/ directory).
## 🧪 3. Inference & Evaluation
This section evaluates the model's accuracy on the independent test set (100 samples) and launches an interactive demo.
```bash
!bash inference.sh
```
### ⏩ Alternative: Quick Inference on Google Colab (Skip Training)
If you just want to run the inference and evaluate the model without going through the entire training pipeline, you can download the pre-trained LoRA weights directly from my public Google Drive and run it on Google Colab.

**Steps to run on Colab:**
1. Open a new Google Colab notebook and enable the **T4 GPU**.
2. Clone this repository and install the requirements:
   ```bash
   git clone https://github.com/MinhHCMUSsv/banking-intent-unsloth.git
   %cd banking-intent-unsloth
   !pip install -r requirements.txt
   ```
3. Download the fine-tuned llama_lora.zip from my public Drive, then upload on Colab and extract it:
   ```bash
   !unzip llama_lora.zip -d llama_lora
   !mv llama_lora/llama_lora/* llama_lora/
   !rm -rf llama_lora/llama_lora/
   ```
4. Run the inference script immediately:
   ```bash
   !bash inference.sh
   ```
## ⚙️ Hyperparameters Configuration
Below are the specific hyperparameters and configurations used for fine-tuning the model (defined in `configs/train.yaml`):

* **Batch size:** `2` (with `gradient_accumulation_steps: 4`, resulting in an effective batch size of 8).
* **Learning rate:** `2e-4` (0.0002) using a `linear` learning rate scheduler.
* **Optimizer:** `adamw_8bit` (optimized for memory efficiency).
* **Number of training epochs:** `1` epoch.
* **Maximum sequence length:** `2048` tokens.
* **Regularization techniques:** * Weight decay set to `0.001`.
  * LoRA adaptation used for parameter-efficient fine-tuning (Rank `r=16`, `alpha=16`, `dropout=0`).
* **Augmentation techniques:** No explicit data augmentation was applied. The dataset was formatted using standard Alpaca instruction-response templates.