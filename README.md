

# GPT-2 Residual Stream Analysis

This project explores the **Residual Stream** of the GPT-2 architecture. It uses PyTorch forward hooks to intercept the outputs of the Attention and MLP sub-layers to analyze their relative contributions (magnitudes) across different layers of the model.

## 📌 Project Overview

In a Transformer block, the output is calculated as:


This project extracts the  component (the "residual contribution") for both the **Self-Attention** and **Feed-Forward (MLP)** blocks. By calculating the  norm of these residuals, we can visualize how the model "updates" its internal representations at each stage.

## 📂 File Structure

* **`main.py`**: The entry point. Orchestrates model loading, inference, and visualization.
* **`model_utils.py`**: Handles Hugging Face model initialization and the registration/removal of PyTorch forward hooks.
* **`data_utils.py`**: Manages text tokenization and padding logic.
* **`analysis_utils.py`**: Contains the mathematical logic for calculating  norms and processing tensor outputs.
* **`plotting_utils.py`**: Generates Matplotlib visualizations for magnitude distribution and layer-wise evolution.

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* PyTorch
* Transformers (Hugging Face)
* Matplotlib
* NumPy

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/gpt2-residual-analysis.git
cd gpt2-residual-analysis

```


2. Install dependencies:
```bash
pip install torch transformers matplotlib numpy

```



### Running the Analysis

Execute the main script to process the sample sentences and generate the plots:

```bash
python main.py

```

1. Residual Distribution
A histogram showing the frequency of different magnitude ranges for Attention vs. MLP layers.

<p align="center"> <img width="1013" alt="Residual Distribution" src="https://github.com/user-attachments/assets/e622a982-faab-4f9b-8ba5-647dd79d8baf" /> </p>

2. Layer-wise Evolution
A line graph tracking how the average "update size" changes as information flows from the embedding layer (Layer 0) to the final output layer.

<p align="center"> <img width="1026" alt="Layer-wise Evolution" src="https://github.com/user-attachments/assets/0949d45e-8865-4cb4-b0df-b362184f0eb3" /> </p>


## 📝 Findings Note

In many transformer models, you will observe that MLP residuals often have higher magnitudes than Attention residuals in later layers, suggesting that MLPs perform the bulk of the "knowledge processing" while Attention 
focus on "information routing."


---

