from transformers import AutoTokenizer, AutoModel

# Load pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Load pre-trained model and configure it to output hidden states
model = AutoModel.from_pretrained('gpt2', output_hidden_states=True)

print("Tokenizer and Model loaded successfully.")




import torch

# 1. Define sample input data
sample_sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "Transformers are powerful models for natural language processing.",
    "Residual connections are crucial for deep neural networks."
]

# 2. Tokenize the input data
# Using the tokenizer loaded in the previous step
input_ids = tokenizer.encode_plus(
    sample_sentences,
    return_tensors='pt',  # Return PyTorch tensors
    padding=True,         # Pad to the longest sequence in the batch
    truncation=True       # Truncate to the maximum input length the model can handle
)

print("Sample sentences:", sample_sentences)
print("Tokenized input IDs shape:", input_ids['input_ids'].shape)
print("Tokenized attention mask shape:", input_ids['attention_mask'].shape)
print("Tokenized input keys:", input_ids.keys())
