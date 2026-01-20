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

# Set the pad token for the tokenizer (GPT-2 tokenizer does not have one by default)
tokenizer.pad_token = tokenizer.eos_token

# 2. Tokenize the input data
# Using the tokenizer loaded in the previous step, now directly call the tokenizer for batch processing
input_ids = tokenizer(
    sample_sentences,
    return_tensors='pt',  # Return PyTorch tensors
    padding=True,         # Pad to the longest sequence in the batch
    truncation=True       # Truncate to the maximum input length the model can handle
)

print("Sample sentences:", sample_sentences)
print("Tokenized input IDs shape:", input_ids['input_ids'].shape)
print("Tokenized attention mask shape:", input_ids['attention_mask'].shape)
print("Tokenized input keys:", input_ids.keys())




import torch

# 1. Initialize two empty lists to store the outputs of the attention sub-layers and the MLP sub-layers
attention_sublayer_outputs = []
mlp_sublayer_outputs = []

# 2. Define two Python functions, save_attn_output and save_mlp_output, to serve as forward hooks.
def save_attn_output(module, input, output):
    # For attention, output is usually a tuple (attention_output, weights), we need the first element
    attention_sublayer_outputs.append(output[0])

def save_mlp_output(module, input, output):
    # For MLP, output is the direct output tensor
    mlp_sublayer_outputs.append(output)

# 3. Create an empty list, hook_handles, to store the references to the registered hooks.
hook_handles = []

# 4. Iterate through each transformer block in model.h
# model.h is the list of GPT2Block modules in GPT2Model
for i, block in enumerate(model.h):
    # 5. Register a forward hook for the attention mechanism of the current block
    handle_attn = block.attn.register_forward_hook(save_attn_output)
    hook_handles.append(handle_attn)
    
    # 6. Register a forward hook for the MLP (feed-forward network) of the current block
    handle_mlp = block.mlp.register_forward_hook(save_mlp_output)
    hook_handles.append(handle_mlp)

# 7. Set the model to evaluation mode and wrap the model inference in a torch.no_grad() block
model.eval()
with torch.no_grad():
    # 8. Pass the input_ids to the model to get the outputs
    # Ensure to unpack the input_ids dictionary
    outputs = model(**input_ids)

# 9. De-register all hooks
for handle in hook_handles:
    handle.remove()

# 10. Extract the all_hidden_states from outputs.hidden_states
all_hidden_states = outputs.hidden_hidden_states

# 11. Initialize two new empty lists, attention_residuals and mlp_residuals
attention_residuals = []
mlp_residuals = []

# 12. Iterate from layer_idx = 0 to len(model.h) - 1 (i.e., for each transformer block):
for layer_idx in range(len(model.h)):
    # a. Retrieve the input to the attention residual for the current layer:
    # all_hidden_states[i+1] corresponds to the input to the i-th transformer block AFTER its initial LayerNorm
    # For GPT-2, the residual connection is LayerNorm(x + Sublayer(x)). The 'x' in the residual is the input
    # to the sublayer before its LayerNorm. However, Hugging Face models often output hidden states
    # *after* the initial LayerNorm of the block (which is the input to the attention). 
    # Let's verify this. GPT2Block's forward method:
    # hidden_states = self.ln_1(hidden_states) + self.attn(hidden_states, ...)
    # The input to attn is `self.ln_1(hidden_states)`. The input to the *residual connection* itself for attention
    # is the `hidden_states` *before* `self.ln_1`. 
    # However, `all_hidden_states[layer_idx]` from `output_hidden_states=True` typically gives the output *after* the block's initial LayerNorm
    # and *before* the first sub-layer of the block. Let's assume `all_hidden_states[layer_idx]` is the input to the attention sub-layer *before* its residual connection.
    # More precisely, `all_hidden_states[layer_idx]` represents the hidden state *before* the i-th Transformer block's self-attention layer.
    # So, input_to_attention_residual refers to the `x` in `x + Sublayer(x)`. For GPT-2, `all_hidden_states[layer_idx]` would be the output of previous block (or embeddings).
    # The actual input to the residual for attention in GPT2 is `hidden_states` BEFORE `self.ln_1`.
    # However, `all_hidden_states[layer_idx]` is the hidden state *before* the i-th block's first sublayer (attention).
    # Let's adjust based on typical Hugging Face `output_hidden_states` meaning:
    # `all_hidden_states[0]` is embeddings
    # `all_hidden_states[1]` is output *after* first block's first layer norm, *before* self-attention. This is what we need for the attention residual input `x`.
    # `all_hidden_states[layer_idx]` represents the input *to* layer `layer_idx` (before its first sublayer, which is attention).
    # The actual input to the *residual* for the attention sublayer of block `i` is `all_hidden_states[i]`.
    input_to_attention_residual = all_hidden_states[layer_idx]
    
    # b. Retrieve the output of the attention sub-layer for the current layer
    attention_sublayer_output = attention_sublayer_outputs[layer_idx]
    
    # c. Calculate the input to the MLP residual for the current layer:
    # For GPT2: `hidden_states = hidden_states + attn_output`. This `hidden_states` then becomes input to MLP residual.
    # The `all_hidden_states[layer_idx]` is the input to the block. After attention and residual, it becomes input to MLP.
    # The input to the MLP residual is the output of the attention block *after* its residual addition.
    # This is `hidden_states` + `attention_sublayer_output` which happens before the MLP LayerNorm.
    input_to_mlp_residual = input_to_attention_residual + attention_sublayer_output
    
    # d. Retrieve the output of the MLP sub-layer for the current layer
    mlp_sublayer_output = mlp_sublayer_outputs[layer_idx]
    
    # e. Append calculated residuals (which are simply the sub-layer outputs before adding back to the skip connection)
    # The 'residual value' is typically considered the output of the sub-layer itself before it's added to the input.
    attention_residuals.append(attention_sublayer_output)
    mlp_residuals.append(mlp_sublayer_output)

print(f"Extracted {len(attention_residuals)} attention sub-layer residuals.")
print(f"Extracted {len(mlp_residuals)} MLP sub-layer residuals.")
print(f"Example attention residual shape: {attention_residuals[0].shape}")
print(f"Example MLP residual shape: {mlp_residuals[0].shape}")

