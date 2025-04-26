import torch
from typing import Tuple

import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F # Not used

hidden_size = 128
ffn_intermediate_ratio = 8 / 3
multiple_of = 32
intermediate_size = int(hidden_size * ffn_intermediate_ratio)
intermediate_size = ((intermediate_size + multiple_of - 1) // multiple_of) * multiple_of
hidden_act = "silu"
rms_norm_eps = 1e-5
ffn_bias = False

# Training Configuration
learning_rate = 1e-4
num_epochs = 50 # Number of training epochs (iterations over the dummy data)
batch_size = 20
sequence_length = 100

print("Configuration:")
print(f"  hidden_size: {hidden_size}")
print(f"  intermediate_size: {intermediate_size}")
print(f"  hidden_act: {hidden_act}")
print(f"  rms_norm_eps: {rms_norm_eps}")
print(f"  learning_rate: {learning_rate}")
print(f"  num_epochs: {num_epochs}")

import tiktoken
from feed_forward import SimplifiedLlama4FFN
from attention_code import SimpleL2Norm, SimplifiedLlama4Attention
import numpy as np
import glob
import os

# --- Data Loading ---
# Note: This assumes the fineweb.py script has been run and data exists in edu_fineweb10B
DATA_CACHE_DIR = "edu_fineweb10B"
train_files = glob.glob(os.path.join(DATA_CACHE_DIR, "edufineweb_train_*.npy"))
val_files = glob.glob(os.path.join(DATA_CACHE_DIR, "edufineweb_val_*.npy"))

# Sort files to process them in order
train_files.sort()
val_files.sort()

# Use the same tokenizer as in fineweb.py to get vocab size
cl100k_base = tiktoken.get_encoding("cl100k_base")

enc = tiktoken.Encoding(
    name="cl100k_poyo",
    pat_str=cl100k_base._pat_str,
    mergeable_ranks=cl100k_base._mergeable_ranks,
    special_tokens={
        "<|endoftext|>": 50256, # end of text token
        "<|pad|>": 50257, # padding token
        "<|sep|>": 50258, # separator token
        "<|system|>": 50259, # AI token
        "<|user|>": 50260, # User token
        "<|assistant|>": 50261, # Assistant token
    },
)

vocab_size = enc.n_vocab
print(f"Tokenizer vocabulary size: {vocab_size}")


def get_batch(split: str, batch_size: int, sequence_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gets a batch of data from the pre-tokenized shards.
    """
    files = train_files if split == "train" else val_files
    if not files:
        raise FileNotFoundError(f"No {split} data files found in {DATA_CACHE_DIR}. Did you run fineweb.py?")

    # For simplicity, load the first shard. A real implementation would iterate through shards.
    # Using memory mapping for potentially large files.
    data = np.load(files[0], mmap_mode='r')

    # Choose random starting points for sequences in the batch
    ix = torch.randint(len(data) - sequence_length, (batch_size,))

    # Create input sequences (x) and target sequences (y)
    x = torch.stack([torch.from_numpy((data[i:i+sequence_length]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+sequence_length]).astype(np.int64)) for i in ix])

    return x, y


# --- Model Definition ---

class TransformerBlock(nn.Module):
    """ A single block of the Transformer """
    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int = 4): # Assuming num_heads, adjust if needed
        super().__init__()
        global sequence_length # Access global sequence_length for max_position_embeddings
        rope_theta = 10000.0 # Common default
        attention_bias = False # Common default
        use_qk_norm = False # Common default

        self.attention = SimplifiedLlama4Attention(config = {
            "hidden_size": hidden_size,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_heads, # Simplified: Set equal to num_attention_heads
            "head_dim": hidden_size // num_heads,
            "max_position_embeddings": sequence_length, # Use defined sequence_length
            "rope_theta": rope_theta,
            "attention_bias": attention_bias,
            "use_qk_norm": use_qk_norm,
        })
        self.feed_forward = SimplifiedLlama4FFN(config = {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "hidden_act": hidden_act, # Use global config
            "ffn_bias": ffn_bias,     # Use global config
            "rms_norm_eps": rms_norm_eps, # Use global config
            "intermediate_size": intermediate_size,
        })
        self.attention_norm = SimpleL2Norm(eps=rms_norm_eps)
        self.ffn_norm = SimpleL2Norm(eps=rms_norm_eps)

    def forward(self, x, attention_mask, position_ids, freqs_cis) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), attention_mask=attention_mask, position_ids=position_ids, freqs_cis=freqs_cis)[0] # Pass required args
        # Feed-forward part
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class SimpleLlama(nn.Module):
    """ Simplified Llama-like model """
    def __init__(self, vocab_size: int, hidden_size: int, intermediate_size: int, num_layers: int = 2): # Using 2 layers for simplicity
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.tok_embeddings = nn.Embedding(vocab_size, hidden_size)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(TransformerBlock(hidden_size=hidden_size, intermediate_size=intermediate_size))

        self.norm = SimpleL2Norm(eps=rms_norm_eps)
        self.output = nn.Linear(hidden_size, vocab_size, bias=False)

        # Precompute RoPE frequencies
        self.freqs_cis = self._precompute_freqs_cis(sequence_length, self.hidden_size // 4) # Assuming num_heads=4

    def _precompute_freqs_cis(self, seq_len: int, n_elem: int, base: int = 10000) -> torch.Tensor:
        """Precomputes the rotary frequency tensor."""
        freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
        t = torch.arange(seq_len, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        # Need to reshape for compatibility with attention layer: (seq_len, n_elem // 2) -> (seq_len, 1, n_elem // 2)
        return freqs_cis # Return the computed frequencies

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = tokens.shape
        h = self.tok_embeddings(tokens)

        # Ensure freqs_cis is on the same device as input
        # Correct slicing: select frequencies up to the current sequence length
        freqs_cis = self.freqs_cis[:seq_len].to(h.device) # Slice up to seq_len, move to device

        # Create position_ids
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=h.device).unsqueeze(0) # Shape: [1, seq_len]
        # position_ids = position_ids.repeat(batch_size, 1) # Shape: [batch_size, seq_len] - Optional, broadcasting might handle it

        # Create attention mask (assuming causal LM)
        # SimplifiedLlama4Attention might handle causal masking internally if mask is None.
        # If an explicit mask is needed:
        # attention_mask = torch.tril(torch.ones(seq_len, seq_len, device=h.device)).unsqueeze(0).unsqueeze(0)
        # attention_mask = attention_mask.expand(batch_size, 1, seq_len, seq_len)
        attention_mask = None # Let's assume None works for causal mask handling in the attention layer

        for layer in self.layers:
            # Pass attention_mask and position_ids along with freqs_cis
            h = layer(h, attention_mask=attention_mask, position_ids=position_ids, freqs_cis=freqs_cis)

        h = self.norm(h)
        output = self.output(h)

        for layer in self.layers:
            h = layer(h, freqs_cis=freqs_cis, attention_mask=attention_mask, position_ids=position_ids)

        h = self.norm(h)
        output = self.output(h)
        return output

# --- Training Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = SimpleLlama(vocab_size, hidden_size, intermediate_size).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss() # Handles logits directly

# --- Training Loop ---
print("\nStarting training...")
model.train()
for epoch in range(num_epochs):
    num_steps_per_epoch = 50
    total_loss = 0.0

    for step in range(num_steps_per_epoch):
        # Get a batch of data
        x, y = get_batch("train", batch_size, sequence_length)
        x, y = x.to(device), y.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(x) # Shape: (batch_size, sequence_length, vocab_size)

        # Calculate loss
        # CrossEntropyLoss expects (N, C) or (N, C, d1, ...) and targets (N) or (N, d1, ...)
        # We need to reshape logits to (batch*seq_len, vocab_size) and targets to (batch*seq_len)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item()

        if (step + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{num_steps_per_epoch}], Loss: {loss.item():.4f}")

        if (step + 1) % 250 == 0:
            model.eval()
            val_loss = 0.0
            num_val_steps = 10 # Evaluate on a few validation batches
            with torch.no_grad():
                for _ in range(num_val_steps):
                    x_val, y_val = get_batch("val", batch_size, sequence_length)
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    logits_val = model(x_val)
                    loss_val = criterion(logits_val.view(-1, vocab_size), y_val.view(-1))
                    val_loss += loss_val.item()
            print(f"Validation Loss: {val_loss / num_val_steps:.4f}")

    avg_loss = total_loss / num_steps_per_epoch
    print(f"--- Epoch {epoch+1} Average Loss: {avg_loss:.4f} ---")

    if (epoch + 1) % 25 == 0:
        # Save the model checkpoint every 5 epochs
        checkpoint_path = f"model_epoch_{epoch+1}.pth"
        os.makedirs("output", exist_ok=True)
        torch.save(model.state_dict(), "output/" + checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")

        # --- Generate some text ---
        model.eval()  # Set the model to evaluation mode
        prompt = "<|system|>You are a helpful AI assistant.<|user|>Hello<|assistant|>"
        encoded_prompt = enc.encode(prompt, allowed_special="all")
        context = torch.tensor(encoded_prompt, dtype=torch.long, device=device).unsqueeze(0)

        generated_tokens = []
        with torch.no_grad():
            for _ in range(50):  # Generate up to 50 tokens
                logits = model(context)
                # Focus only on the last token for prediction
                logits = logits[:, -1, :]  # Shape: (1, vocab_size)
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # Shape: (1, 1)
                generated_tokens.append(next_token.item())
                context = torch.cat([context, next_token.unsqueeze(0)], dim=1)  # Append to context

        # Decode the generated tokens
        generated_text = enc.decode(generated_tokens)
        print(f"Generated Text: {generated_text}")

        model.train()  # Return to training mode

print("Training finished.")