#!/usr/bin/env python
# coding: utf-8

# # The PoyoModel3000
# This jupyter notebook will contain the model itself.

# In[1]:


import sys
sys.path.append('.')
from minbpe import BasicTokenizer

tokenizer = BasicTokenizer()
tokenizer.load(model_file="./output/tokenizer/poyo_tokenizer.model")
def get_vocab_size(tokenizer: BasicTokenizer) -> int:
    vocab = tokenizer.vocab
    special_tokens = tokenizer.special_tokens

    return len(vocab) + len(special_tokens)


# In[2]:


import torch
torch.manual_seed(6969)

block_size = 1024
n_embd = 384
n_head = 12
n_layer = 12
dropout = 0.2
vocab_size = get_vocab_size(tokenizer)
print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.cuda.is_available() and torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
a = torch.tensor([0], dtype=torch.float32, device=device)  # Fixed the syntax error in tensor creation


# # The Head (drama here)

# In[3]:


from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """ one head of self-attention btw I do not have an attention span """

    def __init__(self, head_size: int) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        _, T, _ = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        weights = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        weights = weights.masked_fill(
            self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        weights = F.softmax(weights, dim=-1)  # (B, T, T)
        weights = self.dropout(weights)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = weights @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


# # Multi-Head Attention

# In[4]:


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads: int, head_size: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out


# # The BLOCK 

# In[5]:


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd: int, n_head: int) -> None:
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.self_attention = MultiHeadAttention(n_head, head_size)
        self.feed_forward = FeedFoward(n_embd)
        self.layer_norm_1 = nn.LayerNorm(n_embd)
        self.layer_norm_2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


# # Assembling the model

# In[6]:


class GPTLanguageModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.final_layer_norm = nn.LayerNorm(n_embd)
        self.final_linear_layer = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the model.

        Args:
            input_tokens: Tensor of token indices of shape (batch_size, sequence_length)
            targets: Optional tensor of target token indices of same shape as input_tokens

        Returns:
            Tuple of (logits, loss) where logits has shape (batch_size, sequence_length, vocab_size)
            and loss is optional cross-entropy loss if targets are provided
        """

        B, T = input_tokens.shape

        # input_tokens and targets are both (B,T) tensor of integers
        token_embedding = self.token_embedding_table(input_tokens)  # (B,T,C)
        positional_embedding = self.position_embedding_table(
            torch.arange(T, device=device))  # (T,C)
        x = token_embedding + positional_embedding  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.final_layer_norm(x)  # (B,T,C)
        logits = self.final_linear_layer(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, input_tokens: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
                Generate new tokens given a context.

                Args:>ns: Starting token indices of shape (batch_size, sequence_length)
                        max_new_tokens: Number of new tokens to generate

                Returns:
                        Tensor of token indices of shape (batch_size, sequence_length + max_new_tokens)
                """

        # input_tokens is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop input_tokens to the last block_size tokens
            cropped_input = input_tokens[:, -block_size:]
            # get the predictions
            logits, _ = self(cropped_input)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            input_tokens = torch.cat(
                (input_tokens, idx_next), dim=1)  # (B, T+1)
        return input_tokens


# In[7]:


model = GPTLanguageModel()
model = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')


# I'll help you create a training loop for the PoyoLLM model. First, let's create a training function that includes learning rate scheduling, gradient clipping, and proper device handling.

# In[8]:


import torch
from torch.optim import AdamW
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm

def train_model(
    model,
    train_data,
    val_data=None,
    n_epochs=5,
    batch_size=32,
    learning_rate=3e-4,
    max_grad_norm=1.0,
    warmup_steps=2000,
    eval_interval=500,
    save_interval=1000,
    checkpoint_dir='checkpoints'
):
    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    def get_lr(step, warmup_steps, learning_rate):
        # Linear warmup followed by cosine decay
        if step < warmup_steps:
            return learning_rate * step / warmup_steps
        return learning_rate * 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (n_epochs * len(train_data) - warmup_steps)))

    # Training loop
    step = 0
    best_val_loss = float('inf')

    for epoch in range(n_epochs):
        model.train()
        pbar = tqdm(range(0, len(train_data), batch_size), desc=f'Epoch {epoch+1}/{n_epochs}')

        for i in pbar:
            # Get batch
            batch_data = train_data[i:i+batch_size]
            if isinstance(batch_data, torch.Tensor):
                x = batch_data
                y = batch_data
            else:
                x = torch.tensor(batch_data, dtype=torch.long, device=device)
                y = torch.tensor(batch_data, dtype=torch.long, device=device)

            # Forward pass
            logits, loss = model(x, y)

            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Update learning rate
            lr = get_lr(step, warmup_steps, learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Update weights
            optimizer.step()

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{lr:.2e}'})

            # Evaluate on validation set
            if val_data is not None and step % eval_interval == 0:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for j in range(0, len(val_data), batch_size):
                        val_batch = val_data[j:j+batch_size]
                        if isinstance(val_batch, torch.Tensor):
                            val_x = val_batch
                            val_y = val_batch
                        else:
                            val_x = torch.tensor(val_batch, dtype=torch.long, device=device)
                            val_y = torch.tensor(val_batch, dtype=torch.long, device=device)
                        _, val_loss = model(val_x, val_y)
                        val_losses.append(val_loss.item())

                avg_val_loss = np.mean(val_losses)
                print(f'\nStep {step}: Validation loss: {avg_val_loss:.4f}')

                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_val_loss,
                    }, f'{checkpoint_dir}/best_model.pt')

                model.train()

            # Save periodic checkpoint
            if step % save_interval == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, f'{checkpoint_dir}/checkpoint_{step}.pt')

            step += 1

    return model

print("Training function created successfully!")


# I'll create a helper function that loads the vocabulary and prepares the training data. Let me build this in steps.

# In[17]:


def load_and_split_data(tokenizer, split_ratio=0.9, sequence_length=block_size):
    # Load all text from decoder vocab
    vocab_list = list(tokenizer.vocab.keys())

    # Convert text to token indices
    tokens = []
    for word in vocab_list:
        token_ids = tokenizer.encode(word)
        tokens.extend(token_ids)

    # Convert to tensor
    data = torch.tensor(tokens, dtype=torch.long, device=device)

    # Create sequences of fixed length
    n_sequences = len(data) - sequence_length
    sequences = torch.stack([data[i:i+sequence_length] for i in range(n_sequences)])

    # Split into train and validation
    split_idx = int(len(sequences) * split_ratio)
    train_data = sequences[:split_idx]
    val_data = sequences[split_idx:]

    print(f"Total sequences: {len(sequences)}")
    print(f"Training sequences: {len(train_data)}")
    print(f"Validation sequences: {len(val_data)}")

    return train_data, val_data

# Test the function
train_data, val_data = load_and_split_data(tokenizer)


# In[20]:


def load_and_split_data(tokenizer, split_ratio=0.9, sequence_length=block_size):
    # Get all text from vocab (converting integers to strings if necessary)
    vocab_list = []
    for word in tokenizer.vocab.keys():
        if isinstance(word, int):
            vocab_list.append(str(word))
        else:
            vocab_list.append(word)

    # Join all words with spaces
    text = " ".join(vocab_list)

    # Convert text to token indices
    tokens = tokenizer.encode(text)

    # Convert to tensor
    data = torch.tensor(tokens, dtype=torch.long, device=device)

    # Create sequences of fixed length
    n_sequences = len(data) - sequence_length
    if n_sequences <= 0:
        raise ValueError(f"Data length ({len(data)}) is shorter than sequence length ({sequence_length})")

    sequences = torch.stack([data[i:i+sequence_length] for i in range(n_sequences)])

    # Split into train and validation
    split_idx = int(len(sequences) * split_ratio)
    train_data = sequences[:split_idx]
    val_data = sequences[split_idx:]

    print(f"Total sequences: {len(sequences)}")
    print(f"Training sequences: {len(train_data)}")
    print(f"Validation sequences: {len(val_data)}")

    return train_data, val_data

# Test the function
train_data, val_data = load_and_split_data(tokenizer)


# In[23]:


# Now we can try training the model with our data
from pathlib import Path

# Create checkpoints directory if it doesn't exist
Path("checkpoints").mkdir(exist_ok=True)

# Start training
train_model(
    model=model,
    train_data=train_data,
    val_data=val_data,
    n_epochs=3,
    batch_size=32,
    learning_rate=3e-4,
    warmup_steps=100,
    eval_interval=100,
    save_interval=500,
    checkpoint_dir='checkpoints'
)


# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=5a923484-3c6f-40ab-ba4a-906a4dff832d' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>

# <div style="font-size: 10px; color: #666; text-align: right;">This notebook was converted with <a href="https://convert.ploomber.io">convert.ploomber.io</a></div>
