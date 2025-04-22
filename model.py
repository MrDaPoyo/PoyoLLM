import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple

# Ensure the project root is in the path
sys.path.append('.')
from minbpe import BasicTokenizer  # Assuming minbpe is a local module

# --- Constants and Configuration ---
BLOCK_SIZE = 1024
N_EMBD = 384
N_HEAD = 12
N_LAYER = 12
DROPOUT = 0.2
SEED = 6969
TOKENIZER_MODEL_PATH = "./output/tokenizer/poyo_tokenizer.model"

torch.manual_seed(SEED)

# --- Device Setup ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# --- Tokenizer Loading ---
tokenizer = BasicTokenizer()
try:
    tokenizer.load(model_file=TOKENIZER_MODEL_PATH)
except FileNotFoundError:
    print(f"Error: Tokenizer model file not found at {TOKENIZER_MODEL_PATH}", file=sys.stderr)
    sys.exit(1)

def get_vocab_size(tokenizer: BasicTokenizer) -> int:
    """Calculates the vocabulary size including special tokens."""
    return len(tokenizer.vocab) + len(tokenizer.special_tokens)

VOCAB_SIZE = get_vocab_size(tokenizer)

# --- Model Definition ---

class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size: int) -> None:
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        # tril is not a parameter, register as buffer
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        # Compute attention scores ("affinities")
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        weights = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        # Mask future positions
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        weights = F.softmax(weights, dim=-1) # (B, T, T)
        weights = self.dropout(weights)

        # Perform the weighted aggregation of the values
        v = self.value(x) # (B, T, head_size)
        out = weights @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, num_heads: int, head_size: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(head_size * num_heads, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Concatenate heads' outputs along the feature dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, num_heads * head_size)
        out = self.dropout(self.projection(out)) # (B, T, N_EMBD)
        return out

class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity."""

    def __init__(self, n_embd: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # Projection back to residual pathway
            nn.Dropout(DROPOUT),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Block(nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, n_embd: int, n_head: int) -> None:
        super().__init__()
        head_size = n_embd // n_head
        self.self_attention = MultiHeadAttention(n_head, head_size)
        self.feed_forward = FeedForward(n_embd)
        # LayerNorm applied before the attention/feedforward layers (Pre-LN)
        self.layer_norm_1 = nn.LayerNorm(n_embd)
        self.layer_norm_2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection around self-attention
        x = x + self.self_attention(self.layer_norm_1(x))
        # Residual connection around feed-forward
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x

class GPTLanguageModel(nn.Module):
    """The main GPT language model."""

    def __init__(self) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(
            *[Block(N_EMBD, n_head=N_HEAD) for _ in range(N_LAYER)]
        )
        self.final_layer_norm = nn.LayerNorm(N_EMBD) # Final LayerNorm after blocks
        self.final_linear_layer = nn.Linear(N_EMBD, VOCAB_SIZE) # Maps to vocabulary logits

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initializes weights for linear and embedding layers."""
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
            input_tokens: Tensor of token indices of shape (B, T)
            targets: Optional tensor of target token indices of shape (B, T)

        Returns:
            Tuple of (logits, loss). Logits shape (B, T, vocab_size).
            Loss is optional cross-entropy loss if targets are provided.
        """
        B, T = input_tokens.shape

        # Get token and position embeddings
        token_embedding = self.token_embedding_table(input_tokens)  # (B, T, C=N_EMBD)
        pos_indices = torch.arange(T, device=DEVICE) # (T)
        positional_embedding = self.position_embedding_table(pos_indices) # (T, C=N_EMBD)

        # Add embeddings (broadcasting position embeddings across batch)
        x = token_embedding + positional_embedding  # (B, T, C)
        x = self.blocks(x)                          # (B, T, C)
        x = self.final_layer_norm(x)                # (B, T, C)
        logits = self.final_linear_layer(x)         # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Reshape for cross_entropy: expects (N, C) and (N)
            B_logits, T_logits, C_logits = logits.shape
            logits_for_loss = logits.view(B_logits * T_logits, C_logits)
            targets_for_loss = targets.view(B_logits * T_logits)
            loss = F.cross_entropy(logits_for_loss, targets_for_loss)

        return logits, loss

    def generate(self, input_tokens: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Generate new tokens given a context.

        Args:
            input_tokens: Starting token indices of shape (B, T_context)
            max_new_tokens: Number of new tokens to generate.

        Returns:
            Tensor of token indices of shape (B, T_context + max_new_tokens)
        """
        # input_tokens is (B, T) array of indices in the current context
        generated_tokens = input_tokens
        for _ in range(max_new_tokens):
            # Crop context if it exceeds block size
            context = generated_tokens[:, -BLOCK_SIZE:]
            # Get predictions (logits) for the next token
            logits, _ = self(context) # Pass context through the model
            # Focus only on the logits for the last time step
            logits_last_step = logits[:, -1, :] # (B, vocab_size)
            # Apply softmax to convert logits to probabilities
            probs = F.softmax(logits_last_step, dim=-1) # (B, vocab_size)
            # Sample the next token index from the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append the sampled index to the running sequence
            generated_tokens = torch.cat((generated_tokens, idx_next), dim=1) # (B, T+1)
        return generated_tokens
