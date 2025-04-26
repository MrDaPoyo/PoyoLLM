import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from feed_forward import SimplifiedLlama4FFN
from attention_code import SimpleL2Norm, SimplifiedLlama4Attention
import os

# Configuration (must match training config)
hidden_size = 128
ffn_intermediate_ratio = 8 / 3
multiple_of = 32
intermediate_size = int(hidden_size * ffn_intermediate_ratio)
intermediate_size = ((intermediate_size + multiple_of - 1) // multiple_of) * multiple_of
hidden_act = "silu"
rms_norm_eps = 1e-5
ffn_bias = False
sequence_length = 100
num_heads = 4

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int = 4):
        super().__init__()
        rope_theta = 10000.0
        attention_bias = False
        use_qk_norm = False

        self.attention = SimplifiedLlama4Attention(config = {
            "hidden_size": hidden_size,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_heads,
            "head_dim": hidden_size // num_heads,
            "max_position_embeddings": sequence_length,
            "rope_theta": rope_theta,
            "attention_bias": attention_bias,
            "use_qk_norm": use_qk_norm,
        })
        self.feed_forward = SimplifiedLlama4FFN(config = {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "hidden_act": hidden_act,
            "ffn_bias": ffn_bias,
            "rms_norm_eps": rms_norm_eps,
            "intermediate_size": intermediate_size,
        })
        self.attention_norm = SimpleL2Norm(eps=rms_norm_eps)
        self.ffn_norm = SimpleL2Norm(eps=rms_norm_eps)

    def forward(self, x, attention_mask, position_ids, freqs_cis) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), attention_mask=attention_mask, position_ids=position_ids, freqs_cis=freqs_cis)[0]
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class SimpleLlama(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, intermediate_size: int, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.tok_embeddings = nn.Embedding(vocab_size, hidden_size)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(TransformerBlock(hidden_size=hidden_size, intermediate_size=intermediate_size, num_heads=num_heads))

        self.norm = SimpleL2Norm(eps=rms_norm_eps)
        self.output = nn.Linear(hidden_size, vocab_size, bias=False)

        self.freqs_cis = self._precompute_freqs_cis(sequence_length, self.hidden_size // 4)

    def _precompute_freqs_cis(self, seq_len: int, n_elem: int, base: int = 10000) -> torch.Tensor:
        freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
        t = torch.arange(seq_len, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = tokens.shape
        h = self.tok_embeddings(tokens)

        freqs_cis = self.freqs_cis[:seq_len].to(h.device)
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=h.device).unsqueeze(0)
        attention_mask = None

        for layer in self.layers:
            h = layer(h, attention_mask=attention_mask, position_ids=position_ids, freqs_cis=freqs_cis)

        h = self.norm(h)
        output = self.output(h)
        return output

# Tokenizer setup (must match training)
cl100k_base = tiktoken.get_encoding("cl100k_base")
enc = tiktoken.Encoding(
    name="cl100k_poyo",
    pat_str=cl100k_base._pat_str,
    mergeable_ranks=cl100k_base._mergeable_ranks,
    special_tokens={
        "<|endoftext|>": 50256,
        "<|pad|>": 50257,
        "<|sep|>": 50258,
        "<|system|>": 50259,
        "<|user|>": 50260,
        "<|assistant|>": 50261,
    },
)
vocab_size = enc.n_vocab

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleLlama(vocab_size, hidden_size, intermediate_size).to(device)

# Load the model checkpoint
model_path = "output/model_epoch_25.pth"  # Replace with your actual path
model.load_state_dict(torch.load(model_path))
model.eval()

# Generate text
prompt = "<|system|>You are a helpful AI assistant.<|user|>Hello<|assistant|>"
encoded_prompt = enc.encode(prompt, allowed_special="all")
context = torch.tensor(encoded_prompt, dtype=torch.long, device=device).unsqueeze(0)

generated_tokens = []
with torch.no_grad():
    for _ in range(50):
        logits = model(context)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens.append(next_token.item())
        context = torch.cat([context, next_token], dim=1)

generated_text = enc.decode(generated_tokens)
print(f"Generated Text: {generated_text}")