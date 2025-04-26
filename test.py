import inspect
from dataclasses import dataclass
import torch
from torch.nn import functional as F
import tiktoken
import numpy as np
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.nn as nn
import torch.distributed as dist

# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 256 # max sequence length
    vocab_size: int = 50261 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        _, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def generate_response(self, tokenizer, prompt, max_new_tokens=50, temperature=0.7, top_k=50):
        """Generates a response using the fine-tuned PoyoSLM model with sampling."""
        self.eval()
        sot_token_id = tokenizer.encode("<|sot|>", allowed_special="all")[0]
        eot_token_id = tokenizer.encode("<|eot|>", allowed_special="all")[0]
        eod_token_id = tokenizer.encode("<|eod|>", allowed_special="all")[0]

        prompt_text = f"<|sot|>user: {prompt}<|eot|><|sot|>assistant:"
        input_ids = tokenizer.encode(prompt_text, allowed_special="all")
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        generated_ids = input_ids[:]
        max_seq_len = model.config.block_size # Get max length from model config

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Ensure input sequence doesn't exceed model's max length
                # Truncate from the left if it does
                current_input_tensor = input_tensor[:, -max_seq_len:]

                # Forward pass
                logits, _ = self(current_input_tensor) # Get logits, ignore loss
                # Get logits for the very last token prediction
                next_token_logits = logits[:, -1, :] # Shape: (batch_size=1, vocab_size)

                # Apply temperature scaling
                if temperature > 0 and temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Apply top-k filtering
                if top_k > 0:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')

                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()

                generated_ids.append(next_token_id)
                input_tensor = torch.cat((input_tensor, torch.tensor([[next_token_id]], device=device)), dim=1)

                if next_token_id in [eot_token_id, eod_token_id]:
                    break

        full_response_text = tokenizer.decode(generated_ids)

        # Extract only the assistant's part
        assistant_marker = "<|sot|>assistant:"
        marker_pos = full_response_text.rfind(assistant_marker)
        if marker_pos != -1:
            assistant_response = full_response_text[marker_pos + len(assistant_marker):]
        else:
            assistant_response = full_response_text # Fallback

        # Clean up trailing special tokens
        assistant_response = assistant_response.replace("<|eot|>", "").replace("<|eod|>", "").strip()

        return assistant_response

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from a local checkpoint"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints

        # create a from-scratch initialized model
        config = GPTConfig(**config_args)
        model = GPT(config)

        # To use this function without transformers, you need to:
        # 1. Download the weights manually from Hugging Face or OpenAI
        # 2. Place them in a known location
        # 3. Load them here
        checkpoint_path = f"./checkpoints/{model_type}.pt"  # Adjust path as needed

        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
            print(f"Successfully loaded weights from {checkpoint_path}")
        except FileNotFoundError:
            print(f"Could not find weights at {checkpoint_path}")
            print("Please download the weights manually and place them in the checkpoints directory")

        return model


    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

import os

# Main execution
if __name__ == "__main__":
    # Initialize a model
    print("Initializing a GPT model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Check for existing checkpoints in the "log" directory
    highest_step = -1
    highest_checkpoint = None

    if os.path.exists("log"):
        # Look for model checkpoint files
        for filename in os.listdir("log"):
            if filename.startswith("model_") and filename.endswith(".pt"):
                try:
                    # Extract step number from filename
                    step = int(filename[6:-3])  # Remove "model_" prefix and ".pt" suffix
                    if step > highest_step:
                        highest_step = step
                        highest_checkpoint = os.path.join("log", filename)
                except ValueError:
                    # Skip files that don't follow the expected naming pattern
                    continue

    # Load the checkpoint with the highest step number if found
    if highest_checkpoint is not None and os.path.exists(highest_checkpoint):
        print(f"Loading model from checkpoint: log/poyoSLM_finetuned.pt")

        # Add GPTConfig to the allowed globals
        torch.serialization.add_safe_globals([GPTConfig])

        # Load the checkpoint
        checkpoint = torch.load("log/poyoSLM_finetuned.pt", map_location=device, weights_only=False)

        # Extract model state_dict, optimizer state_dict, and any other needed states from the checkpoint
        model_state_dict = checkpoint['model']
        optimizer_state_dict = checkpoint.get('optimizer_state_dict', None)
        step = checkpoint.get('step', 0)  # Get the step number from the checkpoint (default is 0)

        # Create the model and load the state_dict
        model = GPT(GPTConfig())
        model.load_state_dict(model_state_dict)
        model = model.to(device)

        # Create the optimizer and load its state_dict if available
        optimizer = model.configure_optimizers(weight_decay=0.01, learning_rate=3e-4, device_type=device)
        if optimizer_state_dict:
            optimizer.load_state_dict(optimizer_state_dict)

        print(f"Resuming training from step {step}")
    else:
        # No checkpoint found, initialize a new model
        model = GPT(GPTConfig())
        model = model.to(device)
        optimizer = model.configure_optimizers(weight_decay=0.01, learning_rate=3e-4, device_type=device)
        print("No checkpoint found. Starting from scratch.")

    # Try to generate some text
    prompt = "Hello"
    print(f"User: {prompt}")

    PAD_TOKEN_ID = 50257 # Often used for padding
    EOT_TOKEN_ID = 50258 # End of Turn
    SOT_TOKEN_ID = 50259 # Start of Turn
    EOD_TOKEN_ID = 50260 # End of Document/Dialogue

    # Tokenize the prompt
    base_enc = tiktoken.get_encoding("gpt2")
    special_tokens = {
        "<|pad|>": PAD_TOKEN_ID,
        "<|eot|>": EOT_TOKEN_ID,
        "<|sot|>": SOT_TOKEN_ID,
        "<|eod|>": EOD_TOKEN_ID,
    }
    enc = tiktoken.Encoding(
        name="gpt2_with_special_tokens",
        pat_str=base_enc._pat_str,
        mergeable_ranks=base_enc._mergeable_ranks,
        special_tokens={**base_enc._special_tokens, **special_tokens}
    )
    tokens = torch.tensor([enc.encode(prompt)], dtype=torch.long).to(device)

    generated_text = model.generate_response(tokenizer=enc, prompt=prompt, max_new_tokens=100, temperature=0.8, top_k=50)

    print(f"Assistant: {generated_text}")

    print("Model parameters summary:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
