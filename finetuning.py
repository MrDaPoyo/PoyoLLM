import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
from torch.nn import functional as F
import tiktoken
import numpy as np
# Removed: from transformers import GPT2LMHeadModel
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.nn as nn
# -----------------------------------------------------------------------------
# Basic GPT model definitions (unchanged from original)
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
        # causal mask moved to forward pass

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # Use PyTorch scaled_dot_product_attention for efficiency
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh') # Using approximate GELU
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
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
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
        B, T = idx.size()
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

    # Removed the from_pretrained method as it depended on the transformers library.
    # @classmethod
    # def from_pretrained(cls, model_type):
    #     ... (code removed) ...

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

# -----------------------------------------------------------------------------
# DataLoader for conversational data
# Assumes pre-tokenized .npy files for train and val splits
# You need to prepare your conversational data into this format.
# Example format: Concatenate turns with special tokens if needed,
# then tokenize and save as numpy arrays.
# e.g., "User: Hello<|endoftext|>Assistant: Hi there!<|endoftext|>User: How are you?<|endoftext|>..."
# -----------------------------------------------------------------------------

def load_tokens(filename):
    # Check if file exists before attempting to load
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Error: Token file not found at {filename}. "
                                f"Please ensure your conversational data is pre-tokenized and saved correctly.")
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, data_dir="data"):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # Use specific filenames for conversational data
        data_filename = f"conversation_{split}.npy"
        shard_path = os.path.join(data_dir, data_filename)

        # Store the single shard path
        self.shard_path = shard_path
        self.tokens = None # Load lazily in reset
        self.current_position = 0

        # Check for file existence during initialization
        if self.process_rank == 0:
            if not os.path.exists(self.shard_path):
                 print(f"Warning: Data file {self.shard_path} not found. DataLoader will fail if reset() is called.")
            else:
                print(f"Using data file: {self.shard_path}")

        self.reset() # Initial load

    def reset(self):
        # Load or reload the tokens from the single file
        try:
            self.tokens = load_tokens(self.shard_path)
        except FileNotFoundError as e:
            # Handle case where file might not exist when worker processes initialize
            print(f"Rank {self.process_rank}: Error loading tokens - {e}")
            # Set tokens to a dummy tensor to avoid crashing later, though training will be invalid
            self.tokens = torch.empty((0,), dtype=torch.long)

        if self.tokens.numel() == 0 and self.process_rank == 0:
             print(f"Warning: Loaded token file {self.shard_path} is empty.")

        # Reset position based on rank to ensure different data for each process
        # Ensure current_position calculation doesn't exceed token length immediately
        self.current_position = self.B * self.T * self.process_rank
        # Simple modulo arithmetic for wrap-around if initial position is too large
        # This might happen if the dataset is very small.
        if self.tokens.numel() > 0:
             self.current_position %= self.tokens.numel()
        else:
             self.current_position = 0 # Cannot set position if no tokens


    def next_batch(self):
        B, T = self.B, self.T
        # Handle potential empty token tensor
        if self.tokens.numel() == 0:
             print(f"Rank {self.process_rank}: Warning - Attempting to fetch batch from empty dataset.")
             # Return dummy tensors to avoid crashing downstream, though loss will be meaningless
             return torch.zeros((B, T), dtype=torch.long), torch.zeros((B, T), dtype=torch.long)

        # Calculate end index, ensuring it doesn't exceed buffer length
        end_idx = self.current_position + B * T + 1

        # Check if we need to wrap around the dataset
        if end_idx > self.tokens.numel():
            print(f"Rank {self.process_rank}: Wrapping around dataset.")
            # Get remaining tokens from the end
            remaining_tokens = self.tokens[self.current_position:]
            # Get needed tokens from the beginning
            needed_from_start = end_idx - self.tokens.numel()
            start_tokens = self.tokens[:needed_from_start]
            # Concatenate to form the buffer
            buf = torch.cat((remaining_tokens, start_tokens), dim=0)
            # Reset position to the start, adjusted for the next batch fetch by this process
            self.current_position = needed_from_start + (B * T * (self.num_processes - 1))
            # Ensure the new position is valid
            self.current_position %= self.tokens.numel()

        else:
            # Standard case: load buffer directly
            buf = self.tokens[self.current_position : end_idx]
            # Advance the position for the next batch fetch across all processes
            self.current_position += B * T * self.num_processes
            # Simple wrap-around if position exceeds length (can happen with stride)
            if self.current_position >= self.tokens.numel():
                self.current_position %= self.tokens.numel()


        # Ensure buffer has the expected size + 1 for x and y
        if buf.numel() != B * T + 1:
             # This might happen if the dataset is smaller than B*T+1 or due to wrap-around logic issues
             print(f"Rank {self.process_rank}: Warning - Buffer size mismatch. Expected {B*T+1}, got {buf.numel()}. "
                   f"Current position: {self.current_position}, Token count: {self.tokens.numel()}. "
                   f"Returning potentially incorrect batch.")
             # Pad or truncate if necessary, though this indicates a problem
             if buf.numel() < B * T + 1:
                 padding = torch.zeros(B * T + 1 - buf.numel(), dtype=torch.long)
                 buf = torch.cat((buf, padding), dim=0)
             else:
                 buf = buf[:B*T+1]


        x = buf[:-1].view(B, T) # inputs
        y = buf[1:].view(B, T) # targets

        return x, y

# -----------------------------------------------------------------------------
# Fine-tuning script setup
# -----------------------------------------------------------------------------
# DDP setup (unchanged)
import torch.distributed as dist

ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    assert torch.cuda.is_available(), "DDP requires CUDA"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {device}")
device_type = "cuda" if device.startswith("cuda") else "cpu"

# Seed setting (unchanged)
torch.manual_seed(1337 + ddp_rank) # Offset seed per rank
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337 + ddp_rank)

# Fine-tuning specific parameters
# --- IMPORTANT: Adjust these for your specific conversational dataset and task ---
B = 8  # Micro batch size (adjust based on GPU memory)
T = 512 # Sequence length (adjust based on GPU memory and context needs)
max_lr = 3e-5 # Fine-tuning often uses smaller learning rates than pre-training
min_lr = max_lr * 0.1
warmup_steps = 50   # Adjust based on total steps
max_steps = 1000    # Total fine-tuning steps (e.g., 1-3 epochs over your dataset)
weight_decay = 0.01
log_interval = 10       # Log training loss frequency
val_interval = 250      # Validation frequency
gen_interval = 500      # Generation frequency
checkpoint_interval = 500 # Checkpoint saving frequency
# --- End of parameters to adjust ---

total_batch_size = B * T * ddp_world_size # Total effective batch size considering DDP
grad_accum_steps = 1 # Gradient accumulation steps (increase if B needs to be smaller due to memory)
# Adjust grad_accum_steps if needed, e.g., to simulate a larger batch size:
# desired_total_batch_size = 524288 # Example: 0.5M tokens
# grad_accum_steps = desired_total_batch_size // (B * T * ddp_world_size)

if master_process:
    print(f"Fine-tuning parameters:")
    print(f"  Batch size (per GPU): B={B}, T={T}")
    print(f"  Total batch size (tokens): {total_batch_size * grad_accum_steps}")
    print(f"  Gradient Accumulation Steps: {grad_accum_steps}")
    print(f"  Max Learning Rate: {max_lr:.2e}")
    print(f"  Min Learning Rate: {min_lr:.2e}")
    print(f"  Warmup Steps: {warmup_steps}")
    print(f"  Max Steps: {max_steps}")
    print(f"  Weight Decay: {weight_decay}")

# Model loading logic: Resume from checkpoint or start from scratch
log_dir = "log" # Use a separate log directory for fine-tuning
os.makedirs(log_dir, exist_ok=True)
model = None
optimizer = None
step = 0 # Initialize step count

# Check for latest checkpoint in the fine-tuning log directory
latest_checkpoint_path = None
if master_process: # Only master process checks for checkpoints
    ckpt_files = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
    if ckpt_files:
        latest_ckpt_file = max(ckpt_files, key=lambda f: int(f.split('_')[1].split('.')[0]))
        latest_checkpoint_path = os.path.join(log_dir, latest_ckpt_file)

# Broadcast checkpoint path from master to other processes if using DDP
if ddp:
    object_list = [latest_checkpoint_path]
    dist.broadcast_object_list(object_list, src=0)
    latest_checkpoint_path = object_list[0]

if latest_checkpoint_path and os.path.exists(latest_checkpoint_path):
    if master_process:
        print(f"Resuming fine-tuning from checkpoint: {latest_checkpoint_path}")
    torch.serialization.safe_globals([GPTConfig])
    checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    if 'optimizer' in checkpoint:
        # Recreate optimizer first, then load state dict
        optimizer = model.configure_optimizers(weight_decay=weight_decay, learning_rate=max_lr, device_type=device_type)
        optimizer.load_state_dict(checkpoint['optimizer'])
    step = checkpoint.get('step', 0) + 1 # Start from the next step
    val_loss = checkpoint.get('val_loss', float('inf'))
    if master_process:
        print(f"Loaded model config: {config}")
        print(f"Resuming from step {step}, last validation loss: {val_loss:.4f}")
    

else:
    if master_process:
        print("No fine-tuning checkpoint found. Starting model from scratch with GPT-2 default parameters.")
    # Start from scratch using default GPT-2 parameters
    # (Removed GPT.from_pretrained('gpt2') call)
    config_args = dict(n_layer=12, n_head=12, n_embd=768) # Default gpt2 config
    config_args['vocab_size'] = 50257 # Standard GPT-2 vocab size
    config_args['block_size'] = 1024 # Standard GPT-2 block size
    config = GPTConfig(**config_args)
    model = GPT(config)
    if master_process:
        print(f"Initialized new model with config: {config}")
    step = 0 # Start from step 0

# Ensure model is on the correct device
model.to(device)

# Configure optimizer if not loaded from checkpoint
if optimizer is None:
    optimizer = model.configure_optimizers(weight_decay=weight_decay, learning_rate=max_lr, device_type=device_type)

# DDP wrapping
use_compile = False # torch.compile can sometimes interfere, keep False for stability
if use_compile and hasattr(torch, 'compile'):
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # Get the underlying model

# DataLoader setup - Point to your pre-tokenized conversational data directory
# Create a 'data' directory and place 'conversation_train.npy' and 'conversation_val.npy' inside it.
data_dir = "data" # Directory containing your tokenized data
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train", data_dir=data_dir)
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val", data_dir=data_dir)

# Tokenizer for generation
enc = tiktoken.get_encoding("gpt2")

# Learning rate schedule function (unchanged, uses fine-tuning parameters)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it >= max_steps: # Use >= for max_steps
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# Fine-tuning loop
log_file = os.path.join(log_dir, f"log.txt")
if master_process and step == 0: # Write header only if starting fresh
     with open(log_file, "w") as f:
         f.write("step,train_loss,val_loss,lr\n") # CSV header

model.train() # Start in training mode

for current_step in range(step, max_steps): # Loop from current step up to max_steps
    t0 = time.time()
    last_step = (current_step == max_steps - 1)

    # Validation phase
    if current_step % val_interval == 0 or last_step:
        model.eval()
        val_loader.reset() # Ensure validation starts from the beginning
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20 # Number of batches to average validation loss over
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                # Use float16 for evaluation if desired and available
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16 if device_type == 'cuda' else torch.float32):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps # Average loss
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            val_loss_val = val_loss_accum.item()
            print(f"Validation Step {current_step}: Loss {val_loss_val:.4f}")
            # Log validation loss (append mode)
            with open(log_file, "a") as f:
                 # Log requires train loss, use NaN or previous value if not available yet
                 # For simplicity, just log val loss here, combine later if needed
                 f.write(f"{current_step},NaN,{val_loss_val:.4f},NaN\n") # Placeholder for train loss/lr

            # Save checkpoint based on validation interval or if it's the last step
            if current_step > 0 and (current_step % checkpoint_interval == 0 or last_step):
                checkpoint_path = os.path.join(log_dir, f"model_{current_step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config.__dict__, # Save config as dict
                    'step': current_step,
                    'val_loss': val_loss_val,
                    'optimizer': optimizer.state_dict(),
                }
                print(f"Saving checkpoint to {checkpoint_path}")
                torch.save(checkpoint, checkpoint_path)
        # Switch back to train mode after validation
        model.train()


    # Generation phase (example conversational prompt)
    if ((current_step > 0 and current_step % gen_interval == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 64 # Max length of generated sequence
        # --- Example Conversational Prompt ---
        # Format your prompt similar to how your training data is formatted
        prompt_text = "User: Hello, how are you today?\nAssistant:"
        # --- End Example Prompt ---

        tokens = enc.encode(prompt_text)
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)

        print("-" * 80)
        print(f"Generating responses at step {current_step} with prompt:")
        print(f"'{prompt_text}'")
        print("-" * 80)

        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                # Use float16 for generation if desired
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16 if device_type == 'cuda' else torch.float32):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)

        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            try:
                decoded = enc.decode(tokens)
                # Simple post-processing: remove prompt, stop at potential end markers if needed
                # decoded_response = decoded[len(prompt_text):]
                # Find end-of-text or other stop sequences if used in training
                # stop_token = "<|endoftext|>" # Example stop token
                # stop_idx = decoded_response.find(stop_token)
                # if stop_idx != -1:
                #     decoded_response = decoded_response[:stop_idx]
                print(f"Rank {ddp_rank} Sample {i}: {decoded}")
            except Exception as e:
                print(f"Rank {ddp_rank} Sample {i}: Error decoding tokens: {e}")
        print("-" * 80)
        # Switch back to train mode after generation
        model.train()


    # Training step (with gradient accumulation)
    optimizer.zero_grad()
    loss_accum = 0.0 # Accumulate loss over micro-steps
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # Set requires_sync for DDP on the last micro-step
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        # Use bfloat16 for training if available and on CUDA
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16 if device_type == 'cuda' else torch.float32):
            logits, loss = model(x, y)

        # Important: scale loss for gradient accumulation
        loss = loss / grad_accum_steps
        loss_accum += loss.detach() # Accumulate detached loss for logging
        loss.backward() # Accumulate gradients

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) # Average accumulated loss across ranks

    # Clip gradients
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Update learning rate
    lr = get_lr(current_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Optimizer step
    optimizer.step()

    if device_type == "cuda":
        torch.cuda.synchronize() # Wait for GPU work to finish

    t1 = time.time()
    dt = t1 - t0 # Time per step

    # Log training progress
    if master_process and current_step % log_interval == 0:
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        print(f"Step {current_step:5d} | Train Loss: {loss_accum.item():.6f} | LR: {lr:.4e} | Grad Norm: {norm:.4f} | Time: {dt*1000:.2f}ms | Tok/sec: {tokens_per_sec:.2f}")
        # Append training loss and LR to log file (validation loss added during validation step)
        # This logging format might need adjustment if validation happens less frequently than logging.
        # Consider logging train loss separately or combining logs post-run.
        # with open(log_file, "a") as f:
        #     f.write(f"{current_step},{loss_accum.item():.6f},NaN,{lr:.4e}\n") # Log train loss, placeholder for val loss


# Cleanup DDP
if ddp:
    destroy_process_group()

print("Fine-tuning finished.")
