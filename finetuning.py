import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
import json
import os
import logging
import math
import time
import inspect
from dataclasses import dataclass

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
# Point this to your pre-trained nanoGPT checkpoint (e.g., from Karpathy's training script)
# Ensure this checkpoint's architecture matches the GPTConfig below.
MODEL_CHECKPOINT = "log/model_02000.pt" # Or your highest step checkpoint
DATA_FILE = "data/supervised_fine_tuning.json"
OUTPUT_MODEL_PATH = "log/poyoSLM_finetuned.pt" # Path to save the fine-tuned model

# Special Tokens (Ensure these IDs are outside the base tokenizer's range)
# Using values often reserved for custom tokens in GPT-2 style tokenizers
PAD_TOKEN_ID = 50257 # Often used for padding
EOT_TOKEN_ID = 50258 # End of Turn
SOT_TOKEN_ID = 50259 # Start of Turn
EOD_TOKEN_ID = 50260 # End of Document/Dialogue

# --- Model Configuration (MUST align with the checkpoint, adjust if needed) ---
@dataclass
class GPTConfig:
    block_size: int = 256    # Max sequence length (context size) - Adjusted from 1024 for fine-tuning example
    vocab_size: int = 50261  # Base vocab (50257) + 4 special tokens
    n_layer: int = 12        # Number of transformer layers
    n_head: int = 12         # Number of attention heads
    n_embd: int = 768        # Embedding dimension

# --- Hyperparameters (Adjust based on your resources and data) ---
LEARNING_RATE = 3e-5
BATCH_SIZE = 2          # Adjust based on GPU memory
EPOCHS = 100            # Number of fine-tuning epochs
MAX_LEN = GPTConfig.block_size # Use block_size from config

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_type = "cuda" if device.type == "cuda" else "cpu" # For nanoGPT compatibility if needed elsewhere
logging.info(f"Using device: {device}")

# --- nanoGPT Model Definition ---
# (Pasted directly from user prompt for clarity, could be in a separate model.py)
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Note: Flash Attention (`is_causal=True`) is used if available, otherwise standard SDPA
        # self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') # Check removed for brevity, assume available or PyTorch handles fallback

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Use scaled_dot_product_attention (efficient implementation)
        # is_causal=True handles the masking automatically for autoregressive models
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh') # Using approximate GELU like nanoGPT
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
                # Scale down initialization for residual projections
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Calculate loss internally if targets are provided
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100) # Use -100 ignore_index

        # Return logits AND loss if targets were provided
        # If targets=None, loss is None
        return logits, loss # Ensure the training loop handles this tuple output

    # configure_optimizers and from_pretrained methods removed for brevity in fine-tuning script
    # We will initialize optimizer directly and load state_dict from checkpoint

# --- Tokenizer ---
logging.info("Loading tokenizer...")
try:
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
    # Verify vocab size matches model expectation
    if enc.n_vocab != GPTConfig.vocab_size:
         logging.warning(f"Tokenizer vocab size ({enc.n_vocab}) doesn't match expected GPTConfig.vocab_size ({GPTConfig.vocab_size}). Ensure model config is correct.")
         # Potentially update GPTConfig.vocab_size here if dynamic adjustment is desired
         # GPTConfig.vocab_size = enc.n_vocab # Uncomment carefully

    logging.info(f"Tokenizer loaded. Vocabulary size: {enc.n_vocab}")

except Exception as e:
    logging.error(f"Failed to load tokenizer: {e}")
    exit(1)


# --- Dataset ---
class FineTuningDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Get actual IDs from the tokenizer instance
        self.pad_token_id = tokenizer.encode("<|pad|>", allowed_special="all")[0]
        self.eot_token_id = tokenizer.encode("<|eot|>", allowed_special="all")[0]
        self.sot_token_id = tokenizer.encode("<|sot|>", allowed_special="all")[0]
        self.eod_token_id = tokenizer.encode("<|eod|>", allowed_special="all")[0]

        logging.info(f"Loading data from {json_file}...")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            if not isinstance(self.data, list):
                raise ValueError("JSON data must be a list of conversations.")
        except FileNotFoundError:
            logging.error(f"Error: Data file not found at {json_file}")
            self.data = []
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Error loading or parsing JSON from {json_file}: {e}")
            self.data = []

        self.processed_data = self._preprocess_data()
        if not self.processed_data:
             logging.warning("Warning: No valid data loaded after preprocessing.")
        else:
             logging.info(f"Loaded and processed {len(self.processed_data)} examples.")

    def _preprocess_data(self):
        processed = []
        skipped_count = 0
        for item in self.data:
            if not isinstance(item, dict) or "messages" not in item or not isinstance(item["messages"], list):
                logging.warning(f"Skipping invalid item format: {item}")
                skipped_count += 1
                continue

            full_text_tokens = []
            valid_item = True
            for msg in item["messages"]:
                 if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                      logging.warning(f"Skipping invalid message format within item: {msg}")
                      valid_item = False
                      break
                 role = msg.get("role", "unknown")
                 content = msg.get("content", "")
                 turn_text = f"<|sot|>{role}: {content}<|eot|>"
                 try:
                    turn_tokens = self.tokenizer.encode(turn_text, allowed_special="all")
                    full_text_tokens.extend(turn_tokens)
                 except Exception as e:
                     logging.warning(f"Could not encode turn: {turn_text}. Error: {e}")
                     valid_item = False
                     break

            if not valid_item:
                skipped_count += 1
                continue

            full_text_tokens.append(self.eod_token_id)

            # Truncate if necessary (Input for model: max_len, Target: max_len)
            # The model input will be tokens[:-1], target tokens[1:]
            if len(full_text_tokens) > self.max_len:
                full_text_tokens = full_text_tokens[:self.max_len]

            # Pad if necessary
            if len(full_text_tokens) < self.max_len:
                padding = [self.pad_token_id] * (self.max_len - len(full_text_tokens))
                full_text_tokens.extend(padding)

            if len(full_text_tokens) == self.max_len:
                 processed.append(torch.tensor(full_text_tokens, dtype=torch.long))
            else:
                 logging.warning(f"Skipping example due to unexpected length ({len(full_text_tokens)}) after processing.")
                 skipped_count += 1

        if skipped_count > 0:
            logging.warning(f"Skipped {skipped_count} invalid or problematic items during preprocessing.")
        return processed

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        tokens = self.processed_data[idx]
        # Input is sequence, target is sequence shifted by one position
        inputs = tokens[:-1].clone()  # Input sequence (length max_len - 1)
        targets = tokens[1:].clone() # Target sequence (length max_len - 1)
        # Mask padding tokens in targets for loss calculation
        # We mask targets where the *corresponding input* was padding,
        # or where the target itself is padding.
        # Since input is tokens[:-1] and target is tokens[1:], if input[i] is pad, target[i] is also pad (or the next token if padding ended)
        # Masking based on target token being pad_token_id is sufficient here.
        targets[targets == self.pad_token_id] = -100 # PyTorch CrossEntropyLoss ignore_index
        return inputs, targets


# --- Model Loading ---
# Initialize model using the config defined above
model_config = GPTConfig() # Uses the parameters defined in the dataclass
model = GPT(model_config)
model.to(device) # Move model to device early

# Load checkpoint if it exists
if os.path.exists(MODEL_CHECKPOINT):
    logging.info(f"Loading model checkpoint from {MODEL_CHECKPOINT}...")
    try:
        # Load the checkpoint. Use map_location to load directly to the correct device.
        # Set weights_only=False as checkpoint likely contains config object.
        # torch.serialization.safe_globals([GPT, GPTConfig]) # May not be needed with newer torch versions or if config is simple dataclass
        checkpoint = torch.load(MODEL_CHECKPOINT, map_location=device, weights_only=False) # Set weights_only based on checkpoint content

        if 'model' not in checkpoint:
            raise KeyError("Checkpoint does not contain 'model' state_dict.")

        state_dict = checkpoint['model']
        loaded_config = checkpoint.get('config', None) # Get config if it exists

        if loaded_config:
            logging.info(f"Checkpoint config found: {loaded_config}")
            # Check for critical architecture mismatches
            if (hasattr(loaded_config, 'n_layer') and loaded_config.n_layer != model_config.n_layer or
                hasattr(loaded_config, 'n_head') and loaded_config.n_head != model_config.n_head or
                hasattr(loaded_config, 'n_embd') and loaded_config.n_embd != model_config.n_embd):
                 logging.warning(f"Critical mismatch detected between checkpoint config ({loaded_config}) and script config ({model_config}).")
                 logging.warning("Re-initializing model with checkpoint's core architecture config.")
                 # Update script config and re-initialize model
                 model_config.n_layer = loaded_config.n_layer if hasattr(loaded_config, 'n_layer') else model_config.n_layer
                 model_config.n_head = loaded_config.n_head if hasattr(loaded_config, 'n_head') else model_config.n_head
                 model_config.n_embd = loaded_config.n_embd if hasattr(loaded_config, 'n_embd') else model_config.n_embd
                 # Keep script's vocab/block size for now, handle resizing below
                 model = GPT(model_config).to(device) # Re-initialize model with potentially updated core dims
                 logging.info(f"Model re-initialized with config: {model_config}")

            # --- Handle vocab_size and block_size mismatches BEFORE load_state_dict ---
            # Checkpoint vocab vs Script vocab
            if 'transformer.wte.weight' in state_dict:
                ckpt_vocab_size = state_dict['transformer.wte.weight'].shape[0]
                if ckpt_vocab_size != model_config.vocab_size:
                    logging.warning(f"Vocab size mismatch: Checkpoint={ckpt_vocab_size}, Script={model_config.vocab_size}. Adapting weights.")
                    # Adjust wte weights
                    current_wte = model.transformer.wte.weight.data
                    ckpt_wte = state_dict['transformer.wte.weight']
                    min_vocab = min(ckpt_vocab_size, model_config.vocab_size)
                    current_wte[:min_vocab, :] = ckpt_wte[:min_vocab, :]
                    if model_config.vocab_size > ckpt_vocab_size:
                        logging.info(f"Initializing {model_config.vocab_size - ckpt_vocab_size} new token embeddings in wte.")
                        current_wte[ckpt_vocab_size:].normal_(mean=0.0, std=0.02) # Initialize new embeddings
                    model.transformer.wte.weight.data = current_wte
                    # Remove the original weight from state_dict to avoid size mismatch error during load
                    del state_dict['transformer.wte.weight']

                    # Adjust lm_head weights (assuming tied weights or separate but matching size)
                    if 'lm_head.weight' in state_dict: # Check if lm_head is separate or tied
                        # Verify if lm_head shape matches wte shape in checkpoint
                        if state_dict['lm_head.weight'].shape[0] == ckpt_vocab_size:
                            current_lm_head = model.lm_head.weight.data
                            ckpt_lm_head = state_dict['lm_head.weight']
                            min_vocab = min(ckpt_vocab_size, model_config.vocab_size)
                            current_lm_head[:min_vocab, :] = ckpt_lm_head[:min_vocab, :]
                            if model_config.vocab_size > ckpt_vocab_size:
                                logging.info(f"Initializing {model_config.vocab_size - ckpt_vocab_size} new weights in lm_head.")
                                current_lm_head[ckpt_vocab_size:].normal_(mean=0.0, std=0.02) # Initialize new weights
                            model.lm_head.weight.data = current_lm_head
                            # Remove the original weight from state_dict
                            del state_dict['lm_head.weight']
                            # Ensure weights remain tied if they should be (model init should handle this)
                            # model.transformer.wte.weight = model.lm_head.weight # Re-tying might be needed if init doesn't guarantee it after manual data setting
                        else:
                            logging.warning("lm_head.weight shape in checkpoint does not match wte.weight shape. Skipping lm_head adaptation.")
                            # Decide how to handle this - maybe remove lm_head from state_dict too?
                            if 'lm_head.weight' in state_dict: del state_dict['lm_head.weight']

            # Checkpoint block_size vs Script block_size
            if 'transformer.wpe.weight' in state_dict:
                ckpt_block_size = state_dict['transformer.wpe.weight'].shape[0]
                if ckpt_block_size != model_config.block_size:
                    logging.warning(f"Block size mismatch: Checkpoint={ckpt_block_size}, Script={model_config.block_size}. Adapting weights.")
                    current_wpe = model.transformer.wpe.weight.data
                    ckpt_wpe = state_dict['transformer.wpe.weight']
                    min_block = min(ckpt_block_size, model_config.block_size)
                    current_wpe[:min_block, :] = ckpt_wpe[:min_block, :]
                    if model_config.block_size > ckpt_block_size:
                        logging.info(f"Initializing {model_config.block_size - ckpt_block_size} new positional embeddings.")
                        current_wpe[ckpt_block_size:].normal_(mean=0.0, std=0.02) # Initialize new embeddings
                    model.transformer.wpe.weight.data = current_wpe
                    # Remove the original weight from state_dict
                    del state_dict['transformer.wpe.weight']
            else:
                 logging.warning("Checkpoint state_dict does not contain 'transformer.wpe.weight'. Skipping block size adaptation.")

        else: # No config in checkpoint
            logging.warning("Checkpoint does not contain 'config'. Assuming script config is compatible.")
            # Add direct shape checks from state_dict if config is missing and adaptation is critical
            # For simplicity here, we proceed assuming compatibility or rely on strict=False below.

        # Fix potential issues with state_dict keys (e.g., DDP prefix)
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        # Load the potentially modified state dictionary
        # Use strict=False because we might have manually handled/removed some keys
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        # Report any remaining issues after our manual adjustments
        if missing_keys:
             # Keys we expect to be missing if we manually handled them
             handled_missing = {'transformer.wte.weight', 'lm_head.weight', 'transformer.wpe.weight'}
             unhandled_missing = set(missing_keys) - handled_missing
             if unhandled_missing:
                 logging.warning(f"State_dict still missing keys after adaptation: {unhandled_missing}")
             # else: # Log less verbosely if keys were handled as expected
             #    logging.info(f"Missing keys handled during adaptation: {missing_keys}")

        if unexpected_keys:
             logging.warning(f"Unexpected keys in state_dict: {unexpected_keys}")

        # Re-tie weights explicitly after loading state_dict if necessary
        # This ensures they point to the same memory IF they were adapted separately.
        if model.transformer.wte.weight is not model.lm_head.weight:
             logging.info("Re-tying weights for wte and lm_head after loading.")
             model.transformer.wte.weight = model.lm_head.weight


        logging.info("Checkpoint loaded successfully with potential adaptations.")

        # Load optimizer state if available and needed (optional)
        # if 'optimizer' in checkpoint:
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        #     logging.info("Optimizer state loaded.")

    except FileNotFoundError:
         logging.error(f"Checkpoint file not found at {MODEL_CHECKPOINT}. Starting with initialized weights.")
         model = GPT(GPTConfig()).to(device) # Ensure model uses script config
    except Exception as e:
        logging.error(f"Error loading checkpoint: {e}. Check architecture and file integrity. Starting with initialized weights.")
        model = GPT(GPTConfig()).to(device) # Ensure model uses script config
else:
    logging.warning(f"Warning: Checkpoint file not found at {MODEL_CHECKPOINT}. Model will use initialized weights (check config).")
    model = GPT(GPTConfig()).to(device) # Ensure model uses script config


# --- Training Setup ---
logging.info("Setting up training...")
dataset = FineTuningDataset(DATA_FILE, enc, MAX_LEN)

if len(dataset) == 0:
    logging.error("Error: Dataset is empty. Cannot proceed with training.")
    exit(1)

if BATCH_SIZE > len(dataset):
    logging.warning(f"Batch size ({BATCH_SIZE}) is larger than dataset size ({len(dataset)}). Setting batch size to dataset size.")
    BATCH_SIZE = len(dataset) if len(dataset) > 0 else 1


dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
# Use CrossEntropyLoss with ignore_index set to -100 (matching the dataset preprocessing)
criterion = nn.CrossEntropyLoss(ignore_index=-100) # Still useful if not using model's internal loss
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# --- Fine-tuning Loop ---
logging.info(f"Starting fine-tuning for {EPOCHS} epoch(s)...")
model.train() # Set model to training mode
for epoch in range(EPOCHS):
    total_loss = 0
    batch_count = 0
    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        t0 = time.time()

        optimizer.zero_grad()

        # Forward pass - Use model's internal loss calculation
        # logits, loss = model(inputs, targets) # Use this if model calculates loss

        # OR: Calculate loss externally (as in original script)
        logits, _ = model(inputs) # Get logits only
        # Reshape for CrossEntropyLoss: (Batch * SeqLen, VocabSize) and (Batch * SeqLen)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

        # Backward pass and optimization
        loss.backward()
        # Optional: Gradient clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

        # Log progress
        t1 = time.time()
        dt = (t1 - t0) * 1000 # Time in ms
        tokens_per_sec = (BATCH_SIZE * inputs.size(1)) / (t1 - t0)

        if (i + 1) % 10 == 0: # Log every 10 batches
             logging.info(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}, Time: {dt:.2f}ms, Tok/sec: {tokens_per_sec:.2f}")

    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    logging.info(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

# --- Save Fine-tuned Model ---
try:
    logging.info(f"Saving fine-tuned model to {OUTPUT_MODEL_PATH}...")
    # Save in a format similar to nanoGPT for consistency
    save_obj = {
        'model': model.state_dict(),
        'config': model.config, # Save the actual config object used by the model
        'optimizer': optimizer.state_dict(), # Optional: save optimizer state
        'epoch': EPOCHS, # Optional: save training progress info
        # Add any other metadata if needed
        'fine_tuning_args': {
             'learning_rate': LEARNING_RATE,
             'batch_size': BATCH_SIZE,
             'data_file': DATA_FILE,
        }
    }
    torch.save(save_obj, OUTPUT_MODEL_PATH)
    logging.info("Model saved successfully.")
except Exception as e:
    logging.error(f"Failed to save model: {e}")


# --- Generation ---
logging.info("\n--- Generating Sample Conversation ---")
model.eval() # Set model to evaluation mode

def generate_response(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7, top_k=50):
    """Generates a response using the fine-tuned nanoGPT model with sampling."""
    model.eval()
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
            logits, _ = model(current_input_tensor) # Get logits, ignore loss
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


# --- Example Generation Usage ---
start_prompt = "What is the capital of France?"
logging.info(f"\nPrompt: {start_prompt}")
response = generate_response(model, enc, start_prompt, max_new_tokens=50, temperature=0.7, top_k=50)
logging.info(f"Generated Response: {response}")

start_prompt = "Explain the concept of supervised fine-tuning for LLMs."
logging.info(f"\nPrompt: {start_prompt}")
response = generate_response(model, enc, start_prompt, max_new_tokens=100, temperature=0.8, top_k=50)
logging.info(f"Generated Response: {response}")

logging.info("\nFine-tuning and generation script finished.")