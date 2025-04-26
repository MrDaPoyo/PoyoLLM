import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
import os
import logging
import time
from dataclasses import dataclass, field
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datasets import load_dataset
from collections import defaultdict

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
MODEL_CHECKPOINT = "log/model_02000.pt"
# DATASET_NAME = "mlabonne/yappgpt-conversation" # Original dataset name
DATASET_NAME = "OpenAssistant/oasst2" # Using oasst2
OUTPUT_MODEL_PATH = "log/poyoSLM_finetuned_oasst2_v2.pt" # Adjusted output path

# --- Special Token IDs ---
# Base GPT2 has 50256 tokens. We add special tokens starting from there.
# Note: tiktoken might assign different IDs if tokens exist in base vocab,
# but we explicitly define them for clarity and consistency.
# Let's reserve a range, assuming gpt2 base vocab size is 50256.
PAD_TOKEN_ID = 50257 # Often PAD is last or a dedicated ID
EOT_TOKEN_ID = 50258 # End Of Turn
SOT_TOKEN_ID = 50259 # Start Of Turn
EOD_TOKEN_ID = 50260 # End Of Dialogue
SYSTEM_TOKEN_ID = 50261
USER_TOKEN_ID = 50262
ASSISTANT_TOKEN_ID = 50263

# Calculate expected vocab size
EXPECTED_VOCAB_SIZE = 50256 + 7 # Base + 7 special tokens

# --- Model Configuration ---
@dataclass
class GPTConfig:
    block_size: int = 1024
    # Update default vocab size to include new special tokens
    vocab_size: int = EXPECTED_VOCAB_SIZE
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

# --- Hyperparameters ---
LEARNING_RATE = int(3e-5)
BATCH_SIZE = 2 # Keep small for memory
EPOCHS = 1 # Reduce epochs for faster testing/demonstration
MAX_LEN = GPTConfig.block_size # Use block_size from config

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_type = "cuda" if device.type == "cuda" else "cpu"
logging.info(f"Using device: {device}")

# --- nanoGPT Model Definition ---
# [Model Definition remains the same - CausalSelfAttention, MLP, Block, GPT]
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Use PyTorch's optimized attention
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh') # Use approximate='tanh' for potentially better performance
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

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # Adjust std dev for residual projections according to GPT-2 paper
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # Forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # Forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # Forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Calculate the loss, ignoring padding index -100
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)

        return logits, loss

# --- Tokenizer ---
logging.info("Loading tokenizer...")
try:
    base_enc = tiktoken.get_encoding("gpt2")
    # Define all special tokens including the new ones
    special_tokens = {
        "<|pad|>": PAD_TOKEN_ID,
        "<|eot|>": EOT_TOKEN_ID,
        "<|sot|>": SOT_TOKEN_ID,
        "<|eod|>": EOD_TOKEN_ID,
        "<|system|>": SYSTEM_TOKEN_ID,
        "<|user|>": USER_TOKEN_ID,
        "<|assistant|>": ASSISTANT_TOKEN_ID,
    }
    enc = tiktoken.Encoding(
        name="gpt2_with_custom_tokens",
        pat_str=base_enc._pat_str,
        mergeable_ranks=base_enc._mergeable_ranks,
        special_tokens={**base_enc._special_tokens, **special_tokens}
    )
    # Verify actual vocab size after adding tokens
    actual_vocab_size = enc.n_vocab
    logging.info(f"Tokenizer loaded. Base vocab: {base_enc.n_vocab}, Target special tokens: {len(special_tokens)}, Actual tokenizer vocab size: {actual_vocab_size}")

    # Update GPTConfig vocab size based on the actual tokenizer size
    # This is crucial if tiktoken merges some special tokens or handles them differently.
    if actual_vocab_size != GPTConfig.vocab_size:
         logging.warning(f"Tokenizer actual vocab size ({actual_vocab_size}) differs from expected ({GPTConfig.vocab_size}). Updating config.")
         GPTConfig.vocab_size = actual_vocab_size # Update config to match tokenizer

except Exception as e:
    logging.error(f"Failed to load tokenizer: {e}")
    exit(1)


# --- Dataset ---
class FineTuningDataset(Dataset):
    def __init__(self, dataset_name, tokenizer, max_len, split="train", lang="en"):
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Get actual token IDs from the loaded tokenizer
        self.pad_token_id = tokenizer.encode("<|pad|>", allowed_special="all")[0]
        self.eot_token_id = tokenizer.encode("<|eot|>", allowed_special="all")[0]
        self.sot_token_id = tokenizer.encode("<|sot|>", allowed_special="all")[0]
        self.eod_token_id = tokenizer.encode("<|eod|>", allowed_special="all")[0]
        self.system_token_id = tokenizer.encode("<|system|>", allowed_special="all")[0]
        self.user_token_id = tokenizer.encode("<|user|>", allowed_special="all")[0]
        self.assistant_token_id = tokenizer.encode("<|assistant|>", allowed_special="all")[0]

        logging.info(f"Token IDs: PAD={self.pad_token_id}, SOT={self.sot_token_id}, EOT={self.eot_token_id}, EOD={self.eod_token_id}, SYSTEM={self.system_token_id}, USER={self.user_token_id}, ASSISTANT={self.assistant_token_id}")


        logging.info(f"Loading dataset {dataset_name} split {split}...")
        try:
            full_dataset = load_dataset(dataset_name, split=split)
            if lang:
                self.dataset = full_dataset.filter(lambda x: x["lang"] == lang)
                logging.info(f"Filtered dataset for language '{lang}'. Size: {len(self.dataset)}")
            else:
                self.dataset = full_dataset
                logging.info(f"Loaded full dataset. Size: {len(self.dataset)}")

        except Exception as e:
            logging.error(f"Failed to load or filter dataset: {e}")
            self.dataset = []
            self.processed_data = []
            return

        self.processed_data = self._preprocess_data()
        if not self.processed_data:
             logging.warning("Warning: No valid data loaded after preprocessing.")
        else:
             logging.info(f"Processed {len(self.processed_data)} conversation trees into sequences.")

    def _reconstruct_conversations(self):
        """Reconstructs conversation trees from the flat OASST dataset structure."""
        trees = defaultdict(list)
        for msg in self.dataset:
            trees[msg['message_tree_id']].append(msg)

        reconstructed_conversations = []
        skipped_trees = 0
        for tree_id, messages in trees.items():
            try:
                msg_map = {msg['message_id']: msg for msg in messages}
                parent_map = defaultdict(list)
                root = None
                for msg in messages:
                    parent_id = msg['parent_id']
                    if parent_id:
                        parent_map[parent_id].append(msg)
                    else:
                        if root is not None:
                            logging.warning(f"Tree {tree_id} has multiple roots? Skipping.")
                            raise ValueError("Multiple roots found")
                        root = msg

                if root is None:
                    logging.warning(f"Tree {tree_id} has no root message? Skipping.")
                    raise ValueError("No root found")

                conversation = []
                stack = [root]
                visited_order = []

                current_msg = root
                while current_msg:
                    visited_order.append(current_msg)
                    children = parent_map.get(current_msg['message_id'], [])
                    if children:
                        children.sort(key=lambda x: x.get('rank', 0) or 0, reverse=True)
                        current_msg = children[0]
                    else:
                        current_msg = None

                formatted_messages = []
                for msg in visited_order:
                     # Map OASST roles ('prompter', 'assistant') to our new tokens
                     # Handle potential 'system' role if dataset includes it
                     role = msg['role']
                     role_token_id = None
                     if role == 'prompter':
                         role_token_id = self.user_token_id
                     elif role == 'assistant':
                         role_token_id = self.assistant_token_id
                     elif role == 'system': # Handle system role if present
                         role_token_id = self.system_token_id
                     else:
                         logging.warning(f"Unknown role '{role}' in tree {tree_id}, message {msg['message_id']}. Skipping message.")
                         continue

                     content = msg['text']
                     if not isinstance(content, str) or not content.strip():
                         logging.warning(f"Empty or invalid content in tree {tree_id}, message {msg['message_id']}. Skipping message.")
                         continue

                     # Store role token ID and content
                     formatted_messages.append({"role_token_id": role_token_id, "content": content})

                if formatted_messages:
                    reconstructed_conversations.append({"messages": formatted_messages})
                else:
                    logging.warning(f"Tree {tree_id} resulted in no valid formatted messages after reconstruction. Skipping.")
                    skipped_trees += 1

            except Exception as e:
                logging.warning(f"Failed to reconstruct tree {tree_id}: {e}. Skipping.")
                skipped_trees += 1

        logging.info(f"Reconstructed {len(reconstructed_conversations)} conversations. Skipped {skipped_trees} trees due to errors or filtering.")
        return reconstructed_conversations


    def _preprocess_data(self):
        """Tokenizes reconstructed conversations using special role tokens."""
        reconstructed_conversations = self._reconstruct_conversations()
        processed = []
        skipped_count = 0

        for conversation in reconstructed_conversations:
            try:
                full_text_tokens = []
                for msg in conversation["messages"]:
                    role_token_id = msg["role_token_id"]
                    content = msg["content"]

                    # Encode content first
                    content_tokens = self.tokenizer.encode(content, allowed_special=set()) # Disallow special tokens within content

                    # Format turn with special tokens: <|sot|> <|role|> content <|eot|>
                    turn_tokens = [self.sot_token_id, role_token_id] + content_tokens + [self.eot_token_id]
                    full_text_tokens.extend(turn_tokens)

                if full_text_tokens:
                    full_text_tokens.append(self.eod_token_id) # Add end-of-dialogue token

                    # Truncate if exceeds max_len
                    if len(full_text_tokens) > self.max_len:
                        full_text_tokens = full_text_tokens[-self.max_len:]
                        # Simple truncation: Ensure first token isn't part of a multi-token special sequence if needed
                        # (More robust handling might be needed depending on tokenizer behavior)

                    # Pad if shorter than max_len
                    if len(full_text_tokens) < self.max_len:
                        padding = [self.pad_token_id] * (self.max_len - len(full_text_tokens))
                        full_text_tokens.extend(padding)

                    # Create input and target tensors
                    input_ids = torch.tensor(full_text_tokens, dtype=torch.long)
                    targets = input_ids.clone()
                    # Shift targets: target at index i is prediction for input at index i+1
                    targets[:-1] = input_ids[1:]
                    # Set target for the last token and all padding tokens to -100 (ignore index)
                    targets[-1] = -100
                    targets[input_ids == self.pad_token_id] = -100

                    # Additionally, ignore loss for the tokens immediately preceding padding,
                    # as they don't predict a meaningful next token.
                    pad_indices = (input_ids == self.pad_token_id).nonzero(as_tuple=True)[0]
                    if len(pad_indices) > 0:
                        first_pad_index = pad_indices[0].item()
                        if first_pad_index > 0:
                            targets[first_pad_index - 1] = -100

                    processed.append((input_ids, targets))

                else:
                    logging.warning("Skipping conversation with no valid messages after formatting.")
                    skipped_count += 1

            except Exception as e:
                # Log the specific conversation that failed if possible (might need index or ID)
                logging.error(f"An error occurred during tokenization/processing of a reconstructed conversation: {e}", exc_info=True)
                skipped_count += 1

        logging.info(f"Skipped {skipped_count} reconstructed conversations during final processing/tokenization.")
        return processed

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]


# --- Model Loading ---
# Initialize model using the config defined above (which now has updated vocab size)
model_config = GPTConfig()
logging.info(f"Initializing model with config: {model_config}")
# Ensure config vocab size matches the final tokenizer vocab size before creating model
if model_config.vocab_size != enc.n_vocab:
     logging.warning(f"Model config vocab size ({model_config.vocab_size}) differs from final tokenizer vocab size ({enc.n_vocab}). Adjusting model config.")
     model_config.vocab_size = enc.n_vocab

model = GPT(model_config)
model.to(device) # Move model to device early

# Load checkpoint if it exists
if os.path.exists(MODEL_CHECKPOINT):
    logging.info(f"Loading model checkpoint from {MODEL_CHECKPOINT}...")
    try:
        checkpoint = torch.load(MODEL_CHECKPOINT, map_location=device, weights_only=False) # Set weights_only=False if loading optimizer etc.

        if 'model' not in checkpoint:
            raise KeyError("Checkpoint does not contain 'model' state_dict.")

        state_dict = checkpoint['model']
        loaded_config_dict = checkpoint.get('config', None)

        # --- Config Mismatch Handling ---
        if loaded_config_dict:
            # Convert dict to GPTConfig object if necessary
            loaded_config = GPTConfig(**loaded_config_dict) if isinstance(loaded_config_dict, dict) else loaded_config_dict
            logging.info(f"Checkpoint config found: {loaded_config}")

            # Critical architecture mismatch check (n_layer, n_head, n_embd)
            if (loaded_config.n_layer != model_config.n_layer or
                loaded_config.n_head != model_config.n_head or
                loaded_config.n_embd != model_config.n_embd):
                 logging.warning(f"Critical mismatch detected between checkpoint config ({loaded_config}) and script config ({model_config}).")
                 logging.warning("Re-initializing model with checkpoint's core architecture config.")
                 model_config.n_layer = loaded_config.n_layer
                 model_config.n_head = loaded_config.n_head
                 model_config.n_embd = loaded_config.n_embd
                 # Keep script's potentially updated vocab/block size for now, handle resizing below
                 model = GPT(model_config).to(device) # Re-initialize model
                 logging.info(f"Model re-initialized with core config: {model_config}")
            else:
                 logging.info("Checkpoint core architecture matches script config.")

            # Compare vocab_size and block_size between loaded config and current script config (which reflects tokenizer)
            # If they differ, prioritize the current script config (tokenizer) and resize weights.
            if loaded_config.vocab_size != model_config.vocab_size:
                logging.warning(f"Vocab size mismatch: Checkpoint={loaded_config.vocab_size}, Current Script/Tokenizer={model_config.vocab_size}. Model will use current size ({model_config.vocab_size}). Weights will be adapted.")
                # No need to re-init model here, adaptation happens during weight loading.
            else:
                 logging.info("Checkpoint vocab size matches current script/tokenizer.")


            if loaded_config.block_size != model_config.block_size:
                logging.warning(f"Block size mismatch: Checkpoint={loaded_config.block_size}, Current Script={model_config.block_size}. Model will use current size ({model_config.block_size}). Weights will be adapted.")
                # No need to re-init model here, adaptation happens during weight loading.
            else:
                 logging.info("Checkpoint block size matches current script.")

            # Ensure the model instance reflects the final decision on config (especially after potential re-init)
            model.config = model_config


        else: # No config in checkpoint
            logging.warning("Checkpoint does not contain 'config'. Attempting load with script config.")
            # Ensure the model's config vocab matches the tokenizer before loading weights
            if model.config.vocab_size != enc.n_vocab:
                 logging.warning(f"Model config vocab ({model.config.vocab_size}) doesn't match tokenizer ({enc.n_vocab}) and no config in checkpoint. Adjusting model config.")
                 model.config.vocab_size = enc.n_vocab
                 # Re-initialize model with correct vocab size before loading weights
                 model = GPT(model.config).to(device)


        # --- Weight Adaptation ---
        # Adapt weights based on the *model's current config* vs the checkpoint's state_dict shapes
        # Vocab size adaptation (wte and lm_head)
        if 'transformer.wte.weight' in state_dict:
            ckpt_vocab_size = state_dict['transformer.wte.weight'].shape[0]
            model_vocab_size = model.config.vocab_size
            if ckpt_vocab_size != model_vocab_size:
                logging.warning(f"Adapting WTE weights: Checkpoint vocab={ckpt_vocab_size}, Model vocab={model_vocab_size}")
                current_wte = model.transformer.wte.weight.data.clone() # Get target shape
                ckpt_wte = state_dict['transformer.wte.weight']
                min_vocab = min(ckpt_vocab_size, model_vocab_size)
                # Copy overlapping weights
                current_wte[:min_vocab, :] = ckpt_wte[:min_vocab, :]
                # Initialize new weights if model vocab is larger
                if model_vocab_size > ckpt_vocab_size:
                    current_wte[ckpt_vocab_size:].normal_(mean=0.0, std=0.02)
                # Initialize weights if model vocab is smaller (less common, usually means error)
                elif model_vocab_size < ckpt_vocab_size:
                     logging.warning("Model vocab size is smaller than checkpoint. Truncating weights.")
                     current_wte = current_wte[:model_vocab_size, :] # Truncate

                model.transformer.wte.weight.data = current_wte
                del state_dict['transformer.wte.weight'] # Remove adapted key

                # Adapt lm_head (tied weights) - ensure it matches the adapted wte
                if 'lm_head.weight' in state_dict:
                    # Check if lm_head shape in checkpoint matches its wte shape
                    if state_dict['lm_head.weight'].shape[0] == ckpt_vocab_size:
                         # We already adapted wte, just ensure lm_head points to it after loading
                         # Remove the key from state_dict to avoid loading it separately
                         del state_dict['lm_head.weight']
                         logging.info("LM head weight will be tied to adapted WTE weight.")
                    else:
                         logging.warning("LM head shape in checkpoint doesn't match WTE shape. Removing from state_dict.")
                         del state_dict['lm_head.weight']
                # Ensure weights are tied *after* loading potentially untied weights
                model.lm_head.weight = model.transformer.wte.weight


        # Block size adaptation (wpe)
        if 'transformer.wpe.weight' in state_dict:
            ckpt_block_size = state_dict['transformer.wpe.weight'].shape[0]
            model_block_size = model.config.block_size
            if ckpt_block_size != model_block_size:
                logging.warning(f"Adapting WPE weights: Checkpoint block_size={ckpt_block_size}, Model block_size={model_block_size}")
                current_wpe = model.transformer.wpe.weight.data.clone() # Get target shape
                ckpt_wpe = state_dict['transformer.wpe.weight']
                min_block = min(ckpt_block_size, model_block_size)
                # Copy overlapping weights
                current_wpe[:min_block, :] = ckpt_wpe[:min_block, :]
                # Initialize new weights if model block size is larger
                if model_block_size > ckpt_block_size:
                    current_wpe[ckpt_block_size:].normal_(mean=0.0, std=0.02)
                # Truncate if model block size is smaller
                elif model_block_size < ckpt_block_size:
                     logging.warning("Model block size is smaller than checkpoint. Truncating weights.")
                     current_wpe = current_wpe[:model_block_size, :] # Truncate

                model.transformer.wpe.weight.data = current_wpe
                del state_dict['transformer.wpe.weight'] # Remove adapted key


        # Fix DDP prefix if present
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        # Load the potentially modified state dictionary
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        # Report unhandled mismatches
        handled_missing = {'transformer.wte.weight', 'lm_head.weight', 'transformer.wpe.weight'}
        unhandled_missing = set(missing_keys) - handled_missing
        if unhandled_missing:
             logging.warning(f"State_dict still missing keys after adaptation: {unhandled_missing}")
        if unexpected_keys:
             # Filter out optimizer keys if they were accidentally included in model state_dict
             unexpected_keys = [k for k in unexpected_keys if 'optimizer' not in k]
             if unexpected_keys:
                 logging.warning(f"Unexpected keys in model state_dict: {unexpected_keys}")

        # Explicitly re-tie weights after loading state_dict, as loading might untie them
        if model.transformer.wte.weight is not model.lm_head.weight:
             logging.info("Re-tying weights for wte and lm_head after loading.")
             model.lm_head.weight = model.transformer.wte.weight

        logging.info("Checkpoint loaded successfully with potential adaptations.")

    except FileNotFoundError:
         logging.error(f"Checkpoint file not found at {MODEL_CHECKPOINT}. Starting with initialized weights.")
         # Model already initialized with correct config above
    except Exception as e:
        logging.error(f"Error loading checkpoint: {e}. Check architecture and file integrity. Starting with initialized weights.", exc_info=True)
        # Model already initialized with correct config above
else:
    logging.warning(f"Warning: Checkpoint file not found at {MODEL_CHECKPOINT}. Model will use initialized weights.")
    # Model already initialized with correct config above


# --- Training Setup ---
logging.info("Setting up training...")
# Instantiate dataset using the new class structure
dataset = FineTuningDataset(DATASET_NAME, enc, MAX_LEN, split="train", lang="en")

if len(dataset) == 0:
    logging.error("Error: Dataset is empty after processing. Cannot proceed with training.")
    exit(1)

# Adjust batch size if it's larger than the dataset
effective_batch_size = min(BATCH_SIZE, len(dataset)) if len(dataset) > 0 else 1
if BATCH_SIZE > len(dataset) and len(dataset) > 0:
    logging.warning(f"Batch size ({BATCH_SIZE}) is larger than dataset size ({len(dataset)}). Setting batch size to {effective_batch_size}.")

# Ensure drop_last=False if effective_batch_size doesn't divide dataset size evenly
# Although with shuffle=True, the last batch size might vary anyway.
dataloader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True, drop_last=False)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# --- Fine-tuning Loop ---
logging.info(f"Starting fine-tuning for {EPOCHS} epoch(s)...")
model.train() # Set model to training mode
for epoch in range(EPOCHS):
    total_loss = 0
    batch_count = 0
    processed_samples = 0
    epoch_start_time = time.time()

    for i, (inputs, targets) in enumerate(dataloader):
        if inputs.nelement() == 0: # Skip empty batches if any occurred
             logging.warning(f"Skipping empty batch {i+1}/{len(dataloader)}")
             continue

        inputs, targets = inputs.to(device), targets.to(device)
        current_batch_size = inputs.size(0) # Actual batch size for this step
        processed_samples += current_batch_size

        t0 = time.time()

        optimizer.zero_grad()

        # Forward pass
        logits, loss = model(inputs, targets)

        # Backward pass and optimization
        if loss is not None and not torch.isnan(loss): # Check if loss was computed and is valid
            loss.backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * current_batch_size # Weight loss by batch size
            batch_count += 1

            # Log progress
            t1 = time.time()
            dt = (t1 - t0) * 1000 # Time in ms
            tokens_processed = current_batch_size * inputs.size(1)
            tokens_per_sec = tokens_processed / (t1 - t0) if (t1 - t0) > 0 else 0
            current_avg_loss = total_loss / processed_samples if processed_samples > 0 else 0


            if (i + 1) % 10 == 0: # Log every 10 batches
                 logging.info(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}, Avg Loss: {current_avg_loss:.4f}, Time: {dt:.2f}ms, Tok/sec: {tokens_per_sec:.2f}")
        elif loss is None:
            logging.warning(f"Skipping batch {i+1} due to None loss (potentially empty targets after processing?)")
        elif torch.isnan(loss):
             logging.error(f"NaN loss detected at batch {i+1}. Skipping batch.")
             # Consider stopping training or reducing LR if NaNs persist


    epoch_duration = time.time() - epoch_start_time
    avg_loss = total_loss / processed_samples if processed_samples > 0 else 0
    logging.info(f"Epoch {epoch+1} finished in {epoch_duration:.2f}s. Average Loss: {avg_loss:.4f}")

# --- Save Fine-tuned Model ---
try:
    logging.info(f"Saving fine-tuned model to {OUTPUT_MODEL_PATH}...")
    # Save the model's config directly
    config_to_save = model.config

    save_obj = {
        'model': model.state_dict(),
        'config': config_to_save, # Save the dataclass object directly
        'optimizer': optimizer.state_dict(),
        'epoch': EPOCHS,
        'fine_tuning_args': {
             'learning_rate': LEARNING_RATE,
             'batch_size': effective_batch_size,
             'dataset_name': DATASET_NAME,
        },
        'tokenizer_name': enc.name, # Store tokenizer info
        'special_tokens': special_tokens # Store special tokens map used
    }
    # Ensure log directory exists
    os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)
    torch.save(save_obj, OUTPUT_MODEL_PATH)
    logging.info("Model saved successfully.")
except Exception as e:
    logging.error(f"Failed to save model: {e}", exc_info=True)


# --- Generation ---
logging.info("\n--- Generating Sample Conversation ---")
model.eval() # Set model to evaluation mode

def generate_response(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7, top_k=50):
    """Generates a response using the fine-tuned model with new special tokens."""
    model.eval()
    # Get token IDs from the tokenizer instance
    sot_token_id = tokenizer.encode("<|sot|>", allowed_special="all")[0]
    eot_token_id = tokenizer.encode("<|eot|>", allowed_special="all")[0]
    eod_token_id = tokenizer.encode("<|eod|>", allowed_special="all")[0]
    user_token_id = tokenizer.encode("<|user|>", allowed_special="all")[0]
    assistant_token_id = tokenizer.encode("<|assistant|>", allowed_special="all")[0]

    # Format the prompt using new special tokens: <|sot|><|user|>prompt<|eot|><|sot|><|assistant|>
    prompt_tokens = tokenizer.encode(prompt, allowed_special=set()) # Encode prompt text only
    input_ids = [sot_token_id, user_token_id] + prompt_tokens + [eot_token_id, sot_token_id, assistant_token_id]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    generated_ids = input_ids[:]
    max_seq_len = model.config.block_size

    stop_token_ids = {eot_token_id, eod_token_id}

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Truncate input sequence if it exceeds max length
            current_input_tensor = input_tensor[:, -max_seq_len:]

            # Forward pass
            logits, _ = model(current_input_tensor)
            next_token_logits = logits[:, -1, :] # Logits for the next token

            # Apply temperature
            if temperature > 0 and temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')

            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()

            # Stop generation if a stop token is generated
            if next_token_id in stop_token_ids:
                break

            # Append the token and update the input tensor
            generated_ids.append(next_token_id)
            input_tensor = torch.cat((input_tensor, torch.tensor([[next_token_id]], device=device)), dim=1)


    full_response_text = tokenizer.decode(generated_ids)

    # Extract only the assistant's generated part
    # Find the last occurrence of the assistant start sequence
    assistant_start_sequence = tokenizer.decode([sot_token_id, assistant_token_id])
    marker_pos = full_response_text.rfind(assistant_start_sequence)

    if marker_pos != -1:
        # Start extracting after the marker sequence
        assistant_response = full_response_text[marker_pos + len(assistant_start_sequence):]
    else:
        # Fallback: if marker not found (unexpected), return everything after initial prompt
        logging.warning("Assistant start marker not found in generated text. Returning text after initial prompt.")
        prompt_decode_len = len(tokenizer.decode(input_ids))
        assistant_response = full_response_text[prompt_decode_len:]


    # Clean up potential trailing special tokens (like <|eot|>) and whitespace
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