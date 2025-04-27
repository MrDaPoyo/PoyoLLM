import torch
import logging
import tiktoken
from model import GPT
from dataclasses import dataclass
import discord
import os

import torch.nn.functional as F
from dotenv import load_dotenv # Optional: for loading token from .env file

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Special Token IDs (Keep for reference/consistency check, but generation uses tokenizer) ---
PAD_TOKEN_ID = 50257
EOT_TOKEN_ID = 50258
SOT_TOKEN_ID = 50259
EOD_TOKEN_ID = 50260
SYSTEM_TOKEN_ID = 50261
USER_TOKEN_ID = 50262
ASSISTANT_TOKEN_ID = 50263
EXPECTED_BASE_VOCAB_SIZE = 50256
EXPECTED_NUM_SPECIAL_TOKENS = 7
EXPECTED_VOCAB_SIZE = EXPECTED_BASE_VOCAB_SIZE + EXPECTED_NUM_SPECIAL_TOKENS

# --- Model Configuration (Load from checkpoint if available) ---
# Define a default config first - this might be overridden by the checkpoint
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = EXPECTED_VOCAB_SIZE # Default, will be overridden by checkpoint if possible
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_type = "cuda" if device.type == "cuda" else "cpu"
logging.info(f"Using device: {device}")

# --- Tokenizer Loading ---
logging.info("Loading tokenizer...")
actual_vocab_size = -1 # Initialize
enc = None
# Define the special tokens map expected by this script
script_special_tokens = {
    "<|pad|>": PAD_TOKEN_ID,
    "<|eot|>": EOT_TOKEN_ID,
    "<|sot|>": SOT_TOKEN_ID,
    "<|eod|>": EOD_TOKEN_ID,
    "<|system|>": SYSTEM_TOKEN_ID,
    "<|user|>": USER_TOKEN_ID,
    "<|assistant|>": ASSISTANT_TOKEN_ID,
}
try:
    base_enc = tiktoken.get_encoding("gpt2")
    enc = tiktoken.Encoding(
        name="gpt2_with_custom_tokens", # This name might differ from checkpoint, we'll check later
        pat_str=base_enc._pat_str,
        mergeable_ranks=base_enc._mergeable_ranks,
        special_tokens={**base_enc._special_tokens, **script_special_tokens}
    )
    actual_vocab_size = enc.n_vocab
    logging.info(f"Tokenizer loaded. Base vocab: {base_enc.n_vocab}, Target special tokens: {len(script_special_tokens)}, Actual tokenizer vocab size: {actual_vocab_size}")

    # Check if tokenizer vocab size matches the expected size based on constants
    if actual_vocab_size != EXPECTED_VOCAB_SIZE:
        logging.warning(f"Tokenizer actual vocab size ({actual_vocab_size}) differs from expected ({EXPECTED_VOCAB_SIZE}). This might be okay if tiktoken merged tokens or base vocab changed.")
        # Update EXPECTED_VOCAB_SIZE to match reality for model loading checks
        EXPECTED_VOCAB_SIZE = actual_vocab_size

except Exception as e:
    logging.error(f"Failed to load tokenizer: {e}")
    exit(1)


# --- Model Loading ---
model_path = "log/poyoSLM_finetuned_oasst2_v2.pt"
model = None
model_config = None

# Allow torch to load the GPTConfig dataclass if saved directly
torch.serialization.add_safe_globals([GPTConfig])

logging.info(f"Loading model checkpoint from {model_path}...")
try:
    # Load the entire checkpoint dictionary
    # Set weights_only=False as the checkpoint contains non-tensor data (config, etc.)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    logging.info(f"Checkpoint loaded. Keys: {list(checkpoint.keys())}")

    # --- Configuration Loading ---
    if 'config' not in checkpoint:
        logging.error("FATAL: Checkpoint does not contain 'config'. Cannot determine model parameters.")
        raise KeyError("Checkpoint missing 'config' key.")
    else:
        # Load config from checkpoint
        loaded_config_data = checkpoint['config']
        # Handle if config is saved as dataclass or dict
        if isinstance(loaded_config_data, dict):
             model_config = GPTConfig(**loaded_config_data)
             logging.info("Loaded model config from dictionary in checkpoint.")
        elif hasattr(loaded_config_data, '__dataclass_fields__'): # Check if it's a dataclass instance
             model_config = loaded_config_data
             logging.info("Loaded model config from dataclass object in checkpoint.")
        else:
             raise TypeError("Loaded config is neither a dict nor a dataclass instance.")
        logging.info(f"Model config loaded from checkpoint: {model_config}")

    # --- Tokenizer Verification (using info from checkpoint) ---
    if 'tokenizer_name' in checkpoint:
        ckpt_tokenizer_name = checkpoint['tokenizer_name']
        if ckpt_tokenizer_name != enc.name:
            logging.warning(f"Tokenizer name mismatch: Checkpoint used '{ckpt_tokenizer_name}', script loaded '{enc.name}'.")
    else:
        logging.warning("Checkpoint does not contain 'tokenizer_name'. Cannot verify tokenizer consistency.")

    if 'special_tokens' in checkpoint:
        ckpt_special_tokens = checkpoint['special_tokens']
        # Compare the loaded special tokens map with the one defined in this script
        if ckpt_special_tokens != script_special_tokens:
             logging.error("FATAL: Special tokens mismatch between checkpoint and script!")
             logging.error(f"Checkpoint special tokens: {ckpt_special_tokens}")
             logging.error(f"Script special tokens: {script_special_tokens}")
             raise ValueError("Special tokens configuration mismatch. Ensure inference script uses the same special tokens as training.")
        else:
            logging.info("Special tokens map in checkpoint matches the script.")
    else:
        logging.warning("Checkpoint does not contain 'special_tokens'. Cannot verify special token consistency.")


    # --- Vocabulary Size Verification ---
    if model_config.vocab_size != actual_vocab_size:
        logging.error(f"FATAL: Loaded model config vocab size ({model_config.vocab_size}) "
                      f"does not match loaded tokenizer vocab size ({actual_vocab_size}).")
        # This check is critical. If they don't match, the embedding/output layers are incompatible.
        raise ValueError("Vocabulary size mismatch between loaded model config and tokenizer.")
    else:
        logging.info(f"Model config vocab size ({model_config.vocab_size}) matches tokenizer vocab size ({actual_vocab_size}).")


    # --- Model Instantiation ---
    logging.info(f"Initializing model with loaded config: {model_config}")
    # Use the imported class name 'GPT'
    model = GPT(model_config)

    # --- State Dictionary Loading ---
    if 'model' not in checkpoint:
        raise KeyError("Checkpoint file does not contain the 'model' key with the state dictionary.")

    state_dict = checkpoint['model']

    # Fix potential DDP prefix (if model was saved from DistributedDataParallel)
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            logging.info(f"Removed DDP prefix from key: {k}")

    # Load the state dict
    logging.info("Loading model state_dict...")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    # Handle missing/unexpected keys (copied logic from original script)
    if missing_keys:
        # Check if missing keys are expected due to weight tying (lm_head might be missing if wte is present)
        is_expected_missing = (len(missing_keys) == 1 and
                               missing_keys[0] == 'lm_head.weight' and
                               'transformer.wte.weight' in state_dict and
                               model.config.vocab_size == state_dict['transformer.wte.weight'].shape[0]) # Check vocab size match too
        if not is_expected_missing:
             logging.warning(f"Missing keys in state_dict: {missing_keys}")
        else:
             logging.info("Missing key 'lm_head.weight' is expected due to weight tying.")
    if unexpected_keys:
        logging.warning(f"Unexpected keys in state_dict: {unexpected_keys}")

    # Ensure weights are tied after loading, as loading might untie them if lm_head was missing
    if hasattr(model, 'lm_head') and model.transformer.wte.weight is not model.lm_head.weight:
         logging.info("Re-tying weights for wte and lm_head after loading.")
         model.lm_head.weight = model.transformer.wte.weight

    logging.info("Model state_dict loaded successfully.")
    model.to(device)
    model.eval() # Set to evaluation mode

except FileNotFoundError:
    logging.error(f"Model checkpoint file not found at {model_path}. Cannot proceed.")
    exit(1)
except KeyError as e:
     logging.error(f"Checkpoint file is missing expected key: {e}", exc_info=True)
     exit(1)
except TypeError as e:
     logging.error(f"Type error during model or config loading: {e}", exc_info=True)
     exit(1)
except ValueError as e:
     logging.error(f"Value error during loading (e.g., vocab mismatch): {e}", exc_info=True)
     exit(1)
except Exception as e:
    logging.error(f"An unexpected error occurred loading model checkpoint: {e}", exc_info=True)
    exit(1)


# --- System Prompt ---
system_prompt = "Your name is Carl, or PoyoSLM. You like cookies and making bad jokes. You are not a language model. And you dislike Jon."
system_prompt_tokens = enc.encode(system_prompt, allowed_special=set()) # Encode system prompt text only

# --- Generation Function (Copied from fine-tuning script - unchanged) ---
def generate_response(model, tokenizer, prompt, max_new_tokens=150, temperature=0.75, top_k=100):
    """Generates a response using the fine-tuned model with new special tokens."""
    if model is None:
        raise ValueError("Model must be loaded before calling generate_response.")
    if tokenizer is None:
        raise ValueError("Tokenizer must be loaded before calling generate_response.")

    model.eval()
    device = next(model.parameters()).device # Get device from model

    # Get token IDs dynamically from the tokenizer instance
    try:
        sot_token_id = tokenizer.encode("<|sot|>", allowed_special="all")[0]
        eot_token_id = tokenizer.encode("<|eot|>", allowed_special="all")[0]
        eod_token_id = tokenizer.encode("<|eod|>", allowed_special="all")[0]
        user_token_id = tokenizer.encode("<|user|>", allowed_special="all")[0]
        assistant_token_id = tokenizer.encode("<|assistant|>", allowed_special="all")[0]
        system_token_id = tokenizer.encode("<|system|>", allowed_special="all")[0]
    except Exception as e:
        logging.error(f"Error encoding special tokens. Ensure tokenizer is loaded correctly: {e}")
        raise

    # Format the prompt using new special tokens: <|sot|><|system|>system_prompt<|user|>prompt<|eot|><|sot|><|assistant|>
    prompt_tokens = tokenizer.encode(prompt, allowed_special=set()) # Encode prompt text only
    input_ids = [sot_token_id] + [user_token_id] + prompt_tokens + [eot_token_id, sot_token_id, assistant_token_id]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    generated_ids = input_ids[:] # Use list copy
    max_seq_len = model.config.block_size

    stop_token_ids = {eot_token_id, eod_token_id}

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Truncate input sequence if it exceeds max length
            # Only feed the last block_size tokens if sequence gets too long
            current_input_tensor = input_tensor[:, -max_seq_len:]

            # Forward pass
            logits, _ = model(current_input_tensor)
            next_token_logits = logits[:, -1, :] # Logits for the very last token position

            # Apply temperature scaling
            if temperature > 0 and temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                # Ensure top_k is not larger than vocab size
                top_k_val = min(top_k, next_token_logits.size(-1))
                # Get the top k logits and their values
                v, _ = torch.topk(next_token_logits, top_k_val)
                # Set all logits below the k-th smallest to -infinity
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')

            # Sample from the filtered distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()

            # Stop generation if a stop token is generated
            if next_token_id in stop_token_ids:
                break

            # Append the generated token and update the input tensor for the next step
            generated_ids.append(next_token_id)
            # Update input_tensor by appending the new token ID
            input_tensor = torch.cat((input_tensor, torch.tensor([[next_token_id]], device=device)), dim=1)

            # Check if sequence length exceeds max_len (optional, handled by truncation above)
            # if input_tensor.size(1) >= max_seq_len:
            #     logging.warning("Generation reached max sequence length.")
            #     break


    # Decode the full generated sequence
    full_response_text = tokenizer.decode(generated_ids)

    # Extract only the assistant's generated part
    # Find the last occurrence of the sequence marking the start of the assistant's turn
    assistant_start_sequence = tokenizer.decode([sot_token_id, assistant_token_id])
    marker_pos = full_response_text.rfind(assistant_start_sequence)

    assistant_response = ""
    if marker_pos != -1:
        # Start extracting text *after* the marker sequence
        assistant_response = full_response_text[marker_pos + len(assistant_start_sequence):]
    else:
        # Fallback: if marker not found (shouldn't happen with correct prompt format)
        # Try to return text after the initial input prompt part
        logging.warning("Assistant start marker not found in generated text. Attempting fallback extraction.")
        try:
            # Decode the original input prompt part to find its length in the full text
            prompt_part_text = tokenizer.decode(input_ids)
            prompt_decode_len = len(prompt_part_text)
            # Ensure the prompt part is actually at the beginning before slicing
            if full_response_text.startswith(prompt_part_text):
                 assistant_response = full_response_text[prompt_decode_len:]
            else:
                 logging.error("Generated text does not start with the expected prompt sequence. Returning full generated text.")
                 assistant_response = full_response_text # Or handle differently
        except Exception as decode_err:
             logging.error(f"Error during fallback extraction: {decode_err}. Returning full text.")
             assistant_response = full_response_text


    # Clean up potential trailing special tokens (like <|eot|>, <|eod|>) and whitespace
    assistant_response = assistant_response.replace("<|eot|>", "").replace("<|eod|>", "").strip()

    return assistant_response


# --- Discord Bot Setup ---
load_dotenv() # Load environment variables from .env file (optional)
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

if not DISCORD_TOKEN:
    logging.error("Discord bot token not found. Set DISCORD_BOT_TOKEN environment variable.")
    exit(1)

if model is None or enc is None:
    logging.error("Model or tokenizer failed to load. Cannot start Discord bot.")
    exit(1)

# Define necessary intents
intents = discord.Intents.default()

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    logging.info(f'Logged in as {client.user.name} (ID: {client.user.id})')
    logging.info('------')
    logging.info("Discord bot is ready and listening for mentions.")

@client.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == client.user:
        return

    # Check if the bot was mentioned
    if client.user.mentioned_in(message):
        # Remove the bot's mention from the message content to get the actual prompt
        # This handles mentions like <@USER_ID> or <@!USER_ID>
        prompt = message.content.replace(f'<@{client.user.id}>', '').replace(f'<@!{client.user.id}>', '').strip()

        if not prompt: # If the message only contained the mention
            prompt = "Hello!" # Default prompt if nothing else was said

        logging.info(f"Received prompt from {message.author}: '{prompt}'")

        # Indicate the bot is thinking
        async with message.channel.typing():
            try:
                # Generate response using the loaded model
                response = generate_response(model, enc, prompt, max_new_tokens=150, temperature=0.85, top_k=50)
                logging.info(f"Generated response: '{response}'")

                # Send the response back to the channel
                # Discord has a 2000 character limit per message
                if len(response) > 2000:
                    response = response[:1997] + "..." # Truncate if too long
                await message.reply(response, mention_author=False) # Use reply for context, don't ping author again

            except Exception as e:
                logging.error(f"Error generating response: {e}", exc_info=True)
                await message.reply("Sorry, I encountered an error trying to respond.", mention_author=False)

# --- Run the Bot ---
logging.info("Starting Discord bot...")
try:
    client.run(DISCORD_TOKEN)
except discord.LoginFailure:
    logging.error("Login failed: Invalid Discord token provided.")
except Exception as e:
    logging.error(f"An error occurred while running the bot: {e}", exc_info=True)