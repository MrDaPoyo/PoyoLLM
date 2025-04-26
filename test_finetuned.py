import torch
import torch.nn.functional as F
import logging
import tiktoken
# Import GPTConfig first
from model import GPT
from dataclasses import dataclass, field

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
LEARNING_RATE = 3e-5
BATCH_SIZE = 2 # Keep small for memory
EPOCHS = 1 # Reduce epochs for faster testing/demonstration
MAX_LEN = GPTConfig.block_size # Use block_size from config
torch.serialization.safe_globals([GPTConfig])

# Instantiate model structure FIRST
cfg = GPTConfig() # Define config if needed elsewhere
model = GPT(cfg)
model.eval() # Set to evaluation mode

# Load the fine-tuned state dict
model_path = "log/poyoSLM_finetuned_oasst2_v2.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading model state_dict from {model_path}...")
checkpoint = torch.load(model_path, map_location=device) # Load the checkpoint dictionary

# Check if the checkpoint contains the expected keys
if 'model' not in checkpoint:
    raise KeyError("Checkpoint file does not contain the 'model' key.")

state_dict = checkpoint['model']

# Load the state dict into the model instance
try:
    model.load_state_dict(state_dict)
    print("Model state_dict loaded successfully.")
except RuntimeError as e:
    print(f"Error loading state_dict: {e}")
    print("Attempting to load with strict=False...")
    # Try loading non-strictly if there are missing/unexpected keys
    model.load_state_dict(state_dict, strict=False)
    print("Model state_dict loaded with strict=False.")

model.to(device) # Move model to device AFTER loading state_dict

base_enc = tiktoken.get_encoding("gpt2")

# Define special token IDs starting from the end of the base vocabulary
# Ensure these values are consistent with the fine-tuning process
PAD_TOKEN_ID = base_enc.n_vocab + 0
EOT_TOKEN_ID = base_enc.n_vocab + 1
SOT_TOKEN_ID = base_enc.n_vocab + 2
EOD_TOKEN_ID = base_enc.n_vocab + 3
SYSTEM_TOKEN_ID = base_enc.n_vocab + 4
USER_TOKEN_ID = base_enc.n_vocab + 5
ASSISTANT_TOKEN_ID = base_enc.n_vocab + 6
# Update GPTConfig vocab size based on the *final* number of tokens
GPTConfig.vocab_size = base_enc.n_vocab + 7 # Base vocab + number of special tokens

actual_vocab_size = enc.n_vocab
logging.info(f"Tokenizer loaded. Base vocab: {base_enc.n_vocab}, Target special tokens: {len(special_tokens)}, Actual tokenizer vocab size: {actual_vocab_size}")
# Check consistency after defining tokens and updating config
if actual_vocab_size != GPTConfig.vocab_size:
    logging.error(f"FATAL: Tokenizer actual vocab size ({actual_vocab_size}) still differs from expected ({GPTConfig.vocab_size}) after update.")
    # Consider raising an error here depending on desired behavior
    # raise ValueError("Vocabulary size mismatch")

def generate_response(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7, top_k=50):
    """Generates a response using the fine-tuned model with new special tokens."""
    if tokenizer is None:
        raise ValueError("Tokenizer must be loaded and passed to generate_response.")
    model.eval()
    device = next(model.parameters()).device # Get device from model
    max_seq_len = model.config.block_size # Use block_size from model config

    # Get token IDs from the tokenizer instance
    try:
        sot_token_id = tokenizer.encode("<|sot|>", allowed_special="all")[0]
        eot_token_id = tokenizer.encode("<|eot|>", allowed_special="all")[0]
        eod_token_id = tokenizer.encode("<|eod|>", allowed_special="all")[0]
        user_token_id = tokenizer.encode("<|user|>", allowed_special="all")[0]
        assistant_token_id = tokenizer.encode("<|assistant|>", allowed_special="all")[0]
    except Exception as e:
        logging.error(f"Error encoding special tokens. Ensure tokenizer is loaded and contains them: {e}")
        raise # Re-raise the exception

    # Encode the user prompt
    prompt_tokens = tokenizer.encode(prompt, allowed_special="all")

    # Construct the initial input sequence for the model
    # Format: <|sot|> <|user|> prompt <|eot|> <|sot|> <|assistant|>
    input_ids = [sot_token_id, user_token_id] + prompt_tokens + [eot_token_id, sot_token_id, assistant_token_id]
    generated_ids = list(input_ids) # Start generated sequence with the input prompt
    input_tensor = torch.tensor([generated_ids], dtype=torch.long, device=device)

    # Define stop token IDs
    stop_token_ids = {eot_token_id, eod_token_id}

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Truncate input sequence if it exceeds max length
            current_input_tensor = input_tensor[:, -max_seq_len:]

            # Forward pass
            logits, _ = model(current_input_tensor)
            next_token_logits = logits[:, -1, :] # Logits for the next token

    # Clean up potential trailing special tokens (like <|eot|>) and whitespace
    assistant_response = assistant_response.replace("<|eot|>", "").replace("<|eod|>", "").strip()
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
            # Append the token and update the input tensor
            generated_ids.append(next_token_id)
            # Update input_tensor for the next iteration
            input_tensor = torch.tensor([generated_ids], dtype=torch.long, device=device)


    full_response_text = tokenizer.decode(generated_ids)

    # Extract only the assistant's generated part
    # Find the last occurrence of the assistant start sequence

    # Stop generation if a stop token is generated
    if not next_token_id in stop_token_ids:

        logging.warning("Assistant start marker not found in generated text. Returning text after initial prompt.")
        # Decode the original input prompt part to find its length in the full text
        prompt_part_text = tokenizer.decode(input_ids)
        prompt_decode_len = len(prompt_part_text)
        # Ensure the prompt part is actually at the beginning before slicing
        if full_response_text.startswith(prompt_part_text):
             assistant_response = full_response_text[prompt_decode_len:]
        else:
             logging.error("Generated text does not start with the expected prompt sequence. Returning full generated text.")
             assistant_response = full_response_text # Or handle differently


        # Clean up potential trailing special tokens (like <|eot|>) and whitespace
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