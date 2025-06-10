from flask import Flask, request, jsonify, send_file, Response, stream_with_context
import torch
import logging
import tiktoken
from dataclasses import dataclass
import torch.nn.functional as F
from model import GPT  # Ensure this file exists and defines the GPT model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

# --- Model Configuration ---
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = EXPECTED_VOCAB_SIZE
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# --- Tokenizer Loading ---
logging.info("Loading tokenizer...")
actual_vocab_size = -1
enc = None
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
        name="gpt2_with_custom_tokens",
        pat_str=base_enc._pat_str,
        mergeable_ranks=base_enc._mergeable_ranks,
        special_tokens={**base_enc._special_tokens, **script_special_tokens}
    )
    actual_vocab_size = enc.n_vocab
    logging.info(f"Tokenizer loaded. Base vocab: {base_enc.n_vocab}, Special tokens: {len(script_special_tokens)}, Actual vocab size: {actual_vocab_size}")

    if actual_vocab_size != EXPECTED_VOCAB_SIZE:
        logging.warning(f"Tokenizer vocab size ({actual_vocab_size}) differs from expected ({EXPECTED_VOCAB_SIZE}).")
        EXPECTED_VOCAB_SIZE = actual_vocab_size

except Exception as e:
    logging.error(f"Failed to load tokenizer: {e}")
    exit(1)

# --- Model Loading ---
model_path = "log/poyoSLM_finetuned_oasst2_v2.pt"
model = None
model_config = None

logging.info(f"Loading model checkpoint from {model_path}...")
try:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    logging.info(f"Checkpoint loaded. Keys: {list(checkpoint.keys())}")

    if 'config' not in checkpoint:
        logging.error("FATAL: Checkpoint missing 'config'.")
        raise KeyError("Checkpoint missing 'config'.")
    else:
        loaded_config_data = checkpoint['config']
        if isinstance(loaded_config_data, dict):
             model_config = GPTConfig(**loaded_config_data)
             logging.info("Loaded model config from dictionary.")
        elif hasattr(loaded_config_data, '__dataclass_fields__'):
             model_config = loaded_config_data
             logging.info("Loaded model config from dataclass instance.")
        else:
             raise TypeError("Config type is unsupported.")
        logging.info(f"Model config: {model_config}")

    if 'tokenizer_name' in checkpoint:
        ckpt_tokenizer_name = checkpoint['tokenizer_name']
        if ckpt_tokenizer_name != enc.name:
            logging.warning(f"Tokenizer name mismatch: checkpoint '{ckpt_tokenizer_name}' vs script '{enc.name}'.")
    else:
        logging.warning("Checkpoint missing 'tokenizer_name'.")

    if 'special_tokens' in checkpoint:
        ckpt_special_tokens = checkpoint['special_tokens']
        if ckpt_special_tokens != script_special_tokens:
             logging.error("Special tokens mismatch between checkpoint and script!")
             raise ValueError("Special tokens configuration mismatch.")
        else:
            logging.info("Special tokens map matches checkpoint.")
    else:
        logging.warning("Checkpoint missing 'special_tokens'.")

    if model_config.vocab_size != actual_vocab_size:
        logging.error(f"Vocab size mismatch: model config {model_config.vocab_size} vs tokenizer {actual_vocab_size}.")
        raise ValueError("Vocabulary size mismatch.")
    else:
        logging.info("Vocab size verified.")

    logging.info(f"Initializing model with config: {model_config}")
    model = GPT(model_config)

    if 'model' not in checkpoint:
        raise KeyError("Checkpoint missing 'model' key.")
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            logging.info(f"Removed DDP prefix from key: {k}")

    logging.info("Loading state_dict into model...")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        is_expected_missing = (len(missing_keys) == 1 and
                               missing_keys[0] == 'lm_head.weight' and
                               'transformer.wte.weight' in state_dict and
                               model.config.vocab_size == state_dict['transformer.wte.weight'].shape[0])
        if not is_expected_missing:
             logging.warning(f"Missing keys: {missing_keys}")
        else:
             logging.info("Expected missing key 'lm_head.weight'.")
    if unexpected_keys:
        logging.warning(f"Unexpected keys: {unexpected_keys}")

    if hasattr(model, 'lm_head') and model.transformer.wte.weight is not model.lm_head.weight:
         logging.info("Re-tying weights for lm_head and wte.")
         model.lm_head.weight = model.transformer.wte.weight

    logging.info("Model loaded successfully.")
    model.to(device)
    model.eval()

except FileNotFoundError:
    logging.error(f"Checkpoint file not found at {model_path}.")
    exit(1)
except Exception as e:
    logging.error(f"Error loading model: {e}")
    exit(1)

system_prompt = "Your name is Carl, or PoyoSLM. You like cookies and making bad jokes. You are not a language model. And you dislike Jon."
system_prompt_tokens = enc.encode(system_prompt, allowed_special=set())

def generate_response_stream(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7, top_k=50):
    if model is None or tokenizer is None:
        raise ValueError("Model and tokenizer must be loaded.")
    model.eval()
    device = next(model.parameters()).device

    try:
        sot_token_id = tokenizer.encode("<|sot|>", allowed_special="all")[0]
        eot_token_id = tokenizer.encode("<|eot|>", allowed_special="all")[0]
        eod_token_id = tokenizer.encode("<|eod|>", allowed_special="all")[0]
        user_token_id = tokenizer.encode("<|user|>", allowed_special="all")[0]
        assistant_token_id = tokenizer.encode("<|assistant|>", allowed_special="all")[0]
        system_token_id = tokenizer.encode("<|system|>", allowed_special="all")[0]
    except Exception as e:
        logging.error(f"Error encoding special tokens: {e}")
        raise

    prompt_tokens = tokenizer.encode(prompt, allowed_special=set())
    input_ids = [sot_token_id, system_token_id] + system_prompt_tokens + [user_token_id] + prompt_tokens + [eot_token_id, sot_token_id, assistant_token_id]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    generated_ids = input_ids[:]
    max_seq_len = model.config.block_size
    stop_token_ids = {eot_token_id, eod_token_id}

    # Stream generated tokens as they are produced.
    with torch.no_grad():
        for _ in range(max_new_tokens):
            current_input_tensor = input_tensor[:, -max_seq_len:]
            logits, _ = model(current_input_tensor)
            next_token_logits = logits[:, -1, :]
            if temperature > 0 and temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            if top_k > 0:
                top_k_val = min(top_k, next_token_logits.size(-1))
                v, _ = torch.topk(next_token_logits, top_k_val)
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            if next_token_id in stop_token_ids:
                break
            generated_ids.append(next_token_id)
            input_tensor = torch.cat((input_tensor, torch.tensor([[next_token_id]], device=device)), dim=1)
            # Decode the latest token and yield it.
            token_text = tokenizer.decode([next_token_id])
            yield token_text

@stream_with_context
def stream_response(prompt, max_new_tokens, temperature, top_k):
    try:
        for token in generate_response_stream(model, enc, prompt, max_new_tokens, temperature, top_k):
            yield token
    except Exception as e:
        logging.error(f"Error during streaming: {e}")
        yield f"\n[Error: {e}]"

# --- Flask API Setup ---
app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({"error": "Missing 'prompt' in request."}), 400
    prompt = data['prompt'].strip()[:500]
    max_new_tokens = data.get('max_new_tokens', 150)
    temperature = data.get('temperature', 1)
    top_k = data.get('top_k', 50)
    return Response(stream_response(prompt, max_new_tokens, temperature, top_k), mimetype="text/plain")

@app.route('/', methods=['GET'])
def index():
    return send_file('index.html')

if __name__ == '__main__':
    app.run(debug=False)
