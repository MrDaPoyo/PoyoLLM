import os
import numpy as np
import multiprocessing as mp
from tqdm import tqdm # pip install tqdm
from datasets import load_dataset # pip install datasets
import tiktoken # pip install tiktoken

# ------------------------------------------
# Configuration
# ------------------------------------------
local_dir = "edu_fineweb10B_tiktoken_cl100k" # Changed dir name for tiktoken
remote_name = "sample-10BT"
shard_size = int(1e8) # 100M tokens per shard
tiktoken_encoding_name = "cl100k_base" # Choose your tiktoken encoding

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
print("Loading dataset...")
# Using streaming=True is generally recommended for large datasets
# fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train", streaming=True)
# For faster iteration during testing, load normally (requires more RAM):
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")
print("Dataset loaded.")

# ------------------------------------------
# Tiktoken Tokenizer Setup
# ------------------------------------------
print(f"Initializing tiktoken tokenizer: {tiktoken_encoding_name}")
enc = tiktoken.get_encoding(tiktoken_encoding_name)
print(f"Tokenizer vocabulary size: {enc.n_vocab}")

# Get the ID for the End-of-Text token, commonly used as a separator
# Note: Special tokens might vary depending on the encoding.
# cl100k_base uses <|endoftext|>
try:
    # Attempt to encode the standard EOT token for cl100k_base
    # Use encode_single_token for robustness if available, otherwise encode with allowed_special
    # eot_token_str = "<|endoftext|>" # Standard for cl100k_base
    # eot = enc.encode(eot_token_str, allowed_special={eot_token_str})[0]
    # Simpler approach if you know the token exists:
    eot = enc.eot_token # Directly access the EOT token ID property
    print(f"Using EOT token ID: {eot}")
except AttributeError:
    print(f"Warning: Encoding '{tiktoken_encoding_name}' might not have a standard .eot_token attribute. Trying to encode '<|endoftext|>'.")
    try:
        eot_token_str = "<|endoftext|>"
        eot = enc.encode(eot_token_str, allowed_special="all")[0] # Allow all special tokens during this check
        print(f"Found EOT token ID for '<|endoftext|>': {eot}")
    except Exception as e:
        print(f"Error: Could not find or encode an EOT token for '{tiktoken_encoding_name}'. Please check the encoding's special tokens. Error: {e}")
        exit(1) # Cannot proceed without a separator token

# Check if vocab size fits uint16
max_token_id = enc.n_vocab - 1
if max_token_id >= 2**16:
    raise ValueError(f"Tokenizer vocabulary size ({enc.n_vocab}) exceeds uint16 limit (65536)")
else:
    print("Vocabulary size fits within uint16.")


# ------------------------------------------
# Define Tokenization Function for Dataset
# ------------------------------------------
def tokenize_doc(doc):
    """Tokenizes a single document using the tiktoken tokenizer."""
    # Ensure 'text' field exists and is string
    text = doc.get("text", "")
    if not isinstance(text, str):
        text = str(text) # Attempt to convert to string if not already

    # Encode the text using tiktoken, allowing special tokens if needed by the model architecture
    # For simple text tokenization, often disallowed_special=() is fine.
    # If your model expects special tokens within the text, use allowed_special="all" or specify them.
    tokens = enc.encode(text, disallowed_special=())

    # Prepend the EOT token ID (or other separator)
    # Note: Some models might expect EOT at the end, or both start/end. Adjust as needed.
    tokens_with_eot = [eot] + tokens
    tokens_np = np.array(tokens_with_eot)

    # Convert to uint16
    # No need to clamp here as tiktoken should only return valid IDs within its vocab range.
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

# ------------------------------------------
# Define Data Writing Function
# ------------------------------------------
def write_datafile(filename, tokens_np):
    """Saves token array to a .npy file."""
    np.save(filename, tokens_np)
    # print(f"Saved shard: {filename}.npy") # Reduce print frequency

# ------------------------------------------
# Process Dataset and Write Shards
# ------------------------------------------
if __name__ == '__main__':
    nprocs = max(1, os.cpu_count() // 2)
    print(f"Using {nprocs} processes for tokenization.")

    # --- Multiprocessing Sharding Logic ---
    shard_index = 0
    # Use a buffer list to accumulate token arrays from documents
    buffer_tokens_list = []
    current_buffer_size = 0
    total_tokens_processed = 0
    docs_processed_count = 0

    print("Starting tokenization and sharding...")
    # Initialize progress bar for documents
    # Note: fw might be an iterable dataset, len(fw) might not be efficient/available.
    # total_docs = len(fw) # This might be slow or fail for iterable datasets
    # Use tqdm without total if length is unknown or costly
    doc_progress_bar = tqdm(desc="Processing Documents", unit="docs")

    # Use Pool.imap for memory efficiency with large datasets
    with mp.Pool(nprocs) as pool:
        # Adjust chunksize based on typical document size and memory constraints
        for tokens_np_uint16 in pool.imap(tokenize_doc, fw, chunksize=128):
            buffer_tokens_list.append(tokens_np_uint16)
            current_buffer_size += len(tokens_np_uint16)
            docs_processed_count += 1
            doc_progress_bar.update(1)

            # While buffer has enough tokens for at least one full shard
            while current_buffer_size >= shard_size:
                tokens_to_write = []
                tokens_collected = 0
                # Collect arrays from buffer until shard_size is reached or exceeded
                while tokens_collected < shard_size:
                    if not buffer_tokens_list: break # Safety check
                    arr = buffer_tokens_list.pop(0)
                    arr_len = len(arr)
                    needed = shard_size - tokens_collected

                    if arr_len <= needed:
                        tokens_to_write.append(arr)
                        tokens_collected += arr_len
                        current_buffer_size -= arr_len
                    else:
                        # Split the array
                        tokens_to_write.append(arr[:needed])
                        # Put the remainder back at the beginning of the buffer
                        buffer_tokens_list.insert(0, arr[needed:])
                        tokens_collected += needed # = shard_size
                        current_buffer_size -= needed
                        break # Shard is full

                # Concatenate collected arrays and write the shard
                if tokens_to_write:
                    # Ensure concatenation happens correctly
                    shard_tokens = np.concatenate(tokens_to_write)
                    # Verify the final type is still uint16 after potential concatenation
                    if shard_tokens.dtype != np.uint16:
                        shard_tokens = shard_tokens.astype(np.uint16)

                    if len(shard_tokens) == shard_size: # Ensure correctness
                        # Determine split (e.g., first shard as validation)
                        split = "val" if shard_index == 0 else "train"
                        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
                        write_datafile(filename, shard_tokens)
                        total_tokens_processed += len(shard_tokens)
                        print(f"  >> Shard {shard_index} written ({len(shard_tokens)} tokens). Buffer size: {current_buffer_size} tokens.")
                        shard_index += 1
                    else:
                         # This indicates a logic error in buffer handling or calculation
                         print(f"Error: Shard {shard_index} size mismatch. Expected {shard_size}, got {len(shard_tokens)}. Skipping write.")


    doc_progress_bar.close()

    # Write any remaining tokens in the buffer as the last shard
    if buffer_tokens_list:
        print("Writing final shard...")
        # Concatenate remaining arrays
        final_tokens = np.concatenate(buffer_tokens_list)
        # Ensure final type is uint16
        if final_tokens.dtype != np.uint16:
            final_tokens = final_tokens.astype(np.uint16)

        final_token_count = len(final_tokens)
        if final_token_count > 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            write_datafile(filename, final_tokens)
            total_tokens_processed += final_token_count
            print(f"  >> Final Shard {shard_index} written ({final_token_count} tokens).")
            shard_index += 1
        else:
            print("Buffer contained empty arrays or became empty. No final shard written.")


    print(f"\nTokenization complete.")
    print(f"Total documents processed: {docs_processed_count}")
    print(f"Total tokens processed: {total_tokens_processed}")
    print(f"Total shards written: {shard_index}")
    print(f"Shards saved in: {DATA_CACHE_DIR}")
