# Processing the data for PoyoLLM
# First of all we process and refine our data, just like in MacroData Refinement from severance.

from datasets import load_dataset, get_dataset_split_names

print(get_dataset_split_names("Helsinki-NLP/opus_books", "ca-en"))

ds = load_dataset("Helsinki-NLP/opus_books", "ca-en")

import os

print(f"There's a total of {ds.num_rows["train"]} rows in the dataset.")
filtered_ds = ds.filter(lambda x: x["translation"]["en"] is not None)
single_string = "".join([f'{x["translation"]["en"]} ' for x in filtered_ds["train"]])
print(single_string[:1000])  # Print the first 1000 characters of the concatenated string

os.makedirs("output", exist_ok="true")
with open("output/single_string.txt", "w") as file:
    file.write(single_string)


# ## Tokenizer
# I'm gonna use minBPE :3

with open("output/single_string.txt", "r") as file:
    text_sequence = file.read()

len(text_sequence)

import sys
sys.path.append('')

from minbpe import BasicTokenizer

tokenizer = BasicTokenizer()
tokenizer.train(text_sequence, vocab_size=1024)

vocab = tokenizer.vocab
vocab

max_vocab_id = list(tokenizer.vocab.keys())[-1]
tokenizer.special_tokens = {
    "<|startofstring|>": max_vocab_id + 1,
    "<|separator|>": max_vocab_id + 2,
    "<|endofstring|>": max_vocab_id + 3,
    "<|unk|>": max_vocab_id + 4
}

len(tokenizer.encode(text_sequence))

os.makedirs("output/tokenizer", exist_ok="true")
tokenizer.save(file_prefix="output/tokenizer/poyo_tokenizer")
