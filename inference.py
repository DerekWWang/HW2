import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
from model import GPT, GPTConfig
from tokenizer import tokenize
import csv

device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = GPTConfig(
    block_size=32,  
    vocab_size=15,
    n_layer=7,
    n_head=4,
    n_embd=32,
    dropout=0.0,
    bias=True
)

# Edit model path here
MODEL_PATH = "/Users/derekzhu/Code/HW2/part_2.4_model.pt"
# MODEL = torch.load(MODEL_PATH, map_location="mps")

# 1. Initialize model
MODEL = GPT(config)

# 2. Load weights
MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=device))

# 3. Prepare for inference
MODEL = MODEL.to(device)
MODEL.eval()

# === Create encode and decode maps === #
DATASET_TEST = "/Users/derekzhu/Code/HW2/data/splits/mod_div_test.csv"

def concat_csv_strings(filepath):
    full_string = ""
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            full_string += ' '.join(row) + ' '
    return full_string.strip()

chars = concat_csv_strings(DATASET_TEST)
vocab = sorted(set(tokenize(chars)))
print(vocab)
stoi = {ch: i for i, ch in enumerate(vocab)}
print(stoi)
itos = {i: ch for ch, i in stoi.items()}
print(itos)
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# 6. Generation
def generate(model, idx, max_new_tokens):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -config.block_size:] 
        logits = model(idx_cond)
        logits = logits[:, -1, :]  
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.argmax(probs, dim=-1)
        idx = torch.cat((idx, idx_next[:, None]), dim=1)

        print(idx)
        if idx[0][-1] == 12:
            break
    return idx

context = "<s>18 / 2" 
tokens = tokenize(context)
context_ids = torch.tensor([[stoi[c] for c in tokens]], dtype=torch.long).to(device)
generated_ids = generate(MODEL, context_ids, max_new_tokens=15)[0].tolist()
print("Generated:", decode(generated_ids))
