import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
from model import GPT, GPTConfig
from tokenizer import tokenize
import numpy as np
# from data.prepare_dataset import concat_csv_strings
from pathlib import Path
from tqdm import tqdm, trange
import csv



def concat_csv_strings(filepath):
    full_string = ""
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            full_string += ' '.join(row) + ' '
    return full_string.strip()

# ==== Configuration ====
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

EPOCHS = 100000 # 10e5

script_dir = Path(__file__).resolve().parent

DATASET_TRAIN = script_dir / "data/splits/mod_div_train.csv"
DATASET_VALID = script_dir / "data/splits/mod_div_val.csv"
DATASET_TEST = script_dir / "data/splits/mod_div_test.csv"

MODEL_NAME = "part_2.4"

# ==== Reproducibility ====
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
seed_everything()

# === Create encode and decode maps === #
chars = concat_csv_strings(DATASET_TRAIN)
vocab = sorted(set(tokenize(chars)))
print(vocab)
stoi = {ch: i for i, ch in enumerate(vocab)}
print(stoi)
itos = {i: ch for ch, i in stoi.items()}
print(itos)
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

config = GPTConfig(
    block_size=32,  
    vocab_size=len(vocab),
    n_layer=7,
    n_head=4,
    n_embd=32,
    dropout=0.0,
    bias=True
)

data = concat_csv_strings(DATASET_TRAIN)
tokens = tokenize(data)

if len(tokens) < config.block_size + 1:
    pad_len = config.block_size + 1 - len(tokens)
    tokens += ['<e>'] * pad_len

X, Y = [], []
for i in range(len(tokens) - config.block_size):
    chunk_input = tokens[i:i + config.block_size]
    chunk_target = tokens[i + 1:i + config.block_size + 1]
    X.append(encode(chunk_input))
    Y.append(encode(chunk_target))

# Convert to torch tensors
x = torch.tensor(X, dtype=torch.long)
y = torch.tensor(Y, dtype=torch.long)

print("Input shape:", x.shape)   # (num_examples, block_size)
print("Target shape:", y.shape)

device = 'cuda' if torch.cuda.is_available() else 'mps'
model = GPT(config).to(device)
x, y = x.to(device), y.to(device)

optimizer = model.configure_optimizers(
    weight_decay=0.0,
    learning_rate=1e-3,
    betas=(0.9, 0.95),
    device_type=device
)

checkpoint_path = "model_checkpoint.pth"

def save_checkpoint(model, optimizer, step, loss, filename=checkpoint_path):
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at step {step}")

def load_checkpoint(model, optimizer, filename=checkpoint_path):
    if os.path.exists(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from step {checkpoint['step']}")
        return checkpoint['step'] + 1  
    else:
        return 0 

start_step = load_checkpoint(model, optimizer)

loss_log_path = "loss_log.txt"
if start_step == 0:
    with open(loss_log_path, "w") as f:
        f.write("step,loss\n")

for step in range(start_step, EPOCHS):
    logits = model(x)
    B, T, C = logits.shape

    logits = logits.view(B * T, C)
    targets = y.view(B * T)
    loss = F.cross_entropy(logits, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0 or step == EPOCHS - 1:
        print(f"Step {step}: Loss = {loss.item():.9f}")
        save_checkpoint(model, optimizer, step, loss.item())
        
        with open(loss_log_path, "a") as f:
            f.write(f"{step},{loss.item():.9f}\n")

# ==== Evaluation on Validation Set ====
print("\nRunning evaluation on validation set...")

val_data = concat_csv_strings(DATASET_VALID)
val_tokens = tokenize(val_data)

if len(val_tokens) < config.block_size + 1:
    pad_len = config.block_size + 1 - len(val_tokens)
    val_tokens += ['<e>'] * pad_len

X_val, Y_val = [], []
for i in tqdm(range(len(val_tokens) - config.block_size), desc="Encoding validation chunks"):
    chunk_input = val_tokens[i:i + config.block_size]
    chunk_target = val_tokens[i + 1:i + config.block_size + 1]
    X_val.append(encode(chunk_input))
    Y_val.append(encode(chunk_target))

x_val = torch.tensor(X_val, dtype=torch.long).to(device)
y_val = torch.tensor(Y_val, dtype=torch.long).to(device)

model.eval()
with torch.no_grad():
    val_logits = model(x_val)
    B, T, C = val_logits.shape
    val_logits = val_logits.view(B * T, C)
    val_targets = y_val.view(B * T)
    val_loss = F.cross_entropy(val_logits, val_targets)

print(f"Validation Loss = {val_loss.item():.9f}")
model.train()

# ==== Save Model ====
save_path = f"{MODEL_NAME}_model.pt"
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")
