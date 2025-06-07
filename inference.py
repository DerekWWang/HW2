import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
from model import GPT, GPTConfig
from tokenizer import tokenize

chars = "<s>I love machine learning<e>"
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
    n_layer=1,
    n_head=4,
    n_embd=32,
    dropout=0.0,
    bias=True
)

data = "<s>I love machine learning<e>"
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

for step in range(1000):
    logits = model(x)
    B, T, C = logits.shape
    # print("B: ", B)
    # print("T: ", T)
    # print("C: ", C)

    logits = logits.view(B * T, C)
    targets = y.view(B * T)
    loss = F.cross_entropy(logits, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0 or step == 499:
        print(f"Step {step}: Loss = {loss.item():.9f}")

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
        if idx[0][-1] == 1:
            break
    return idx

context = "<s>" 
tokens = tokenize(context)
context_ids = torch.tensor([[stoi[c] for c in tokens]], dtype=torch.long).to(device)
generated_ids = generate(model, context_ids, max_new_tokens=15)[0].tolist()
print("Generated:", decode(generated_ids))
