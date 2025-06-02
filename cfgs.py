from model import GPTConfig

"""
Meant for part 1.5, Sanity Checking the Model
Vocab is count of 16 (<s>, <e> + "I love machine learning" chars)
""" 
CONFIG_SANITY = GPTConfig(
    block_size=1024,
    vocab_size=16,
    n_layer=1,
    n_head=12,
    n_embd=768,
    dropout=0,
    bias=True
)

### 