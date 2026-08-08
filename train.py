"""
Train TinyGPT from scratch on data/input.txt. Everything here is real
training: real gradient descent, real backpropagation, real loss going
down — just at a small enough scale to finish on a CPU in a few minutes
instead of needing a datacenter.
"""

import torch
import time
from model import TinyGPT

# ---------------------------------------------------------------------
# Hyperparameters — small enough to train on CPU in a reasonable time
# ---------------------------------------------------------------------
BLOCK_SIZE = 96        # how many characters of context the model sees
BATCH_SIZE = 48
N_EMBD = 96
N_HEAD = 4
N_LAYER = 3
DROPOUT = 0.1
LEARNING_RATE = 3e-4
MAX_ITERS = 3000
EVAL_INTERVAL = 200
EVAL_ITERS = 30

torch.manual_seed(1337)

# ---------------------------------------------------------------------
# Data loading + tokenization (character-level: each unique character
# in the corpus becomes one token)
# ---------------------------------------------------------------------
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# ---------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------
if __name__ == '__main__':
    print(f'Vocab size: {vocab_size} unique characters')
    print(f'Training data: {len(train_data):,} chars | Validation: {len(val_data):,} chars')

    model = TinyGPT(vocab_size, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, DROPOUT)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {n_params:,}')

    import os
    start_iter = 0
    if os.path.exists('tinygpt.pt'):
        ckpt = torch.load('tinygpt.pt', map_location='cpu')
        model.load_state_dict(ckpt['model_state'])
        start_iter = ckpt.get('iter', 0) + 1
        print(f"Resuming from existing checkpoint at iter {start_iter} (train loss was {ckpt.get('train_loss', '?')})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    start = time.time()
    for iter in range(start_iter, MAX_ITERS):
        if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
            losses = estimate_loss(model)
            elapsed = time.time() - start
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} ({elapsed:.0f}s elapsed)")

            # Save a checkpoint every eval, so an interrupted run still
            # leaves behind a usable (if less-trained) model file.
            torch.save({
                'model_state': model.state_dict(),
                'vocab_size': vocab_size,
                'stoi': stoi,
                'itos': itos,
                'config': {
                    'n_embd': N_EMBD, 'n_head': N_HEAD, 'n_layer': N_LAYER,
                    'block_size': BLOCK_SIZE, 'dropout': DROPOUT,
                },
                'iter': iter,
                'train_loss': losses['train'],
                'val_loss': losses['val'],
            }, 'tinygpt.pt')

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f'\nTraining complete in {time.time() - start:.0f}s')
    print('Final checkpoint already saved to tinygpt.pt')

    # Generate a sample to prove it actually learned something
    print('\n--- Sample generation ---')
    context = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(context, max_new_tokens=400)
    print(decode(generated[0].tolist()))
