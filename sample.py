"""
Load the trained TinyGPT model and generate text from it. Run this after
train.py has produced tinygpt.pt.

Usage:
    python3 sample.py                      # generate from empty context
    python3 sample.py --prompt "ROMEO:"    # generate continuing a prompt
    python3 sample.py --tokens 800         # generate more/less text
    python3 sample.py --temperature 0.7    # lower = safer/more repetitive, higher = wilder
"""

import argparse
import torch
from model import TinyGPT


def load_model(path='tinygpt.pt'):
    checkpoint = torch.load(path, map_location='cpu')
    cfg = checkpoint['config']
    model = TinyGPT(
        checkpoint['vocab_size'], cfg['n_embd'], cfg['n_head'],
        cfg['n_layer'], cfg['block_size'], cfg['dropout']
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, checkpoint['stoi'], checkpoint['itos']


def encode(s, stoi):
    return [stoi.get(c, 0) for c in s]


def decode(tokens, itos):
    return ''.join([itos[i] for i in tokens])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='')
    parser.add_argument('--tokens', type=int, default=500)
    parser.add_argument('--temperature', type=float, default=0.8)
    args = parser.parse_args()

    model, stoi, itos = load_model()

    if args.prompt:
        context = torch.tensor([encode(args.prompt, stoi)], dtype=torch.long)
    else:
        context = torch.zeros((1, 1), dtype=torch.long)

    generated = model.generate(context, max_new_tokens=args.tokens, temperature=args.temperature)
    print(decode(generated[0].tolist(), itos))
