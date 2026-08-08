"""
Simple interactive REPL for talking to your trained TinyGPT model.

Note on what this actually is: TinyGPT is trained purely to predict the
next character given the previous ones (this is called "next-token
prediction", the same core task real LLMs are trained on) — it has NOT
been fine-tuned to follow instructions or hold a conversation the way
ChatGPT has. So typing a question won't get you a helpful answer; it'll
get you a Shakespeare-flavored continuation of whatever you typed. This
script is here so you can feel that difference for yourself.

Usage:
    python3 chat.py
"""

import torch
from sample import load_model, encode, decode

print("Loading tinygpt.pt...")
model, stoi, itos = load_model()
print("Loaded. Type some text and press Enter — TinyGPT will continue it")
print("in the style of the Shakespeare text it was trained on.")
print("Type 'quit' to exit.\n")

while True:
    prompt = input("You: ")
    if prompt.strip().lower() in ('quit', 'exit'):
        break
    context = torch.tensor([encode(prompt, stoi)], dtype=torch.long)
    generated = model.generate(context, max_new_tokens=200, temperature=0.8)
    print("TinyGPT:", decode(generated[0].tolist(), itos))
    print()
