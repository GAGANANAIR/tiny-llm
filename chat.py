
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
