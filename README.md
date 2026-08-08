# TinyGPT

A small GPT-style language model, **built and trained from scratch** — not a wrapper around someone else's API, not a fine-tune of a pretrained model. Every weight in `tinygpt.pt` was learned by gradient descent, right here, from a blank random initialization.

## What this actually is (read this first)

This is a **character-level transformer**, trained on ~1MB of Shakespeare text (the standard "tiny Shakespeare" dataset used in most from-scratch LLM tutorials — public domain). It has:
- **356,609 parameters** — for comparison, GPT-3 has 175 *billion*. This is about 500,000x smaller.
- Been trained for 3,000 steps on a single CPU core in about 13 minutes total.
- Learned to predict the next *character*, not the next *word* or *concept* — it doesn't understand language, it's learned statistical patterns of English/Shakespearean text at the character level.

**What it can do:** generate text that looks structurally like Shakespeare — character names in caps followed by colons, roughly English-shaped words, punctuation in plausible places, occasional real words and short real phrases.

**What it can't do:** hold a conversation, answer questions, follow instructions, or produce coherent long-form text. It was never trained to do any of that — real assistants like ChatGPT are additionally fine-tuned on millions of human conversations after this same kind of base training, which is a separate, much larger undertaking.

This project exists to show *how* an LLM works and *actually build one*, not to compete with production AI assistants — for an assistant you can actually talk to, see the companion [`mychat`](https://github.com/GAGANANAIR/mychat) project instead.

## Sample output (from the trained model)

```
Hat the but caboth hose nould ben bourd,
Whith pupcion; goven to this is ie, grater,
I with hall to'dounfur.

Thend she thy his your stakecous: didh
Thou and sons I lif; teye yourt cel of thouk ton'd By of
Thong my a my seen to liek nhistall,
no anstidee.

INESTINER:
Yenry I you bether thee a dis he nicknows ther
gis he ofust briabe wand shat ourd togusuent;
```

Not coherent — but notice the structure: capitalized character-name-like tokens followed by colons, apostrophes in plausible spots, sentence-like punctuation. That structure is *learned*, not hardcoded — the model started from random noise and figured this out purely from the training data.

## Architecture

Same fundamental design as production LLMs, just tiny:
- Token + positional embeddings
- 3 stacked transformer blocks, each with:
  - Multi-head causal self-attention (4 heads)
  - Feedforward network
  - Layer normalization + residual connections
- Final linear layer projecting to vocabulary logits

See `model.py` — it's ~120 lines and implements the whole thing with plain PyTorch, no external transformer library.

## Running it

```bash
git clone https://github.com/GAGANANAIR/tiny-llm.git
cd tiny-llm
pip install -r requirements.txt

# Generate text from the already-trained model
python3 sample.py --prompt "ROMEO:" --tokens 400

# Or talk to it interactively (see the note above on what to expect)
python3 chat.py

# Or train it yourself from scratch (deletes/overwrites tinygpt.pt)
rm tinygpt.pt
python3 train.py
```

Training automatically checkpoints every 200 steps and can resume if interrupted — just run `python3 train.py` again and it'll pick up where it left off, as long as `tinygpt.pt` still exists.

## Making it better

Things that would improve output quality, roughly in order of impact:
1. **Train longer** — loss was still dropping at 3,000 steps; 10,000+ would help noticeably
2. **Bigger model** — more layers/embedding dimensions (costs more compute)
3. **Word or subword tokenization** instead of character-level (what real LLMs use)
4. **More/better training data**
5. **Instruction fine-tuning** on conversation data — this is the step that turns a base model into something like ChatGPT

## Author

**Gagan A Nair**
- [Website](https://gagagananair.netlify.app/)
