"""
Train 32K BPE tokenizer using HuggingFace tokenizers (Rust-based, fast)
- Fixes \n merging by switching to the robust ByteLevel pre-tokenizer.
- ByteLevel handles all whitespace (including \n) and punctuation inherently.
"""

import os
import time
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, decoders

CORPUS_FILE = "fineweb_edu_1gb_uncased.txt"
VOCAB_SIZE = 32000
OUTPUT_FILE = "bpe-32k-uncased-bytelevel.json"

print(f"Training {VOCAB_SIZE} vocab BPE on {CORPUS_FILE}...")

# 1. Initialize the BPE model
tokenizer = Tokenizer(models.BPE())

# 2. Normalization: Use ByteLevel's default normalization, which is none,
#    or apply lowercase *before* tokenization if truly uncased is required.
#    Note: For true ByteLevel BPE, normalization is often skipped.
tokenizer.normalizer = normalizers.Lowercase()

# 3. Pre-tokenization FIX: Use ByteLevel.
#    This handles splitting by whitespace (including \n) and isolating punctuation
#    naturally, eliminating the need for complex Sequences.
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

# 4. Decoder: Use ByteLevel decoder to correctly reassemble the bytes
tokenizer.decoder = decoders.ByteLevel()

# 5. Trainer: Configure the trainer.
trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    show_progress=True,
    # No explicit special_tokens argument used here.
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
)

t0 = time.time()
tokenizer.train([CORPUS_FILE], trainer)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

tokenizer.save(OUTPUT_FILE)
print(f"Saved to {OUTPUT_FILE}")

# Test
test = "Hello World,\nthis is a test. 12323132+4 23= 52"
encoded = tokenizer.encode(test)
print(f"\nTest: '{test}'")
print(f"Tokens: {encoded.tokens}")
print(f"IDs: {encoded.ids}")
decoded = tokenizer.decode(encoded.ids)
print(f"Decoded: '{decoded}'")
