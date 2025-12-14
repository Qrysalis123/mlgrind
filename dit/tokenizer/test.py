"""Test trained tokenizer on poem.txt"""

from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("bpe-32k-uncased.json")

with open("../poem.txt", "r") as f:
    text = f.read()

encoded = tokenizer.encode(text)

print("=== Original Text ===")
print(text)

print("\n=== Tokens ===")
print(encoded.tokens)

print(f"\n=== Stats ===")
print(f"Characters: {len(text)}")
print(f"Tokens: {len(encoded.ids)}")
print(f"Compression: {len(text) / len(encoded.ids):.2f} chars/token")

print("\n=== Decode Test ===")
decoded = tokenizer.decode(encoded.ids)
print(decoded)
print(f"\nRoundtrip match: {text.lower() == decoded}")
