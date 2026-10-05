# 01: numbers are not meaning. any numeric encoding works, but only a learned one captures similarity.

import numpy as np


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}\n")


section("text as numbers")
word = "cat"
for position, character in enumerate(word):
    print(f"  [{position}] '{character}'  codepoint {ord(character)}  utf-8 byte {character.encode('utf-8')[0]}")
print(f"\n  '{word}' -> {[ord(c) for c in word]}")

section("do the numbers capture meaning?")
words = ["cat", "feline", "dog", "car"]
width = max(len(w) for w in words)
vectors = {w: np.array([ord(c) for c in w] + [0] * (width - len(w)), dtype=float) for w in words}

for w, vector in vectors.items():
    print(f"  {w:7} {vector.astype(int).tolist()}")

print("\n  cosine similarity of the codepoint vectors")
print("  " + "-" * 30)
for a, b in [("cat", "feline"), ("cat", "dog"), ("cat", "car"), ("dog", "car")]:
    print(f"  {a:4} <-> {b:7} {cosine_similarity(vectors[a], vectors[b]):.3f}")

print("""
  cat and feline mean the same thing, yet cat and car (two shared letters)
  scores higher. every pair looks similar because all codepoints are positive.
  a numeric encoding is not a meaningful representation. meaning has to be learned.
""")
