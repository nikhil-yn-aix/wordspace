# 05: cbow and skip-gram training pairs, plus the global co-occurrence matrix glove starts from.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import plot_style as style

style.apply_style()

SENTENCE = "the dog eats food"
WINDOW_SIZE = 2

GLOVE_CORPUS = [
    "the dog chases the cat",
    "the cat chases the mouse",
    "the dog barks at the cat",
    "the cat sleeps on the mat",
    "the dog eats meat",
    "the cat drinks milk",
    "the car drives on the road",
    "the bike rides on the road",
    "the car passes the bike",
]

# grouped on purpose so related words sit next to each other in the heatmap.
MATRIX_ORDER = ["the", "dog", "cat", "mouse", "car", "bike", "chases", "barks", "at", "sleeps",
                "on", "mat", "eats", "meat", "drinks", "milk", "drives", "road", "rides", "passes"]


def context_window(tokens, position):
    start, end = max(0, position - WINDOW_SIZE), min(len(tokens), position + WINDOW_SIZE + 1)
    return [tokens[j] for j in range(start, end) if j != position]


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}\n")


tokens = SENTENCE.split()
print(f'\nsentence: "{SENTENCE}"   window: {WINDOW_SIZE}')

section("cbow training examples")
print(f"{'context':28}target")
print("-" * 60)
for position, target in enumerate(tokens):
    print(f"{', '.join(context_window(tokens, position)):28}{target}")
print("-" * 60)
print("cbow: the context words together predict the target word.")

section("skip-gram training examples")
print(f"{'target':13}context")
print("-" * 60)
for position, target in enumerate(tokens):
    for context in context_window(tokens, position):
        print(f"{target:13}{context}")
print("-" * 60)
print("skip-gram: the target word predicts each context word on its own.")

section("glove: global co-occurrence counts")
sentences = [sentence.split() for sentence in GLOVE_CORPUS]
assert set(MATRIX_ORDER) == {word for sentence in sentences for word in sentence}
index = {word: i for i, word in enumerate(MATRIX_ORDER)}
matrix = np.zeros((len(MATRIX_ORDER), len(MATRIX_ORDER)), dtype=int)
for sentence in sentences:
    for position, word in enumerate(sentence):
        for context in context_window(sentence, position):
            matrix[index[word], index[context]] += 1

print(f"{len(GLOVE_CORPUS)} sentences, {len(MATRIX_ORDER)} words, window {WINDOW_SIZE}.")
print(f"the matrix has {matrix.shape[0]} x {matrix.shape[1]} counts, e.g. cat/the = {matrix[index['cat'], index['the']]}.\n")
print("""cbow and skip-gram slide a window over text and learn from one local example at a time.
glove first counts every co-occurrence across the whole corpus, then fits vectors
so that dot products match the log of those global counts.""")

fig, ax = plt.subplots(figsize=(13, 11))
cmap = LinearSegmentedColormap.from_list("paper_to_indigo", ["#F8FAFC", "#C7D2FE", style.INDIGO, "#1E1B4B"])
ax.imshow(matrix, cmap=cmap, vmin=0)
ax.set_xticks(range(len(MATRIX_ORDER)))
ax.set_yticks(range(len(MATRIX_ORDER)))
ax.set_xticklabels(MATRIX_ORDER, rotation=45, ha="right", fontsize=15)
ax.set_yticklabels(MATRIX_ORDER, fontsize=15)
ax.set_xticks(np.arange(-0.5, len(MATRIX_ORDER)), minor=True)
ax.set_yticks(np.arange(-0.5, len(MATRIX_ORDER)), minor=True)
ax.grid(which="minor", color="white", linewidth=2)
ax.grid(which="major", visible=False)
ax.tick_params(which="both", length=0)
ax.spines[:].set_visible(False)
for row, column in zip(*np.nonzero(matrix)):
    ax.text(column, row, matrix[row, column], ha="center", va="center", fontsize=15, fontweight="bold",
            color="white" if matrix[row, column] > matrix.max() * 0.45 else style.INK)
ax.set_xlabel("context word")
ax.set_ylabel("target word")
ax.set_title(f"word x word co-occurrence counts (window = {WINDOW_SIZE}, {len(GLOVE_CORPUS)} sentences)", pad=14)
style.save_plot(fig, "05_cooccurrence_matrix.png")
