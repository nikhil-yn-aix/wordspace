# 04: train word embeddings from scratch (skip-gram with negative sampling), in plain numpy.

import math
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from sklearn.decomposition import PCA

import plot_style as style

style.apply_style()
np.random.seed(42)

GROUPS = {
    "animals": ["dog", "cat", "horse"],
    "vehicles": ["car", "bike", "truck"],
    "food": ["meat", "milk", "bread"],
}

# words in the same group share templates, so they share contexts.
TEMPLATES = {
    "animals": ["the {} is an animal", "the {} eats some food", "the {} sleeps at home", "a {} lives with people"],
    "vehicles": ["the {} is a vehicle", "the {} drives on the road", "the {} needs some fuel", "a {} carries people"],
    "food": ["the {} is fresh food", "we eat the {} at home", "the {} is in the kitchen", "a {} tastes good"],
}

CORPUS = [template.format(word) for group, words in GROUPS.items() for word in words for template in TEMPLATES[group]]

WINDOW_SIZE = 2
EMBEDDING_DIM = 50
EPOCHS = 200
LEARNING_RATE = 0.05
NEGATIVE_SAMPLES = 5

TRACKED_PAIRS = [("dog", "cat"), ("dog", "car"), ("car", "bike"), ("meat", "milk"), ("dog", "horse")]


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


sentences = [sentence.split() for sentence in CORPUS]
counts = Counter(word for sentence in sentences for word in sentence)
vocabulary = [word for word, _ in counts.most_common()]
word_to_index = {word: i for i, word in enumerate(vocabulary)}
vocabulary_size = len(vocabulary)

pairs = []
for sentence in sentences:
    for i, center in enumerate(sentence):
        for j in range(max(0, i - WINDOW_SIZE), min(len(sentence), i + WINDOW_SIZE + 1)):
            if i != j:
                pairs.append((word_to_index[center], word_to_index[sentence[j]]))

# negative samples are drawn from word frequency ** 0.75, as in word2vec.
sampling_weights = np.array([counts[word] for word in vocabulary], dtype=float) ** 0.75
sampling_weights /= sampling_weights.sum()

center_vectors = np.random.uniform(-0.5 / EMBEDDING_DIM, 0.5 / EMBEDDING_DIM, (vocabulary_size, EMBEDDING_DIM))
context_vectors = np.random.uniform(-0.5 / EMBEDDING_DIM, 0.5 / EMBEDDING_DIM, (vocabulary_size, EMBEDDING_DIM))
initial_vectors = center_vectors.copy()
pairs = np.array(pairs)


def tracked_similarities():
    return [cosine_similarity(center_vectors[word_to_index[a]], center_vectors[word_to_index[b]]) for a, b in TRACKED_PAIRS]


section("training word embeddings (skip-gram, negative sampling)")
print(f"\n  corpus size      : {len(CORPUS)} sentences")
print(f"  vocabulary size  : {vocabulary_size}")
print(f"  embedding size   : {EMBEDDING_DIM}")
print(f"  context window   : {WINDOW_SIZE}")
print(f"  training pairs   : {len(pairs)}")
print(f"  negative samples : {NEGATIVE_SAMPLES}")

before = tracked_similarities()
history = {"epoch": [0], "loss": [], "similarity": [before]}

print("\n  training...")
print("  " + "-" * 40)
for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    for center_index, context_index in pairs[np.random.permutation(len(pairs))]:
        center = center_vectors[center_index]
        center_gradient = np.zeros(EMBEDDING_DIM)

        # one real context word (label 1) and a few random words (label 0).
        negatives = np.random.choice(vocabulary_size, NEGATIVE_SAMPLES, p=sampling_weights)
        targets = [(context_index, 1.0)] + [(n, 0.0) for n in negatives if n != context_index]

        for output_index, label in targets:
            probability = sigmoid(center @ context_vectors[output_index])
            total_loss += -math.log((probability if label else 1 - probability) + 1e-10)

            # gradient of the loss with respect to the score is (probability - label).
            gradient = probability - label
            center_gradient += gradient * context_vectors[output_index]
            context_vectors[output_index] -= LEARNING_RATE * gradient * center

        center_vectors[center_index] -= LEARNING_RATE * center_gradient

    history["loss"].append(total_loss / len(pairs))
    if epoch == 1 or epoch % 50 == 0:
        history["epoch"].append(epoch)
        history["similarity"].append(tracked_similarities())
        print(f"  epoch {epoch:>3}/{EPOCHS}   loss {history['loss'][-1]:.2f}")
print("  " + "-" * 40)
print("  training complete.")

after = tracked_similarities()
print("\n  before training               after training")
print("  " + "-" * 44)
for (a, b), start, end in zip(TRACKED_PAIRS, before, after):
    print(f"  {a:6} <-> {b:6} {start:>6.2f}        {a:6} <-> {b:6} {end:>6.2f}")

print("\n  nearest neighbours after training")
for word in ["dog", "car", "meat"]:
    similarities = {other: cosine_similarity(center_vectors[word_to_index[word]], center_vectors[i])
                    for other, i in word_to_index.items() if other != word}
    top = sorted(similarities, key=similarities.get, reverse=True)[:3]
    print(f"  {word:5} -> " + ", ".join(f"{w} ({similarities[w]:.2f})" for w in top))

print("\n  the corpus is synthetic and templated, so the structure is by design.")
print("  the point is the mechanism: similar contexts lead to similar vectors.")


def draw_map(ax, vectors, title):
    # fit pca on the whole vocabulary, then draw only the grouped words.
    coordinates = PCA(n_components=2, random_state=42).fit_transform(vectors)
    texts = []
    for (group, words), color in zip(GROUPS.items(), style.GROUP_COLORS):
        points = coordinates[[word_to_index[w] for w in words]]
        ax.scatter(points[:, 0], points[:, 1], s=260, color=color, label=group, edgecolors="white", linewidth=2, zorder=3)
        texts += [ax.text(x, y, word, fontsize=17, fontweight="bold", color=color, zorder=4)
                  for word, (x, y) in zip(words, points)]
    adjust_text(texts, ax=ax, expand=(1.6, 1.8), arrowprops=dict(arrowstyle="-", color=style.SLATE, lw=0.8))
    ax.set_title(title)
    ax.set_xlabel("pca component 1")
    ax.set_ylabel("pca component 2")
    ax.margins(0.18)


fig = plt.figure(figsize=(18, 11))
grid = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], hspace=0.38, wspace=0.18)

draw_map(fig.add_subplot(grid[0, 0]), initial_vectors, "before training: random vectors")
after_ax = fig.add_subplot(grid[0, 1])
draw_map(after_ax, center_vectors, "after training: groups emerge")
after_ax.legend(loc="best", markerscale=0.8)

loss_ax = fig.add_subplot(grid[1, 0])
loss_ax.plot(range(1, EPOCHS + 1), history["loss"], color=style.INDIGO, linewidth=3)
loss_ax.set_title("training loss")
loss_ax.set_xlabel("epoch")
loss_ax.set_ylabel("average loss per pair")

similarity_ax = fig.add_subplot(grid[1, 1])
for n, ((a, b), color) in enumerate(zip(TRACKED_PAIRS, [style.TEAL, style.ROSE, style.AMBER, style.SKY, style.VIOLET])):
    similarity_ax.plot(history["epoch"], [row[n] for row in history["similarity"]], linewidth=3,
                       marker="o", markersize=7, color=color, label=f"{a} - {b}")
similarity_ax.axhline(0, color=style.SLATE, linewidth=1)
similarity_ax.set_title("cosine similarity while training")
similarity_ax.set_xlabel("epoch")
similarity_ax.set_ylabel("cosine similarity")
similarity_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=15)

fig.suptitle("skip-gram learns structure from context alone (synthetic templated corpus)",
             fontsize=22, fontweight="bold", x=0.07, ha="left", y=0.97)
style.save_plot(fig, "04_embeddings_before_after.png")
