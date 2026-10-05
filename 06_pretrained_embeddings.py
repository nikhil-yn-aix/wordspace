# 06: real pretrained glove vectors. inspect them, find neighbours, look at the geometry.

import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from sklearn.decomposition import PCA

import plot_style as style

style.apply_style()

DATA_DIR = Path(__file__).parent / "data"
GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_FILE = DATA_DIR / "glove.6B.50d.txt"
MAX_WORDS = 100_000

GROUPS = {
    "animals": ["dog", "cat", "mouse", "lion", "tiger", "elephant"],
    "vehicles": ["car", "bike", "truck", "bus", "train", "plane"],
    "food": ["apple", "bread", "meat", "milk", "cheese", "fish"],
    "people": ["king", "queen", "man", "woman", "boy", "girl"],
    "actions": ["run", "walk", "jump", "eat", "sleep", "drive"],
    "colors": ["red", "blue", "green", "yellow", "black", "white"],
}


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}\n")


def download_glove_if_missing():
    if GLOVE_FILE.exists():
        print(f"using cached vectors: {GLOVE_FILE}")
        return
    DATA_DIR.mkdir(exist_ok=True)
    archive = DATA_DIR / "glove.6B.zip"
    print("downloading glove 6b from stanford nlp (~80 mb, one time only)...")
    urllib.request.urlretrieve(GLOVE_URL, archive)
    with zipfile.ZipFile(archive) as zipped:
        zipped.extract(GLOVE_FILE.name, DATA_DIR)
    archive.unlink()
    print(f"saved to {GLOVE_FILE}")


def load_glove():
    words, vectors = [], []
    with open(GLOVE_FILE, encoding="utf-8") as file:
        for line, text in zip(range(MAX_WORDS), file):
            word, *numbers = text.split()
            words.append(word)
            vectors.append(numbers)
    return {word: i for i, word in enumerate(words)}, np.array(vectors, dtype=np.float32)


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def nearest(vector, k, exclude=()):
    similarities = embeddings @ vector / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(vector))
    ranked = [(words[i], float(similarities[i])) for i in np.argsort(-similarities)]
    return [(word, score) for word, score in ranked if word not in exclude][:k]


section("pretrained word embeddings (glove 6b, 50d)")
download_glove_if_missing()
word_to_index, embeddings = load_glove()
words = list(word_to_index)

print(f"\n  words in the table : {len(words):,}")
print(f"  embedding size     : {embeddings.shape[1]}")
print(f"  vector for 'dog'   : {np.round(embeddings[word_to_index['dog']][:6], 2)} ... ({embeddings.shape[1]} numbers)")

print("\n  cosine similarity")
print("  " + "-" * 28)
for a, b in [("dog", "cat"), ("dog", "car"), ("car", "bike"), ("king", "queen"), ("hot", "cold")]:
    print(f"  {a:6} <-> {b:6} {cosine_similarity(embeddings[word_to_index[a]], embeddings[word_to_index[b]]):.2f}")

print("\n  nearest neighbours")
print("  " + "-" * 28)
for word in ["dog", "car", "king", "red"]:
    neighbours = nearest(embeddings[word_to_index[word]], 5, exclude={word})
    print(f"  {word:5} -> " + ", ".join(f"{w} ({s:.2f})" for w, s in neighbours))

analogy_vector = sum(sign * embeddings[word_to_index[w]] for sign, w in [(1, "king"), (-1, "man"), (1, "woman")])
analogy = nearest(analogy_vector, 6, exclude={"king", "man", "woman"})
print("\n  king - man + woman")
print("  " + "-" * 28)
print("  " + ", ".join(f"{w} ({s:.2f})" for w, s in analogy))
print("  this works well here, but analogies are not guaranteed for every model or every pair.")

plotted_words = [word for members in GROUPS.values() for word in members]
pca = PCA(n_components=2, random_state=42)
coordinates = dict(zip(plotted_words, pca.fit_transform(embeddings[[word_to_index[w] for w in plotted_words]])))
print(f"\n  pca keeps {pca.explained_variance_ratio_.sum():.0%} of the variance in 2d, so distances are approximate.")

fig, (map_ax, analogy_ax) = plt.subplots(1, 2, figsize=(21, 10), gridspec_kw={"width_ratios": [1.9, 1]})

texts = []
for (group, members), color in zip(GROUPS.items(), style.GROUP_COLORS):
    points = np.array([coordinates[w] for w in members])
    map_ax.scatter(points[:, 0], points[:, 1], s=240, color=color, label=group, edgecolors="white", linewidth=2, zorder=3)
    texts += [map_ax.text(x, y, word, fontsize=17, fontweight="bold", color=color, zorder=4)
              for word, (x, y) in zip(members, points)]
adjust_text(texts, ax=map_ax, expand=(1.4, 1.6), arrowprops=dict(arrowstyle="-", color=style.SLATE, lw=0.8))
map_ax.set_title("glove 50d: 36 words projected to 2d with pca")
map_ax.set_xlabel(f"pca component 1 ({pca.explained_variance_ratio_[0]:.0%} of variance)")
map_ax.set_ylabel(f"pca component 2 ({pca.explained_variance_ratio_[1]:.0%} of variance)")
map_ax.legend(loc="best", ncol=2, markerscale=0.8)
map_ax.margins(0.08)

# the analogy is shown as a ranking, so nothing is lost to the 2d projection.
bars = analogy_ax.barh(range(len(analogy)), [score for _, score in analogy], height=0.62,
                       color=[style.ROSE if word == "queen" else style.SLATE for word, _ in analogy])
analogy_ax.set_yticks(range(len(analogy)))
analogy_ax.set_yticklabels([word for word, _ in analogy], fontsize=15)
analogy_ax.invert_yaxis()
analogy_ax.set_xlim(0, 1)
analogy_ax.grid(axis="y", visible=False)
for bar, (_, score) in zip(bars, analogy):
    analogy_ax.text(score + 0.01, bar.get_y() + bar.get_height() / 2, f"{score:.2f}", va="center", fontsize=16, fontweight="bold")
analogy_ax.set_title("king - man + woman = ?")
analogy_ax.set_xlabel("cosine similarity to the result vector")

fig.suptitle("a pretrained embedding space has geometry", fontsize=22, fontweight="bold", x=0.06, ha="left", y=0.99)
fig.tight_layout()
style.save_plot(fig, "06_embedding_geometry.png")
