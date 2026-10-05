# 07: static vs contextual embeddings. one word ("bank"), financial vs river sentences.
# static: bert's input embedding table has exactly one vector for "bank".
# contextual: bert's final layer gives "bank" a different vector in every sentence.

import matplotlib.pyplot as plt
import numpy as np
import torch
from adjustText import adjust_text
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as transformers_logging

import plot_style as style

style.apply_style()
transformers_logging.set_verbosity_error()
transformers_logging.disable_progress_bar()

MODEL_NAME = "bert-base-uncased"
TARGET_WORD = "bank"

SENTENCES = [
    ("financial", "I deposited money in the bank."),
    ("financial", "The bank approved my loan application."),
    ("financial", "She opened a savings account at the bank."),
    ("financial", "The bank raised its interest rates."),
    ("river", "I sat on the bank of the river."),
    ("river", "Fishermen stood along the muddy bank."),
    ("river", "The river overflowed its bank after the storm."),
    ("river", "We walked along the grassy bank of the stream."),
]


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def contextual_vector(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", return_offsets_mapping=True)
    offsets = inputs.pop("offset_mapping")[0].tolist()

    # find "bank" by its character span instead of guessing a token index.
    start = sentence.lower().index(TARGET_WORD)
    position = offsets.index([start, start + len(TARGET_WORD)])
    with torch.no_grad():
        return model(**inputs).last_hidden_state[0, position].numpy()


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}\n")


section("static vs contextual embeddings")
print(f"model: {MODEL_NAME} (downloaded once, then cached by hugging face)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).eval()

static_vector = model.embeddings.word_embeddings.weight[tokenizer.convert_tokens_to_ids(TARGET_WORD)].detach().numpy()
vectors = np.array([contextual_vector(sentence) for _, sentence in SENTENCES])
labels = [f"{kind} {sum(k == kind for k, _ in SENTENCES[:n]) }" for n, (kind, _) in enumerate(SENTENCES, 1)]

print(f"\nstatic: 'bank' is one vector of {static_vector.shape[0]} numbers, identical in every sentence.")
print("\ncontextual: the final-layer vector at the 'bank' token")
for label, (_, sentence) in zip(labels, SENTENCES):
    print(f"  {label:12} {sentence}")

similarity = np.array([[cosine_similarity(a, b) for b in vectors] for a in vectors])
half = len(SENTENCES) // 2
same_meaning = np.mean([similarity[i, j] for i in range(len(vectors)) for j in range(len(vectors))
                        if i != j and (i < half) == (j < half)])
different_meaning = similarity[:half, half:].mean()
pair = similarity[0, half]

print("\ncosine similarity")
print("  " + "-" * 44)
print(f"  financial 1 vs river 1       : {pair:.2f}")
print(f"  average, same meaning        : {same_meaning:.2f}")
print(f"  average, different meaning   : {different_meaning:.2f}")
print("\nthe vector for a word is computed from the whole sentence, so the same word gets different")
print("vectors in different contexts. a static table cannot do that.")

fig, (map_ax, matrix_ax) = plt.subplots(1, 2, figsize=(20, 9), gridspec_kw={"width_ratios": [1.15, 1]})

coordinates = PCA(n_components=2, random_state=42).fit_transform(vectors)
colors = [style.INDIGO if kind == "financial" else style.TEAL for kind, _ in SENTENCES]
map_ax.scatter(coordinates[:, 0], coordinates[:, 1], s=260, c=colors, edgecolors="white", linewidth=2, zorder=3)
texts = [map_ax.text(x, y, label, fontsize=16, fontweight="bold", color=color)
         for label, (x, y), color in zip(labels, coordinates, colors)]
adjust_text(texts, ax=map_ax, expand=(1.4, 1.6), arrowprops=dict(arrowstyle="-", color=style.SLATE, lw=0.8))
map_ax.text(0.02, 0.03, "a static table would draw one point here for all 8 sentences.",
            transform=map_ax.transAxes, fontsize=16, color=style.ROSE, fontweight="bold")
map_ax.set_title('contextual "bank" vectors (pca of the final layer)')
map_ax.set_xlabel("pca component 1")
map_ax.set_ylabel("pca component 2")
map_ax.margins(0.15)

image = matrix_ax.imshow(similarity, cmap="RdYlBu", vmin=similarity.min(), vmax=1)
matrix_ax.set_xticks(range(len(labels)))
matrix_ax.set_yticks(range(len(labels)))
matrix_ax.set_xticklabels(labels, rotation=45, ha="right")
matrix_ax.set_yticklabels(labels)
matrix_ax.grid(False)
matrix_ax.tick_params(length=0)
matrix_ax.spines[:].set_visible(False)
for i in range(len(labels)):
    for j in range(len(labels)):
        matrix_ax.text(j, i, f"{similarity[i, j]:.2f}", ha="center", va="center", fontsize=14, fontweight="bold")
matrix_ax.set_title("cosine similarity between the 8 'bank' vectors")
fig.colorbar(image, ax=matrix_ax, shrink=0.8, label="cosine similarity")

fig.suptitle(f"same word, different vectors. financial 1 vs river 1: cosine = {pair:.2f}",
             fontsize=22, fontweight="bold", x=0.06, ha="left", y=1.0)
fig.tight_layout()
style.save_plot(fig, "07_contextual_embeddings.png")
