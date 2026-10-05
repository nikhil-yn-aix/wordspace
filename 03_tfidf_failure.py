# 03: tf-idf is lexical. relevant documents that use different words score zero.

import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import plot_style as style

style.apply_style()

CORPUS = [
    "Machine learning algorithms learn patterns from data automatically.",
    "Neural networks adjust weights through gradient descent optimization.",
    "Deep learning models require large labeled datasets for training.",
    "Algorithms that learn patterns from data improve with experience.",
    "Statistical models discover structure in observational data.",
    "Database systems store and retrieve structured information efficiently.",
    "B-tree indexes accelerate range queries on sorted column values.",
    "Hash tables provide constant-time key-value lookups.",
    "Computer vision systems classify images using convolutional layers.",
    "Natural language processing transforms text into numerical representations.",
]

# relevance judged by hand, 0-based document indexes
RELEVANT_DOCS = {
    "machine learning": {0, 1, 2, 3},
    "algorithms learn from data": {0, 1, 2, 3, 4},
    "data structure optimization": {6, 7},
}


def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())


def tfidf_vector(tokens, vocabulary, idf):
    counts = Counter(tokens)
    return np.array([counts[word] / len(tokens) for word in vocabulary]) * idf


def cosine_similarity(a, b):
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm else 0.0


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


tokenized_documents = [tokenize(document) for document in CORPUS]
vocabulary = sorted({word for tokens in tokenized_documents for word in tokens})
document_frequency = np.array([sum(word in tokens for tokens in tokenized_documents) for word in vocabulary])
idf = np.log(len(CORPUS) / (document_frequency + 1)) + 1
document_vectors = np.array([tfidf_vector(tokens, vocabulary, idf) for tokens in tokenized_documents])

section("tf-idf semantic failure")
results = {}
for query, relevant in RELEVANT_DOCS.items():
    query_tokens = [word for word in tokenize(query) if word in vocabulary]
    query_vector = tfidf_vector(query_tokens, vocabulary, idf)
    scores = [cosine_similarity(query_vector, vector) for vector in document_vectors]
    results[query] = scores

    print(f'\nquery: "{query}"')
    print(f"  {'rank':5}{'doc':5}{'score':>7}   why")
    print("  " + "-" * 62)
    for rank, doc_index in enumerate(np.argsort(-np.array(scores))[:6], 1):
        shared = sorted(set(query_tokens) & set(tokenized_documents[doc_index]))
        if shared:
            why = "shares: " + ", ".join(shared)
        elif doc_index in relevant:
            why = "relevant, but no shared words (missed)"
        else:
            why = "no shared words"
        print(f"  {rank:<5}D{doc_index + 1:<4}{scores[doc_index]:>7.3f}   {why}")


fig, axes = plt.subplots(1, len(results), figsize=(21, 7.5))
for ax, (query, scores) in zip(axes, results.items()):
    relevant = RELEVANT_DOCS[query]
    order = np.argsort(-np.array(scores), kind="stable")
    colors = [(style.TEAL if scores[i] > 0 else style.ROSE) if i in relevant else style.SLATE for i in order]

    bars = ax.barh(range(len(order)), [scores[i] for i in order], color=colors, height=0.66)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"d{i + 1}  {' '.join(CORPUS[i].split()[:4]).lower()}" for i in order], fontsize=15)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("cosine similarity")
    ax.set_title(f'"{query}"', fontsize=17, pad=10)
    for bar, i in zip(bars, order):
        missed = i in relevant and scores[i] == 0
        ax.text(scores[i] + 0.01, bar.get_y() + bar.get_height() / 2,
                "0.00  missed" if missed else f"{scores[i]:.2f}",
                va="center", fontsize=14, fontweight="bold", color=style.ROSE if missed else style.INK)

fig.legend(handles=[
    Patch(color=style.TEAL, label="relevant, found (shares words with query)"),
    Patch(color=style.ROSE, label="relevant, missed (score 0, different words)"),
    Patch(color=style.SLATE, label="not relevant"),
], loc="lower center", ncol=3, fontsize=17, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("tf-idf is lexical: relevant documents with different words score zero",
             fontsize=22, fontweight="bold", x=0.01, ha="left", y=1.02)
fig.tight_layout()
style.save_plot(fig, "03_tfidf_failure.png")

print("""
tf-idf matches words, not meanings. "machine learning" misses the neural network
and "algorithms that learn" documents, and "data structure optimization" ranks a
statistics document above b-trees and hash tables because it shares "structure".
""")
