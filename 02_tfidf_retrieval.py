# 02: tf-idf retrieval from scratch. tf, df, idf, vectors, cosine similarity, ranking.

import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import plot_style as style

style.apply_style()

CORPUS = [
    "Neural networks learn representations from data through gradient descent.",
    "Neural networks learn patterns from training data using backpropagation.",
    "Deep learning models require large datasets for effective training.",
    "Database indexes accelerate information retrieval in large systems.",
    "B-tree indexes optimize range queries on ordered datasets.",
    "Hash indexes provide constant-time lookup for exact matches.",
    "Operating system schedulers allocate CPU time to running processes.",
    "Virtual memory enables processes to use more memory than physically available.",
    "Network protocols like TCP ensure reliable data transmission.",
    "Routing algorithms determine optimal paths through network topologies.",
]

QUERIES = [
    "neural networks learn",
    "database index optimization",
    "operating system memory management",
]


def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())


def term_frequency(tokens, vocabulary):
    counts = Counter(tokens)
    return np.array([counts[word] / len(tokens) for word in vocabulary])


def tfidf_vector(tokens, vocabulary, idf):
    return term_frequency(tokens, vocabulary) * idf


def cosine_similarity(a, b):
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm else 0.0


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


tokenized_documents = [tokenize(document) for document in CORPUS]
vocabulary = sorted({word for tokens in tokenized_documents for word in tokens})
document_frequency = np.array([sum(word in tokens for tokens in tokenized_documents) for word in vocabulary])

# idf downweights words that appear in many documents.
idf = np.log(len(CORPUS) / (document_frequency + 1)) + 1
document_vectors = np.array([tfidf_vector(tokens, vocabulary, idf) for tokens in tokenized_documents])

section("tf-idf from scratch")
print(f"\n  {len(CORPUS)} documents, {len(vocabulary)} words in the vocabulary")
print(f"  document vectors: {document_vectors.shape[0]} x {document_vectors.shape[1]} (documents x words)\n")
print(f"  {'word':14}{'df':>4}{'idf':>8}")
print("  " + "-" * 26)
for word, df, word_idf in sorted(zip(vocabulary, document_frequency, idf), key=lambda row: -row[1])[:8]:
    print(f"  {word:14}{df:>4}{word_idf:>8.2f}")
print("  ... (most common words first)")

scores_by_query = {}
for query in QUERIES:
    query_tokens = [word for word in tokenize(query) if word in vocabulary]
    query_vector = tfidf_vector(query_tokens, vocabulary, idf)
    scores = np.array([cosine_similarity(query_vector, vector) for vector in document_vectors])
    scores_by_query[query] = scores

    section(f'query: "{query}"')
    print("\n  non-zero query weights:", {vocabulary[i]: round(float(query_vector[i]), 3) for i in np.nonzero(query_vector)[0]})
    print(f"\n  {'rank':5}{'doc':5}{'score':>7}   text")
    print("  " + "-" * 56)
    for rank, doc_index in enumerate(np.argsort(-scores)[:3], 1):
        print(f"  {rank:<5}D{doc_index + 1:<4}{scores[doc_index]:>7.3f}   {CORPUS[doc_index][:50]}...")


fig, (heat_ax, idf_ax) = plt.subplots(1, 2, figsize=(24, 10.5), gridspec_kw={"width_ratios": [1.5, 1]})

score_matrix = np.array([scores_by_query[q] for q in QUERIES]).T
cmap = LinearSegmentedColormap.from_list("paper_to_indigo", ["#F8FAFC", "#C7D2FE", style.INDIGO, "#1E1B4B"])
heat_ax.imshow(score_matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
heat_ax.set_xticks(range(len(QUERIES)))
heat_ax.set_xticklabels([q.replace(" ", "\n") for q in QUERIES], fontsize=16)
heat_ax.xaxis.tick_top()
heat_ax.set_yticks(range(len(CORPUS)))
heat_ax.set_yticklabels([f"d{i + 1}  {text[:38].lower()}..." for i, text in enumerate(CORPUS)], fontsize=15)
heat_ax.set_xticks(np.arange(-0.5, len(QUERIES)), minor=True)
heat_ax.set_yticks(np.arange(-0.5, len(CORPUS)), minor=True)
heat_ax.grid(which="minor", color="white", linewidth=3)
heat_ax.grid(which="major", visible=False)
heat_ax.tick_params(which="both", length=0)
heat_ax.spines[:].set_visible(False)
for row, column in zip(*np.nonzero(score_matrix)):
    value = score_matrix[row, column]
    heat_ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=17,
                 fontweight="bold", color="white" if value > 0.4 else style.INK)
heat_ax.set_title("cosine score per document and query (blank = 0)", pad=50)

by_idf = sorted(zip(vocabulary, idf), key=lambda pair: pair[1])
shown = by_idf[:6] + by_idf[-6:]
idf_ax.barh(range(12), [value for _, value in shown], color=[style.SLATE] * 6 + [style.INDIGO] * 6, height=0.65)
idf_ax.set_yticks(range(12))
idf_ax.set_yticklabels([word for word, _ in shown], fontsize=16)
idf_ax.invert_yaxis()
idf_ax.grid(axis="y", visible=False)
for position, (_, value) in enumerate(shown):
    idf_ax.text(value + 0.03, position, f"{value:.2f}", va="center", fontsize=15, fontweight="bold")
idf_ax.set_xlim(0, 3)
idf_ax.set_xlabel("idf = log(n / (df + 1)) + 1")
idf_ax.set_title("lowest vs highest idf words")

fig.suptitle("tf-idf retrieval: only documents that share query words score",
             fontsize=22, fontweight="bold", x=0.01, ha="left", y=1.02)
fig.tight_layout()
style.save_plot(fig, "02_tfidf_retrieval.png")
