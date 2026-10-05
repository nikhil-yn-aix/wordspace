# 08: semantic search. same corpus, same queries, two retrievers: tf-idf and dense embeddings.
# the queries use everyday words, not the wording of the documents.

import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as transformers_logging

import plot_style as style

style.apply_style()
transformers_logging.set_verbosity_error()
transformers_logging.disable_progress_bar()

CORPUS = [
    "Dijkstra's algorithm finds shortest paths in weighted graphs with non-negative edges.",
    "A* search uses heuristics to guide pathfinding toward the goal efficiently.",
    "Breadth-first search explores all neighbors level by level in unweighted graphs.",
    "Depth-first search traverses as far as possible along each branch before backtracking.",
    "Bellman-Ford algorithm handles negative edge weights and detects negative cycles.",
    "Floyd-Warshall computes all-pairs shortest paths in dense graphs.",
    "B-tree indexes maintain sorted data for efficient range queries and inserts.",
    "Hash indexes provide O(1) lookup for exact-match queries on keys.",
    "LSM trees optimize write-heavy workloads by batching writes sequentially.",
    "Database query optimizers choose execution plans using cost-based estimation.",
    "ACID transactions guarantee atomicity, consistency, isolation, durability.",
    "MVCC enables concurrent reads and writes without locking conflicts.",
    "Process schedulers allocate CPU time using priority queues and time slices.",
    "Virtual memory maps virtual addresses to physical frames via page tables.",
    "Page replacement algorithms like LRU evict least recently used pages.",
    "File systems organize data using inodes, directories, and block allocation.",
    "TCP ensures reliable ordered delivery through sequence numbers and acknowledgments.",
    "Congestion control algorithms like CUBIC adapt sending rates to network capacity.",
    "Routing protocols like OSPF and BGP exchange topology information.",
    "DNS resolves domain names to IP addresses through hierarchical delegation.",
    "Neural networks learn representations through gradient descent on loss functions.",
    "Backpropagation computes gradients efficiently using the chain rule.",
    "Convolutional layers extract spatial hierarchies in image data.",
    "Transformer attention mechanisms model long-range dependencies in sequences.",
    "Language models predict next tokens using autoregressive decoding.",
    "Embedding layers map discrete tokens to continuous vector spaces.",
    "Fine-tuning adapts pretrained models to downstream tasks with small data.",
    "Vector databases index high-dimensional embeddings for similarity search.",
    "HNSW graphs enable approximate nearest neighbor search at scale.",
    "Retrieval-augmented generation grounds LLM outputs in external knowledge.",
]

# (query, index of the document a person would call the best answer, chosen by hand)
QUERIES = [
    ("How can I find the shortest route between two nodes?", 0),
    ("Speed up lookups of rows in a table", 7),
    ("What happens when the computer runs out of RAM?", 13),
    ("Keep a network from getting overloaded", 17),
    ("Teach a model by following the slope of its error", 20),
]

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3


def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())


def tfidf_vector(tokens, vocabulary, idf):
    counts = Counter(tokens)
    return np.array([counts[word] / max(len(tokens), 1) for word in vocabulary]) * idf


def cosine_similarity(a, b):
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm else 0.0


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def print_top(name, scores):
    print(f"\n{'-' * 60}\n{name}\n{'-' * 60}")
    for place, doc_index in enumerate(np.argsort(-scores)[:TOP_K], 1):
        print(f"{place}. d{doc_index + 1}  score = {scores[doc_index]:.3f}\n   {CORPUS[doc_index]}")


tokenized_documents = [tokenize(document) for document in CORPUS]
vocabulary = sorted({word for tokens in tokenized_documents for word in tokens})
document_frequency = np.array([sum(word in tokens for tokens in tokenized_documents) for word in vocabulary])
idf = np.log(len(CORPUS) / (document_frequency + 1)) + 1
tfidf_documents = np.array([tfidf_vector(tokens, vocabulary, idf) for tokens in tokenized_documents])

section("semantic search")
print(f"\ncorpus : {len(CORPUS)} engineering snippets")
model = SentenceTransformer(MODEL_NAME)
dense_documents = model.encode(CORPUS, normalize_embeddings=True)
print(f"dense  : {MODEL_NAME}, {dense_documents.shape[1]}-dimensional vectors")

results = []
for query, best in QUERIES:
    query_tokens = [word for word in tokenize(query) if word in vocabulary]
    query_vector = tfidf_vector(query_tokens, vocabulary, idf)
    tfidf_scores = np.array([cosine_similarity(query_vector, vector) for vector in tfidf_documents])
    dense_scores = dense_documents @ model.encode([query], normalize_embeddings=True)[0]
    results.append({"query": query, "best": best, "tfidf": tfidf_scores, "dense": dense_scores})

    section(f'query: "{query}"')
    print_top("tf-idf", tfidf_scores)
    print_top("dense embeddings", dense_scores)


def place_of_best(scores, best):
    return int(np.where(np.argsort(-scores) == best)[0][0]) + 1


section("where did each method rank the best answer?")
print(f"\n{'query':54}{'tf-idf':>8}{'dense':>8}")
print("-" * 70)
for result in results:
    print(f"{result['query'][:52].lower():54}{place_of_best(result['tfidf'], result['best']):>8}"
          f"{place_of_best(result['dense'], result['best']):>8}")
print("\nbest answers were picked by hand before running. five queries is a demo, not a benchmark.")


def draw_top(ax, scores, best, title, color):
    top = np.argsort(-scores)[:TOP_K]
    bars = ax.barh(range(TOP_K), scores[top], height=0.6, color=[color if i == best else style.SLATE for i in top])
    ax.set_yticks(range(TOP_K))
    ax.set_yticklabels([f"d{i + 1}  {CORPUS[i][:30].lower()}..." for i in top], fontsize=15)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.8)
    ax.grid(axis="y", visible=False)
    ax.set_title(title, fontsize=20, color=color)
    ax.set_xlabel("cosine similarity")
    for bar, score in zip(bars, scores[top]):
        ax.text(score + 0.01, bar.get_y() + bar.get_height() / 2, f"{score:.2f}", va="center", fontsize=16, fontweight="bold")


fig = plt.figure(figsize=(20, 14))
grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.5], hspace=0.7, wspace=0.75)
hero = results[0]
draw_top(fig.add_subplot(grid[0, 0]), hero["tfidf"], hero["best"], "tf-idf top 3", style.ROSE)
draw_top(fig.add_subplot(grid[0, 1]), hero["dense"], hero["best"], "dense embeddings top 3", style.TEAL)
fig.text(0.125, 0.925, f'query: "{hero["query"].lower()}"   (green bar = best answer)', fontsize=19, fontweight="bold")

rank_ax = fig.add_subplot(grid[1, :])
for row, result in enumerate(results):
    tfidf_place = place_of_best(result["tfidf"], result["best"])
    dense_place = place_of_best(result["dense"], result["best"])
    rank_ax.plot([tfidf_place, dense_place], [row, row], color=style.SLATE, linewidth=4, zorder=1)
    for place, color in [(tfidf_place, style.ROSE), (dense_place, style.TEAL)]:
        rank_ax.scatter(place, row, s=1100, color=color, zorder=3, edgecolors="white", linewidth=2)
        rank_ax.text(place, row, str(place), ha="center", va="center", fontsize=16, fontweight="bold", color="white", zorder=4)
rank_ax.set_yticks(range(len(results)))
rank_ax.set_yticklabels([r["query"].lower() for r in results], fontsize=16)
rank_ax.invert_yaxis()
rank_ax.set_xlim(-0.5, len(CORPUS) + 0.5)
rank_ax.set_xlabel(f"rank of the best answer (1 = best, out of {len(CORPUS)} documents)", fontsize=17)
rank_ax.set_title("where each method ranked the best answer", fontsize=20)
rank_ax.grid(axis="y", visible=False)
rank_ax.legend(handles=[Patch(color=style.ROSE, label="tf-idf"), Patch(color=style.TEAL, label="dense embeddings")],
               loc="upper right", fontsize=17)

fig.suptitle("lexical vs semantic retrieval on the same 30 documents", fontsize=26, fontweight="bold", x=0.125, ha="left", y=0.985)
style.save_plot(fig, "08_retrieval_comparison.png")
