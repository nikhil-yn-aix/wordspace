# word embeddings demos

eight small python scripts for a class on word embeddings and semantic search. one idea per file, each short enough to read top to bottom while you talk.

## run it

you need [uv](https://docs.astral.sh/uv/) and internet on the first run.

```
uv sync
uv run python run_all.py              # everything, in lesson order
uv run python 02_tfidf_retrieval.py   # or one at a time
```

models and data download themselves the first time they are needed.

## the demos

| script | what it shows | output |
|---|---|---|
| `01_representation.py` | numbers are not meaning | terminal |
| `02_tfidf_retrieval.py` | tf-idf built from scratch | `plots/02_tfidf_retrieval.png` |
| `03_tfidf_failure.py` | tf-idf misses words with the same meaning | `plots/03_tfidf_failure.png` |
| `04_train_word_embeddings.py` | skip-gram trained from scratch | `plots/04_embeddings_before_after.png` |
| `05_cbow_skipgram.py` | cbow and skip-gram pairs, glove co-occurrence | `plots/05_cooccurrence_matrix.png` |
| `06_pretrained_embeddings.py` | real glove vectors, neighbours, analogy | `plots/06_embedding_geometry.png` |
| `07_contextual_embeddings.py` | the word "bank" in two contexts with bert | `plots/07_contextual_embeddings.png` |
| `08_semantic_search.py` | tf-idf vs dense retrieval | `plots/08_retrieval_comparison.png` |

`plot_style.py` only holds the shared colors and fonts. `run_all.py` runs the eight scripts in order.

## downloads

| what | size | used by | cached in |
|---|---|---|---|
| glove 6b 50d (stanford nlp) | ~80 mb | 06 | `data/` |
| bert-base-uncased (hugging face) | ~440 mb | 07 | `~/.cache/huggingface` |
| all-MiniLM-L6-v2 (hugging face) | ~90 mb | 08 | `~/.cache/huggingface` |

reruns reuse the cache and download nothing.

## how long it takes

after the downloads, about 2 minutes for everything on a normal laptop cpu. 04 is the slowest at about 40 seconds, 06, 07 and 08 take 15 to 20 seconds each, the rest a few seconds. training in 04 is seeded, but tiny numeric differences between machines are normal.
