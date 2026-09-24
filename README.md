# UNREAL

UNREAL turns a frozen LLM into a retriever by training only a small head on top of it.
This repository has the inference code for UNREAL on
[NVIDIA-Nemotron-3.5-Lightning-30B-A3B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16):
encoding a corpus, retrieving chunks for questions, and measuring recall.

## How it works

- **Chunks.** A chunk runs through the frozen backbone up to block 42. The outputs are split into
  7 contiguous parts, and each part is averaged and RMS-normalized: 7 vectors per chunk.
- **Queries.** The question is preceded by its top-5 BM25 passages from the corpus and followed by
  64 learned retrieval tokens (plus one learned soft-prompt token at the start). The outputs at the
  retrieval tokens from the 6 attention blocks (5, 12, 19, 26, 33, 42) are combined with learned
  weights and averaged into 4 query vectors.
- **Scoring.** MaxSim: each query vector takes its best-matching chunk vector, and the score is the
  mean over the 4. The search is exact, over the whole corpus.

Only the head (65 token vectors and 6 layer weights) is trained; the backbone is never changed.

## Setup

Python 3.10+ and one GPU with 80 GB of memory (the backbone up to block 42 takes about 50 GB).

```bash
pip install -r requirements.txt
```

`mamba-ssm` and `causal-conv1d` are CUDA extensions: if pip cannot build them, install their prebuilt wheels
for your PyTorch and CUDA versions. Tested with Python 3.12, PyTorch 2.13, transformers 4.57 and 5.17,
mamba-ssm 2.3 and causal-conv1d 1.6 and 1.7. The backbone is downloaded from Hugging Face on first use;
`--backbone` also takes a local copy.

## Data

Corpus: a JSONL file with one chunk per line. The head was trained with passages of about 100 words;
chunks are cut at 512 tokens.

```json
{"id": "c1", "text": "..."}
```

Queries: a JSONL file with one question per line. `gold_ids` is only needed by `evaluate.py`.

```json
{"id": "q1", "question": "...", "gold_ids": ["c1", "c7"]}
```

In both files `id` must be unique; if it is missing, the row number (from 0) is used.

## Usage

```bash
HEAD=path/to/unreal_head.safetensors

# 1. Encode the corpus: chunk vectors and a BM25 index (the head is not needed here).
python encode_corpus.py --corpus corpus.jsonl --index_dir index

# 2. Retrieve, for one question or a file of queries.
python retrieve.py --index_dir index --head_weights $HEAD --question "Who designed the first ...?"
python retrieve.py --index_dir index --head_weights $HEAD --queries queries.jsonl --output results.jsonl

# 3. Complete-evidence recall@k: the share of queries with all their gold chunks in the top k.
python evaluate.py --queries queries.jsonl --results results.jsonl --k 1 5 10 20
```

`retrieve.py` gives each question the corpus's top-5 BM25 passages as context. To use other passages,
add them to the query: `{"question": "...", "context": ["passage 1", "passage 2"]}`.

## Python

```python
from unreal import BM25, UNREAL, search

texts = ["...", "..."]  # the corpus chunks
model = UNREAL.from_pretrained("path/to/unreal_head.safetensors")
chunk_vectors = model.encode_chunks(texts).numpy()  # [num_chunks, 7, 2688]

bm25 = BM25.build(texts)
question = "..."
context = [texts[row] for row in bm25.search([question], k=5)[0]]
query_vectors = model.encode_queries([question], [context])  # [1, 4, 2688]
scores, rows = search(query_vectors, chunk_vectors, top_k=10)
print([texts[row] for row in rows[0]])
```
