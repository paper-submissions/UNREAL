"""Retrieves the top chunks for questions, from a corpus encoded by encode_corpus.py.

Give one --question, or a --queries JSONL file with {"id": ..., "question": ...} per line ("id" is optional).
Each question is read after the corpus's top-5 BM25 passages, as in training. A query in the file
can bring its own passages instead: {"question": ..., "context": ["passage", ...]}.
"""

import argparse
import json
import os

import numpy as np

from unreal import BM25, NUM_CONTEXT_PASSAGES, UNREAL, search
from unreal.model import DEFAULT_BACKBONE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index_dir", required=True, help="folder written by encode_corpus.py")
    parser.add_argument("--head_weights", required=True, help="UNREAL head weights (.safetensors)")
    queries_from = parser.add_mutually_exclusive_group(required=True)
    queries_from.add_argument("--question")
    queries_from.add_argument("--queries", help="JSONL file, one query per line")
    parser.add_argument("--output", help="JSONL results file; required with --queries")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE, help="Hugging Face model id or local folder")
    args = parser.parse_args()
    if args.queries and not args.output:
        parser.error("--queries needs --output")

    with open(os.path.join(args.index_dir, "chunks.jsonl")) as f:
        chunks = [json.loads(line) for line in f]
    if args.question is not None:
        queries = [{"id": 0, "question": args.question}]
    else:
        with open(args.queries) as f:
            queries = [json.loads(line) for line in f if line.strip()]
        if not queries:
            raise SystemExit(f"{args.queries} has no queries.")
        for row, query in enumerate(queries):
            query.setdefault("id", row)

    need_context = [query for query in queries if "context" not in query]
    if need_context:
        bm25 = BM25.load(os.path.join(args.index_dir, "bm25"))
        top_rows = bm25.search([query["question"] for query in need_context], NUM_CONTEXT_PASSAGES)
        for query, rows in zip(need_context, top_rows):
            query["context"] = [chunks[row]["text"] for row in rows]

    model = UNREAL.from_pretrained(args.head_weights, backbone=args.backbone)
    vectors = model.encode_queries([query["question"] for query in queries], [query["context"] for query in queries])
    embeddings = np.load(os.path.join(args.index_dir, "embeddings.npy"), mmap_mode="r")
    scores, rows = search(vectors, embeddings, args.top_k)

    if args.question is not None:
        for rank, (score, row) in enumerate(zip(scores[0].tolist(), rows[0].tolist()), start=1):
            print(f"{rank:>3}  score {score:.3f}  id {chunks[row]['id']}\n     {chunks[row]['text']}")
        return
    with open(args.output, "w") as f:
        for query, query_scores, query_rows in zip(queries, scores.tolist(), rows.tolist()):
            chunk_ids = [chunks[row]["id"] for row in query_rows]
            f.write(json.dumps({"id": query["id"], "chunk_ids": chunk_ids, "scores": query_scores}) + "\n")


if __name__ == "__main__":
    main()
