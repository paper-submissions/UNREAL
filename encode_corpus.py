"""Encodes a corpus for UNREAL: 7 vectors per chunk, plus a BM25 index for the query context.

The corpus is a JSONL file with one chunk per line: {"id": ..., "text": ...}. Ids must be unique;
"id" is optional (default: the row number, from 0).
"""

import argparse
import json
import os

import numpy as np

from unreal import BM25, UNREAL
from unreal.model import CHUNK_VECTORS, DEFAULT_BACKBONE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, help="JSONL file, one chunk per line")
    parser.add_argument("--index_dir", required=True, help="output folder")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE, help="Hugging Face model id or local folder")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    with open(args.corpus) as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    if not chunks:
        raise SystemExit(f"{args.corpus} has no chunks.")
    ids = [chunk.get("id", row) for row, chunk in enumerate(chunks)]
    texts = [chunk["text"] for chunk in chunks]
    if len({str(chunk_id) for chunk_id in ids}) != len(ids):
        raise SystemExit("Chunk ids must be unique.")
    for chunk_id, text in zip(ids, texts):
        if not text.strip():
            raise SystemExit(f"Chunk {chunk_id} has no text.")
    os.makedirs(args.index_dir, exist_ok=True)
    with open(os.path.join(args.index_dir, "chunks.jsonl"), "w") as f:
        for chunk_id, text in zip(ids, texts):
            f.write(json.dumps({"id": chunk_id, "text": text}) + "\n")
    BM25.build(texts).save(os.path.join(args.index_dir, "bm25"))

    model = UNREAL.from_pretrained(head_weights=None, backbone=args.backbone)
    vectors = np.lib.format.open_memmap(
        os.path.join(args.index_dir, "embeddings.npy"),
        mode="w+",
        dtype=np.float16,
        shape=(len(texts), CHUNK_VECTORS, model.backbone.embeddings.embedding_dim),
    )
    step = 4096
    for start in range(0, len(texts), step):
        vectors[start : start + step] = model.encode_chunks(texts[start : start + step], args.batch_size).numpy()
        print(f"Encoded {min(start + step, len(texts))}/{len(texts)} chunks.", flush=True)
    vectors.flush()


if __name__ == "__main__":
    main()
