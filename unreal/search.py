"""Exact MaxSim search over the chunk vectors."""

import numpy as np
import torch


def maxsim(queries: torch.Tensor, chunks: torch.Tensor) -> torch.Tensor:
    """[q, 4, dim] x [n, 7, dim] -> [q, n]: for each query vector, the best-matching chunk vector; then the mean."""
    num_chunks, num_vectors, dim = chunks.shape
    similarity = torch.matmul(queries, chunks.reshape(-1, dim).T)
    return similarity.view(*similarity.shape[:2], num_chunks, num_vectors).amax(-1).mean(1)


@torch.no_grad()
def search(
    queries: torch.Tensor, chunks: np.ndarray, top_k: int, block_size: int = 16384, query_batch: int = 256
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scores every query against every chunk; returns the top-k (scores, row indices), best first.

    `chunks` is the [num_chunks, 7, dim] array written by encode_corpus.py, and may be memory-mapped:
    it is read block by block. Scoring is in bf16.
    """
    queries = queries.to(torch.bfloat16)
    best_scores = best_ids = None
    for start in range(0, len(chunks), block_size):
        block = torch.tensor(chunks[start : start + block_size], device=queries.device).to(torch.bfloat16)
        scores = torch.cat([maxsim(queries[i : i + query_batch], block) for i in range(0, len(queries), query_batch)])
        scores = scores.float()
        ids = torch.arange(start, start + len(block), device=queries.device).expand(len(queries), -1)
        if best_scores is not None:
            scores = torch.cat([best_scores, scores], dim=1)
            ids = torch.cat([best_ids, ids], dim=1)
        top = scores.topk(min(top_k, scores.shape[1]), dim=1)
        best_scores, best_ids = top.values, ids.gather(1, top.indices)
    return best_scores.cpu(), best_ids.cpu()
