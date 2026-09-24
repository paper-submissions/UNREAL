"""UNREAL: retrieval with a small learned head on a frozen LLM."""

import os

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from .backbone import NemotronH
from .prompt import query_token_ids

DEFAULT_BACKBONE = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
READOUT_LAYERS = (5, 12, 19, 26, 33, 42)  # the backbone's attention blocks, up to the last one
CHUNK_LAYER = READOUT_LAYERS[-1]
NUM_RETRIEVAL_TOKENS = 64
NUM_QUERY_VECTORS = 4
CHUNK_VECTORS = 7
MAX_CHUNK_TOKENS = 512
PLACEHOLDER_ID = 0  # token id at the head's positions; its embedding is replaced


def pool_chunk(hidden: torch.Tensor) -> torch.Tensor:
    """[length, dim] -> [7, dim]: mean over 7 contiguous parts of the chunk, each RMS-normalized.

    A chunk shorter than 7 tokens repeats its last part, which leaves MaxSim scores unchanged.
    """
    num_parts = min(CHUNK_VECTORS, len(hidden))
    base, extra = divmod(len(hidden), num_parts)
    sizes = [base + 1] * extra + [base] * (num_parts - extra)
    parts = torch.stack([part.mean(0) for part in hidden.float().split(sizes)])
    parts = torch.cat([parts, parts[-1:].expand(CHUNK_VECTORS - num_parts, -1)])
    return parts * torch.rsqrt(parts.pow(2).mean(-1, keepdim=True) + 1e-6)


class UNREAL:
    """Frozen backbone + head: 64 retrieval tokens, 1 soft-prompt token and one weight per read-out layer.

    A query is [soft prompt | prompt | retrieval tokens]. Its embedding is the weighted sum, over the
    read-out layers, of the backbone's outputs at the retrieval tokens, averaged into 4 vectors.
    A chunk is represented by 7 vectors pooled from the backbone's output at the last read-out layer.
    """

    def __init__(self, backbone: NemotronH, tokenizer, head: dict[str, torch.Tensor] | None):
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.head = head
        self.device = backbone.embeddings.weight.device

    @classmethod
    def from_pretrained(cls, head_weights: str | None, backbone: str = DEFAULT_BACKBONE, device: str = "cuda"):
        """Loads the backbone (a local folder or a Hugging Face model id) and the head weights.

        The head is needed only to encode queries; chunks are encoded by the backbone alone.
        """
        if os.path.isdir(backbone):
            path = backbone
        else:
            path = snapshot_download(backbone, allow_patterns=["*.json", "*.jinja", "*.safetensors"])
        # UNREAL was trained with the tokenizer exactly as shipped, so its regex is left as is.
        tokenizer = AutoTokenizer.from_pretrained(path, fix_mistral_regex=False)
        model = NemotronH.from_pretrained(path, num_layers=CHUNK_LAYER + 1, device=device)
        attention_layers = tuple(i for i, kind in enumerate(model.block_types) if kind == "attention")
        if attention_layers != READOUT_LAYERS:
            raise ValueError(f"{backbone} is not the backbone UNREAL was trained on.")
        head = None
        if head_weights is not None:
            head = load_file(head_weights, device=str(device))
            dim = model.embeddings.embedding_dim
            expected = {
                "retrieval_tokens": (NUM_RETRIEVAL_TOKENS, dim),
                "soft_prompt": (1, dim),
                "layer_weights": (len(READOUT_LAYERS),),
            }
            if {name: tuple(tensor.shape) for name, tensor in head.items()} != expected:
                raise ValueError(f"{head_weights} does not hold the expected head tensors {expected}.")
        return cls(model, tokenizer, head)

    @torch.no_grad()
    def encode_queries(self, questions: list[str], contexts: list[list[str]], batch_size: int = 32) -> torch.Tensor:
        """Returns [num_queries, 4, dim] bf16 query vectors. `contexts` holds the passages shown before each question."""
        if self.head is None:
            raise ValueError("Encoding queries needs the head weights.")
        prompts = [query_token_ids(self.tokenizer, q, c) for q, c in zip(questions, contexts, strict=True)]
        vectors = []
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            lengths = torch.tensor([1 + len(ids) + NUM_RETRIEVAL_TOKENS for ids in batch])
            tokens = torch.full((len(batch), int(lengths.max())), PLACEHOLDER_ID)
            for row, ids in enumerate(batch):
                tokens[row, 1 : 1 + len(ids)] = torch.tensor(ids)
            embeds = self.backbone.embeddings(tokens.to(self.device))
            rows = torch.arange(len(batch), device=self.device)[:, None]
            slots = (lengths[:, None] - NUM_RETRIEVAL_TOKENS + torch.arange(NUM_RETRIEVAL_TOKENS)).to(self.device)
            embeds[:, 0] = self.head["soft_prompt"].to(embeds.dtype)
            embeds[rows, slots] = self.head["retrieval_tokens"].to(embeds.dtype)
            outputs = self.backbone(embeds, capture=READOUT_LAYERS)
            query = 0
            for weight, layer in zip(self.head["layer_weights"].float(), READOUT_LAYERS):
                query = query + outputs[layer][rows, slots].float() * weight
            vectors.append(query.view(len(batch), NUM_QUERY_VECTORS, -1, query.shape[-1]).mean(2))
        return torch.cat(vectors).to(torch.bfloat16)

    @torch.no_grad()
    def encode_chunks(self, texts: list[str], batch_size: int = 128) -> torch.Tensor:
        """Returns [num_chunks, 7, dim] float16 chunk vectors (on CPU)."""
        for i, text in enumerate(texts):
            if not text.strip():
                raise ValueError(f"Chunk {i} is empty.")
        token_ids = self.tokenizer(texts, add_special_tokens=False)["input_ids"]
        token_ids = [([self.tokenizer.bos_token_id] + ids)[:MAX_CHUNK_TOKENS] for ids in token_ids]
        order = sorted(range(len(texts)), key=lambda i: -len(token_ids[i]))
        vectors = torch.empty(len(texts), CHUNK_VECTORS, self.backbone.embeddings.embedding_dim, dtype=torch.float16)
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            tokens = torch.full((len(batch), len(token_ids[batch[0]])), PLACEHOLDER_ID)
            for row, i in enumerate(batch):
                tokens[row, : len(token_ids[i])] = torch.tensor(token_ids[i])
            hidden = self.backbone(self.backbone.embeddings(tokens.to(self.device)), capture=(CHUNK_LAYER,))
            for row, i in enumerate(batch):
                vectors[i] = pool_chunk(hidden[CHUNK_LAYER][row, : len(token_ids[i])]).half().cpu()
        return vectors
