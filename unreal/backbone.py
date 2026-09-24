"""The frozen Nemotron-H backbone (NVIDIA-Nemotron-3.5-Lightning-30B-A3B), run up to a given block."""

import json
import os
import re

import torch
import torch.nn.functional as F
from causal_conv1d import causal_conv1d_fn
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from safetensors import safe_open
from torch import nn


class GatedRMSNorm(nn.Module):
    """Mamba2's grouped RMSNorm, with the gate applied before normalizing."""

    def __init__(self, dim: int, group_size: int, eps: float):
        super().__init__()
        self.group_size = group_size
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float() * F.silu(gate.float())
        x = x.unflatten(-1, (-1, self.group_size))
        x = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).flatten(-2)
        return (x * self.weight.float()).to(dtype)


class Mamba2(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        dim = config["hidden_size"]
        self.num_heads = config["mamba_num_heads"]
        self.head_dim = config["mamba_head_dim"]
        self.num_groups = config["n_groups"]
        self.state_size = config["ssm_state_size"]
        self.chunk_size = config["chunk_size"]
        self.inner_size = self.num_heads * self.head_dim
        self.conv_size = self.inner_size + 2 * self.num_groups * self.state_size
        self.in_proj = nn.Linear(dim, self.inner_size + self.conv_size + self.num_heads, bias=False)
        self.conv1d = nn.Conv1d(self.conv_size, self.conv_size, config["conv_kernel"], groups=self.conv_size)
        self.dt_bias = nn.Parameter(torch.empty(self.num_heads))
        self.A_log = nn.Parameter(torch.empty(self.num_heads))
        self.D = nn.Parameter(torch.empty(self.num_heads))
        self.norm = GatedRMSNorm(self.inner_size, self.inner_size // self.num_groups, config["layer_norm_epsilon"])
        self.out_proj = nn.Linear(self.inner_size, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        gate, conv_in, dt = self.in_proj(x).split([self.inner_size, self.conv_size, self.num_heads], dim=-1)
        conv_out = causal_conv1d_fn(
            conv_in.contiguous().transpose(1, 2),
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            activation="silu",
        ).transpose(1, 2)
        group_size = self.num_groups * self.state_size
        h, b, c = conv_out.split([self.inner_size, group_size, group_size], dim=-1)
        y = mamba_chunk_scan_combined(
            h.reshape(batch, length, self.num_heads, self.head_dim),
            dt,
            -torch.exp(self.A_log.float()),
            b.reshape(batch, length, self.num_groups, self.state_size),
            c.reshape(batch, length, self.num_groups, self.state_size),
            chunk_size=self.chunk_size,
            D=self.D,
            dt_bias=self.dt_bias,
            dt_softplus=True,
        )
        return self.out_proj(self.norm(y.reshape(batch, length, self.inner_size), gate))


class Attention(nn.Module):
    """Causal grouped-query attention. Nemotron-H uses no positional encoding."""

    def __init__(self, config: dict):
        super().__init__()
        dim = config["hidden_size"]
        self.head_dim = config["head_dim"]
        self.q_proj = nn.Linear(dim, config["num_attention_heads"] * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, config["num_key_value_heads"] * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, config["num_key_value_heads"] * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config["num_attention_heads"] * self.head_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        q, k, v = (
            proj(x).view(batch, length, -1, self.head_dim).transpose(1, 2)
            for proj in (self.q_proj, self.k_proj, self.v_proj)
        )
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.head_dim**-0.5, enable_gqa=True)
        return self.o_proj(out.transpose(1, 2).reshape(batch, length, -1))


class MLP(nn.Module):
    """down(relu(up(x))^2)."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.relu(self.up_proj(x)).square())


class Router(nn.Module):
    def __init__(self, dim: int, num_experts: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, dim))
        self.register_buffer("e_score_correction_bias", torch.empty(num_experts, dtype=torch.float32))


class MoE(nn.Module):
    """Top-k routed ReLU^2 experts plus a shared expert."""

    def __init__(self, config: dict):
        super().__init__()
        dim = config["hidden_size"]
        self.top_k = config["num_experts_per_tok"]
        self.scale = config["routed_scaling_factor"]
        self.gate = Router(dim, config["n_routed_experts"])
        self.experts = nn.ModuleList(
            MLP(dim, config["moe_intermediate_size"]) for _ in range(config["n_routed_experts"])
        )
        self.shared_experts = MLP(dim, config["moe_shared_expert_intermediate_size"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        # The router runs in fp32; the correction bias only chooses the experts, it does not weight them.
        scores = torch.sigmoid(F.linear(x.float(), self.gate.weight.float()))
        chosen = torch.topk(scores + self.gate.e_score_correction_bias, self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(-1, chosen)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * self.scale
        out = torch.zeros_like(x)
        for expert in chosen.unique().tolist():
            token, slot = torch.where(chosen == expert)
            y = self.experts[expert](x[token])
            out.index_add_(0, token, (y.float() * weights[token, slot, None]).to(x.dtype))
        return (out + self.shared_experts(x)).view(shape)


MIXERS = {"mamba": Mamba2, "attention": Attention, "moe": MoE}


class Block(nn.Module):
    def __init__(self, config: dict, block_type: str):
        super().__init__()
        self.norm = nn.RMSNorm(config["hidden_size"], eps=config["layer_norm_epsilon"])
        self.mixer = MIXERS[block_type](config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


class NemotronH(nn.Module):
    """The first `num_layers` blocks of the backbone. Final norm and LM head are not needed."""

    def __init__(self, config: dict, num_layers: int):
        super().__init__()
        self.block_types = config["layers_block_type"][:num_layers]
        self.embeddings = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList(Block(config, block_type) for block_type in self.block_types)

    def forward(self, embeds: torch.Tensor, capture: tuple[int, ...]) -> dict[int, torch.Tensor]:
        """Runs all blocks on `embeds` [batch, length, dim]; returns the output of each block in `capture`."""
        outputs = {}
        h = embeds
        for index, block in enumerate(self.layers):
            h = block(h)
            if index in capture:
                outputs[index] = h
        return outputs

    @classmethod
    def from_pretrained(cls, path: str, num_layers: int, device: str) -> "NemotronH":
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        with torch.device("meta"):
            model = cls(config, num_layers)
        with open(os.path.join(path, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]
        files = {}
        for name, file in weight_map.items():
            match = re.match(r"backbone\.(embeddings\.|layers\.(\d+)\.)", name)
            if match and (match.group(2) is None or int(match.group(2)) < num_layers):
                files.setdefault(file, []).append(name)
        state = {}
        for file, names in files.items():
            with safe_open(os.path.join(path, file), framework="pt", device=str(device)) as f:
                for name in names:
                    state[name.removeprefix("backbone.")] = f.get_tensor(name)
        model.load_state_dict(state, strict=True, assign=True)
        return model.eval().requires_grad_(False)
