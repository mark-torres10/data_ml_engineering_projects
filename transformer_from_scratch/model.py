from __future__ import annotations

import inspect
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from transformer_from_scratch.config import GPTConfig


class LayerNorm(nn.Module):
    """LayerNorm with optional bias for GPT-2 parity.
    
    GPT-2 didn't have it, according to NanoGPT, but the PyTorch
    implementation does. So we're going to keep it optional."""

    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class MultiAttentionHead(nn.Module):
    """Projects hidden states to multi-head Q/K/V and merges heads back."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

    def project_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Projects the input tensor `x` to Q, K, and V tensors."""
        bsz, seq_len, _ = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(bsz, seq_len, self.n_head, self.head_size).transpose(1, 2)
        q = q.view(bsz, seq_len, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, self.head_size).transpose(1, 2)
        return q, k, v

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merges the heads of the input tensor `x`."""
        bsz, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_embd)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.multi_head = MultiAttentionHead(config)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.size()
        q, k, v = self.multi_head.project_qkv(x)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.multi_head.head_size))

        # add causal mask to the attention scores
        att = att.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v

        y = self.multi_head.merge_heads(y)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with residual connections."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    """Decoder-only GPT language model."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.vocab_size is None or config.block_size is None:
            raise ValueError("vocab_size and block_size must be set")
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = idx.device
        _, seq_len = idx.size()
        if seq_len > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {seq_len}, block size is only {self.config.block_size}"
            )

        # build position indices
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)

        # do token embedding lookup for each position, getting the embedding
        # tensor for each position
        tok_emb = self.transformer.wte(idx)

        # do positional embedding lookup for each position, getting the embedding
        # tensor for each position
        pos_emb = self.transformer.wpe(pos)

        # adds position embeddings to token embeddings. also applies dropout.
        x = self.transformer.drop(tok_emb + pos_emb)

        # apply each transformer block to the input
        for block in self.transformer.h:
            x = block(x)

        # final layer normalization
        x = self.transformer.ln_f(x)

        # apply the linear head project
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = {"fused": True} if use_fused else {}
        return torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
