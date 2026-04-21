from __future__ import annotations

from pathlib import Path
from typing import Literal
import urllib.request

import torch

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


class Dataloader:
    """Loads Tiny Shakespeare and serves train/val autoregressive batches."""

    def __init__(
        self,
        dataset_path: Path,
        batch_size: int,
        block_size: int,
        device: str,
        train_split_ratio: float = 0.9,
    ) -> None:
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.train_split_ratio = train_split_ratio

        self._ensure_tiny_shakespeare()
        text = self.dataset_path.read_text(encoding="utf-8")
        self.stoi, self.itos = self._build_vocab(text)
        self.data = torch.tensor(self.encode(text), dtype=torch.long)

        split_index = int(self.train_split_ratio * len(self.data))
        self.train_data = self.data[:split_index]
        self.val_data = self.data[split_index:]

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @property
    def train_tokens(self) -> int:
        return int(self.train_data.numel())

    @property
    def val_tokens(self) -> int:
        return int(self.val_data.numel())

    def _ensure_tiny_shakespeare(self) -> None:
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        if self.dataset_path.exists():
            return
        print(f"Downloading Tiny Shakespeare dataset to {self.dataset_path}...")
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, self.dataset_path)

    @staticmethod
    def _build_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        return stoi, itos

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.itos[i] for i in token_ids)

    def get_batch(self, split: Literal["train", "val"]) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])
        return x.to(self.device), y.to(self.device)
