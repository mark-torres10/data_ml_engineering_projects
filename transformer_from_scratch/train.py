"""Train a small autoregressive LM on Tiny Shakespeare.

Run from repository root:
    PYTHONPATH=. uv run python transformer_from_scratch/train.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal
import urllib.request

import torch

from transformer_from_scratch.model import GPT, GPTConfig

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


@dataclass
class TrainConfig:
    batch_size: int = 64
    block_size: int = 128
    max_iters: int = 5000
    eval_interval: int = 250
    eval_iters: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1337


@dataclass
class LossEvent:
    step: int
    train_loss: float
    val_loss: float
    wall_time: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("transformer_from_scratch/input.txt"),
        help="Path to Tiny Shakespeare text file (downloaded if missing).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("transformer_from_scratch/output"),
        help="Directory where timestamped run folders are created.",
    )
    return parser.parse_args()


def ensure_tiny_shakespeare(dataset_path: Path) -> None:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    if dataset_path.exists():
        return
    print(f"Downloading Tiny Shakespeare dataset to {dataset_path}...")
    urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, dataset_path)


def make_output_dir(output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    output_dir = output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def build_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> list[int]:
    return [stoi[ch] for ch in text]


def get_batch(
    split: Literal["train", "val"],
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    cfg: TrainConfig,
) -> dict[str, float]:
    out: dict[str, float] = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            xb, yb = get_batch(
                split=split,
                train_data=train_data,
                val_data=val_data,
                batch_size=cfg.batch_size,
                block_size=cfg.block_size,
                device=cfg.device,
            )
            _, loss = model(xb, yb)
            if loss is None:
                raise RuntimeError("Loss should not be None during evaluation.")
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main() -> None:
    args = parse_args()
    cfg = TrainConfig()
    torch.manual_seed(cfg.seed)

    ensure_tiny_shakespeare(args.dataset_path)
    output_dir = make_output_dir(args.output_root)

    text = args.dataset_path.read_text(encoding="utf-8")
    stoi, itos = build_vocab(text)
    data = torch.tensor(encode(text, stoi), dtype=torch.long)

    split_index = int(0.9 * len(data))
    train_data = data[:split_index]
    val_data = data[split_index:]

    model_cfg = GPTConfig(
        block_size=cfg.block_size,
        vocab_size=len(stoi),
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
    )
    model = GPT(model_cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    metadata = {
        "train_config": asdict(cfg),
        "model_config": asdict(model_cfg),
        "dataset_path": str(args.dataset_path),
        "output_dir": str(output_dir),
        "vocab_size": len(stoi),
        "train_tokens": int(train_data.numel()),
        "val_tokens": int(val_data.numel()),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (output_dir / "vocab.json").write_text(
        json.dumps({"stoi": stoi, "itos": itos}, indent=2), encoding="utf-8"
    )

    history: list[LossEvent] = []
    print(f"Output directory: {output_dir}")
    print(
        f"Training on {cfg.device} with vocab_size={len(stoi)}, "
        f"train_tokens={train_data.numel()}, val_tokens={val_data.numel()}"
    )

    for step in range(cfg.max_iters + 1):
        if step % cfg.eval_interval == 0 or step == cfg.max_iters:
            losses = estimate_loss(model, train_data, val_data, cfg)
            event = LossEvent(
                step=step,
                train_loss=losses["train"],
                val_loss=losses["val"],
                wall_time=datetime.now().isoformat(timespec="seconds"),
            )
            history.append(event)
            print(
                f"step {step:5d} | train loss {event.train_loss:.4f} | "
                f"val loss {event.val_loss:.4f}"
            )

        xb, yb = get_batch(
            split="train",
            train_data=train_data,
            val_data=val_data,
            batch_size=cfg.batch_size,
            block_size=cfg.block_size,
            device=cfg.device,
        )
        _, loss = model(xb, yb)
        if loss is None:
            raise RuntimeError("Loss should not be None during training.")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    (output_dir / "loss_history.json").write_text(
        json.dumps([asdict(event) for event in history], indent=2), encoding="utf-8"
    )
    torch.save(model.state_dict(), output_dir / "model.pt")
    print(f"Training complete. Saved model and logs to {output_dir}")


if __name__ == "__main__":
    main()
