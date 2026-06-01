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

import torch
import wandb

from transformer_from_scratch.config import TrainConfig
from transformer_from_scratch.dataloader import Dataloader
from transformer_from_scratch.model import GPT, GPTConfig


@dataclass
class LossEvent:
    epoch: int
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


def make_output_dir(output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    output_dir = output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


class GptTrainer:
    def __init__(
        self,
        cfg: TrainConfig,
        dataloader: Dataloader,
        output_dir: Path,
        dataset_path: Path,
    ) -> None:
        """Initializes the GPT trainer."""
        self.cfg = cfg
        self.dataloader = dataloader
        self.output_dir = output_dir
        self.dataset_path = dataset_path
        self.history: list[LossEvent] = []

        model_cfg = GPTConfig(
            block_size=cfg.block_size,
            vocab_size=dataloader.vocab_size,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_embd=cfg.n_embd,
            dropout=cfg.dropout,
        )
        self.model_cfg = model_cfg
        self.model = GPT(model_cfg).to(cfg.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay,
        )
        self.wandb_run: wandb.sdk.wandb_run.Run | None = None

    def init_wandb(self) -> None:
        """Initializes a Weights & Biases run for this training job."""
        if not self.cfg.use_wandb:
            return
        self.wandb_run = wandb.init(
            project=self.cfg.wandb_project,
            entity=self.cfg.wandb_entity,
            name=self.cfg.wandb_run_name,
            mode=self.cfg.wandb_mode,
            dir=str(self.output_dir),
            config={
                "train_config": asdict(self.cfg),
                "model_config": asdict(self.model_cfg),
                "dataset_path": str(self.dataset_path),
            },
        )

    def log_to_wandb(self, metrics: dict[str, float], epoch: int) -> None:
        if not self.wandb_run:
            return
        wandb.log(metrics, step=epoch)

    @torch.no_grad()
    def estimate_loss(self) -> dict[str, float]:
        """Esimating the loss. Computing exact loss over the entire dataset is
        expensive and we do this for logging purposes anyways. Instead,
        we do random Monte-Carlo style estimates from random mini-batches.
        """
        self.model.eval()

        train_losses = torch.zeros(self.cfg.eval_iters)
        for eval_iter in range(self.cfg.eval_iters):
            batch_x, batch_y = self.dataloader.get_batch(split="train")
            _, loss = self.model(batch_x, batch_y)
            if loss is None:
                raise RuntimeError("Loss should not be None during train evaluation.")
            train_losses[eval_iter] = loss.item()

        val_losses = torch.zeros(self.cfg.eval_iters)
        for eval_iter in range(self.cfg.eval_iters):
            batch_x, batch_y = self.dataloader.get_batch(split="val")
            _, loss = self.model(batch_x, batch_y)
            if loss is None:
                raise RuntimeError("Loss should not be None during val evaluation.")
            val_losses[eval_iter] = loss.item()

        self.model.train()
        return {
            "train": train_losses.mean().item(),
            "val": val_losses.mean().item(),
        }

    def save_run_metadata(self) -> None:
        """Exports run metadata to the output directory."""
        metadata = {
            "train_config": asdict(self.cfg),
            "model_config": asdict(self.model_cfg),
            "dataset_path": str(self.dataset_path),
            "output_dir": str(self.output_dir),
            "vocab_size": self.dataloader.vocab_size,
            "train_tokens": self.dataloader.train_tokens,
            "val_tokens": self.dataloader.val_tokens,
        }
        run_config_path = self.output_dir / "run_config.json"

        run_config_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def evaluate_and_log_epoch(self, epoch: int) -> None:
        """Runs periodic evaluation, appends metrics, and prints status."""
        losses = self.estimate_loss()
        event = LossEvent(
            epoch=epoch,
            train_loss=losses["train"],
            val_loss=losses["val"],
            wall_time=datetime.now().isoformat(timespec="seconds"),
        )
        self.history.append(event)
        print(
            f"epoch {epoch:5d} | train loss {event.train_loss:.4f} | "
            f"val loss {event.val_loss:.4f}"
        )
        self.log_to_wandb(
            metrics={
                "epoch": float(epoch),
                "train/loss": event.train_loss,
                "val/loss": event.val_loss,
                "train/lr": self.optimizer.param_groups[0]["lr"],
            },
            epoch=epoch,
        )

    def train(self) -> None:
        """Trains the model."""
        self.save_run_metadata()
        self.init_wandb()

        print(f"Output directory: {self.output_dir}")
        print(
            f"Training on {self.cfg.device} with vocab_size={self.dataloader.vocab_size}, "
            f"train_tokens={self.dataloader.train_tokens}, val_tokens={self.dataloader.val_tokens}"
        )

        total_epochs = self.cfg.max_epochs + 1

        try:
            for epoch in range(total_epochs):

                if epoch % self.cfg.eval_interval == 0 or epoch == self.cfg.max_epochs:
                    self.evaluate_and_log_epoch(epoch)

                batch_x, batch_y = self.dataloader.get_batch(split="train")
                _, loss = self.model(batch_x, batch_y)
                if loss is None:
                    raise RuntimeError("Loss should not be None during training.")

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                self.log_to_wandb(
                    metrics={
                        "epoch": float(epoch),
                        "train/loss_step": loss.item(),
                    },
                    epoch=epoch,
                )

            loss_history_path = self.output_dir / "loss_history.json"
            loss_history_path.write_text(
                json.dumps([asdict(event) for event in self.history], indent=2),
                encoding="utf-8",
            )

            output_model_path = self.output_dir / "model.pt"
            torch.save(self.model.state_dict(), output_model_path)
            print(f"Training complete. Saved model and logs to {output_model_path}")
        finally:
            if self.wandb_run:
                self.wandb_run.finish()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig()
    torch.manual_seed(cfg.seed)

    dataloader = Dataloader(
        dataset_path=args.dataset_path,
        batch_size=cfg.batch_size,
        block_size=cfg.block_size,
        device=cfg.device,
    )
    output_dir = make_output_dir(args.output_root)
    trainer = GptTrainer(
        cfg=cfg,
        dataloader=dataloader,
        output_dir=output_dir,
        dataset_path=args.dataset_path,
    )
    trainer.train()


if __name__ == "__main__":
    main()
