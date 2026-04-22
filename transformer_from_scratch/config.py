
from dataclasses import dataclass
import torch


@dataclass
class TrainConfig:
    batch_size: int = 64
    block_size: int = 128
    max_epochs: int = 5000
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
    use_wandb: bool = True
    wandb_project: str = "transformer-from-scratch"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str = "online"


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
