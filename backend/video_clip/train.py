import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class LoRAConfig:
    r: int = 4
    alpha: int = 16
    dropout: float = 0.0
    target_modules: Optional[Sequence[str]] = None


class LoRALinear(nn.Module):
    """Injects a low-rank adapter into an existing Linear layer."""

    def __init__(self, linear: nn.Linear, config: LoRAConfig):
        super().__init__()
        self.linear = linear
        self.r = config.r
        self.scaling = config.alpha / config.r if config.r > 0 else 0.0
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        if self.r > 0:
            self.lora_A = nn.Parameter(torch.zeros(self.r, linear.out_features))
            self.lora_B = nn.Parameter(torch.zeros(linear.in_features, self.r))
            nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))
            nn.init.zeros_(self.lora_A)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.linear(x)
        if self.r > 0:
            lora_update = (self.dropout(x) @ self.lora_B) @ self.lora_A
            result = result + self.scaling * lora_update
        return result


def _target_matches(name: str, targets: Optional[Sequence[str]]) -> bool:
    if not targets:
        return True
    return any(target in name for target in targets)


def apply_lora_to_model(module: nn.Module, config: LoRAConfig, prefix: str = "") -> None:
    for name, child in module.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and _target_matches(child_prefix, config.target_modules):
            setattr(module, name, LoRALinear(child, config))
        else:
            apply_lora_to_model(child, config, child_prefix)


def freeze_clip_backbone(model: nn.Module, keep_lora_trainable: bool) -> None:
    for name, param in model.named_parameters():
        if keep_lora_trainable and "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def maybe_clamp_logit_scale(model: nn.Module, max_value: float = math.log(100)) -> None:
    if hasattr(model, "logit_scale"):
        with torch.no_grad():
            model.logit_scale.clamp_(0, max_value)

def read_frame(video: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = video.read()
    if not ret:
        raise ValueError(f"Could not read frame {frame_index}")

    return frame

class Fusion(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=2,
            batch_first=True
        )

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, x):               # x: B x seq x 1024
        _, h_n = self.gru(x)           # h_n: num_layers x B x 512
        h_last = h_n[-1]               # B x 512   (top layer's final hidden)
        return self.mlp(h_last)        # B x 1024

class EmbeddingModel(nn.Module):
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        freeze_clip: bool = False,
        lora_config: Optional[LoRAConfig] = None,
    ):
        super().__init__()
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained="openai")
        self.tokenizer = open_clip.get_tokenizer(model_name)

        if lora_config:
            apply_lora_to_model(self.model, lora_config)

        if freeze_clip:
            freeze_clip_backbone(self.model, keep_lora_trainable=bool(lora_config))

        clip_embed_dim = getattr(self.model, "text_projection", None)
        if isinstance(clip_embed_dim, torch.Tensor):
            embed_dim = clip_embed_dim.shape[1]
        else:
            embed_dim = getattr(self.model, "embed_dim", 512)

        self.fusion = Fusion(embed_dim)

    def encode_video(self, x):
        model_device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        x = x.to(device=model_device, dtype=model_dtype, non_blocking=True)
        og_shape = x.shape
        x = x.view(-1, *x.shape[2:]).contiguous()
        x = self.model.encode_image(x)
        x = x.view(og_shape[0], og_shape[1], *x.shape[1:]).contiguous()

        x = self.fusion(x)
        return x
    
    def encode_text(self, x: Sequence[str]):
        tokenized = self.tokenizer(x)
        tokenized = tokenized.to(next(self.model.parameters()).device)
        x = self.model.encode_text(tokenized)
        return x

    def compute_loss_and_accuracy(
        self, video: torch.Tensor, text: Sequence[str]
    ) -> Tuple[torch.Tensor, dict]:
        if video.shape[0] != len(text):
            raise ValueError("Video and text batch sizes must match for contrastive training.")

        video_embedding = self.encode_video(video)
        text_embedding = self.encode_text(text)

        video_embedding = F.normalize(video_embedding, dim=-1)
        text_embedding = F.normalize(text_embedding, dim=-1)
        logit_scale = self.model.logit_scale.exp()
        logits_per_video = logit_scale * video_embedding @ text_embedding.t()
        logits_per_text = logits_per_video.t()
        labels = torch.arange(video_embedding.shape[0], device=video_embedding.device)

        loss_i = F.cross_entropy(logits_per_video, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = 0.5 * (loss_i + loss_t)

        with torch.no_grad():
            acc_video = (logits_per_video.argmax(dim=1) == labels).float().mean()
            acc_text = (logits_per_text.argmax(dim=1) == labels).float().mean()
            mean_acc = 0.5 * (acc_video + acc_text)

        metrics = {
            "video_to_text_acc": acc_video.detach(),
            "text_to_video_acc": acc_text.detach(),
            "mean_acc": mean_acc.detach(),
        }
        return loss, metrics

class EmbeddingDataset(Dataset):
    def __init__(self, video_paths, text_paths, tokenizer, keep_frames=20):
        self.video_paths = video_paths  # .mp4 files
        self.text_paths = text_paths  # .txt files
        self.keep_frames = keep_frames
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        # video_path = self.video_paths[idx]
        # text_path = self.text_paths[idx]

        # with open(text_path, 'r') as f:
        #     text_description = f.read()

        # cap = cv2.VideoCapture(video_path)
        # if not cap.isOpened():
        #     raise ValueError(f"Could not open video: {video_path}")

        # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # frames = []
        # # pick a random frame from every keep_frames frames
        # for i in range(total_frames // self.keep_frames):
        #     frame = read_frame(cap, random.randint(0, self.keep_frames - 1) + i * self.keep_frames)
        #     frames.append(cv2.resize(frame, (224, 224)))
        # cap.release()

        # # convert frames to numpy arrays
        # frames = np.array(frames)

        # # frames should be (frames_count x 3 x height x width)
        # frames = frames.transpose(0, 3, 1, 2)

        # # normalize frames with mean and std as in pretrained open_clip model mean = [0.48145466, 0.4578275, 0.40821073] and std = [0.26862954, 0.26130258, 0.27577711]
        # mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(1, 3, 1, 1)
        # std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(1, 3, 1, 1)
        # frames = (frames - mean) / std

        # frames = torch.tensor(frames, dtype=torch.float32)

        frames = torch.randn(self.keep_frames, 3, 224, 224)
        text_description = random.choice(["This is a test", "This is a test 2", "This is a test 3"])

        return frames, text_description


def expand_paths(paths: Sequence[str], desired_length: int) -> Sequence[str]:
    if len(paths) == desired_length:
        return list(paths)
    if len(paths) == 1:
        return list(paths) * desired_length
    raise ValueError(
        f"Expected 1 path or {desired_length} paths, but received {len(paths)}."
    )


def run_epoch(
    model: EmbeddingModel,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    desc: str = "",
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    pbar_desc = desc or ("Train" if is_train else "Eval")
    pbar = tqdm(dataloader, desc=pbar_desc)
    for video, text in pbar:
        video = video.to(device, non_blocking=True)
        grad_context = torch.enable_grad() if is_train else torch.no_grad()
        with grad_context:
            loss, metrics = model.compute_loss_and_accuracy(video, text)

        batch_size = video.size(0)
        total_loss += loss.item() * batch_size
        total_acc += metrics["mean_acc"].item() * batch_size
        total_samples += batch_size

        if is_train and optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            maybe_clamp_logit_scale(model.model)

        avg_loss = total_loss / total_samples
        avg_acc = total_acc / total_samples
        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{avg_acc:.4f}"})

    return total_loss / total_samples, total_acc / total_samples


def save_history(history: Iterable[dict], output_path: Path) -> None:
    df = pd.DataFrame(history)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def plot_history(history: Iterable[dict], output_path: Path) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed; skipping loss curve plotting.")
        return
    df = pd.DataFrame(history)
    if df.empty:
        return

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(df["epoch"], df["train_loss"], label="Train Loss")
    ax1.plot(df["epoch"], df["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df["train_accuracy"], "--", color="tab:green", label="Train Acc")
    ax2.plot(df["epoch"], df["val_accuracy"], "--", color="tab:orange", label="Val Acc")
    ax2.set_ylabel("Accuracy")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper center")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CLIP-based video encoder with optional LoRA.")
    parser.add_argument("--model-name", type=str, default="ViT-B-32")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--freeze-clip", action="store_true")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora-targets",
        nargs="*",
        default=None,
        help="Optional list of module name substrings to restrict LoRA injection.",
    )
    parser.add_argument("--keep-frames", type=int, default=20)
    parser.add_argument("--train-video-paths", nargs="+", default=["clip.mp4"])
    parser.add_argument("--train-text-paths", nargs="+", default=["sample_text.txt"])
    parser.add_argument("--val-video-paths", nargs="+", default=["clip.mp4"])
    parser.add_argument("--val-text-paths", nargs="+", default=["sample_text.txt"])
    parser.add_argument("--train-size", type=int, default=128)
    parser.add_argument("--val-size", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="artifacts/video_clip")
    parser.add_argument("--device", type=str, default=None, help="Override auto device selection.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    lora_config = None
    if args.use_lora:
        lora_config = LoRAConfig(
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=args.lora_targets,
        )

    model = EmbeddingModel(
        model_name=args.model_name,
        freeze_clip=args.freeze_clip,
        lora_config=lora_config,
    ).to(device)
    tokenizer = open_clip.get_tokenizer(args.model_name)

    train_video_paths = expand_paths(args.train_video_paths, args.train_size)
    train_text_paths = expand_paths(args.train_text_paths, args.train_size)
    val_video_paths = expand_paths(args.val_video_paths, args.val_size)
    val_text_paths = expand_paths(args.val_text_paths, args.val_size)

    train_dataset = EmbeddingDataset(
        train_video_paths, train_text_paths, tokenizer, keep_frames=args.keep_frames
    )
    val_dataset = EmbeddingDataset(
        val_video_paths, val_text_paths, tokenizer, keep_frames=args.keep_frames
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise ValueError("No trainable parameters found. Check freeze/LoRA settings.")
    optimizer = torch.optim.Adam(trainable_parameters, lr=args.lr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "training_history.csv"
    curves_path = output_dir / "training_curves.png"
    best_model_path = output_dir / "best_model.pth"
    last_model_path = output_dir / "last_model.pth"
    best_epoch_path = output_dir / "best_epoch.txt"

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_since_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = run_epoch(
            model, train_loader, device, optimizer=optimizer, desc=f"Train {epoch}"
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, device, optimizer=None, desc=f"Val {epoch}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )
        save_history(history, history_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_since_improve = 0
            torch.save(model.state_dict(), best_model_path)
            best_epoch_path.write_text(f"{epoch},{val_loss:.6f}")
            print(f"New best model saved at epoch {epoch} with val loss {val_loss:.4f}.")
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs).")
                break

    torch.save(model.state_dict(), last_model_path)
    if history:
        plot_history(history, curves_path)

    if best_epoch != -1:
        print(f"Best epoch: {best_epoch} with val loss {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
