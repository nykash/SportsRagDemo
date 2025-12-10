import os
import glob
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from decord import VideoReader, cpu
import pandas as pd
import numpy as np
import clip

# config
BATCH_SIZE = 4
FRAMES_PER_CLIP = 16
LEARNING_RATE = 5e-6  
MAX_EPOCHS = 20
VAL_SPLIT = 0.1

# paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.join(SCRIPT_DIR, "/home/richard/Desktop/workspace/SportsRagDemo/experiments")
EXCLUDE_EXACT_FOLDERS = ["clips", "germany_v_japan_clips"]

def gather_all_data(experiments_root):
    all_rows = []
    if not os.path.exists(experiments_root):
        raise FileNotFoundError(f"Experiments folder not found at: {experiments_root}")

    candidates = [d for d in os.listdir(experiments_root) if os.path.isdir(os.path.join(experiments_root, d))]
    print(f"\nScanning {experiments_root}...")
    
    for folder_name in candidates:
        folder_path = os.path.join(experiments_root, folder_name)

        if folder_name in EXCLUDE_EXACT_FOLDERS:
            continue
        if "germany_v_japan" in folder_name:
            continue

        print(f"Processing: {folder_name}...")
        match_name = folder_name.replace("clips_", "").replace("clips", "")
        csv_candidates = glob.glob(os.path.join(experiments_root, f"*{match_name}*.csv"))
        
        if not csv_candidates:
            print(f"  -> WARNING: No matching CSV found for {folder_name}")
            continue
            
        csv_path = csv_candidates[0]
        df = pd.read_csv(csv_path)
        
        id_to_filename = {}
        for f in os.listdir(folder_path):
            if f.endswith(".mp4") and "clip_" in f:
                try:
                    parts = f.split('_')
                    if len(parts) >= 2:
                        idx_id = int(parts[1])
                        id_to_filename[idx_id] = f
                except: continue
        
        valid_count = 0
        for idx, row in df.iterrows():
            filename = None
            if idx in id_to_filename:
                filename = id_to_filename[idx]
            
            if filename:
                full_path = os.path.join(folder_path, filename)
                text_content = str(row.get('text', row.get('transcript', '')))
                all_rows.append({"video_path": full_path, "text": text_content})
                valid_count += 1
                
        print(f"  -> Added {valid_count} clips.")

    return pd.DataFrame(all_rows)

class VideoTextDataset(Dataset):
    def __init__(self, dataframe, preprocess, tokenizer):
        self.df = dataframe.reset_index(drop=True)
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def _load_video_frames(self, video_path):
        if not os.path.exists(video_path):
            return torch.zeros((FRAMES_PER_CLIP, 3, 224, 224))
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            if len(vr) < FRAMES_PER_CLIP: 
                return torch.zeros((FRAMES_PER_CLIP, 3, 224, 224))
            
            indices = np.linspace(0, len(vr) - 1, FRAMES_PER_CLIP).astype(int)
            frames = vr.get_batch(indices).asnumpy()
            
            from PIL import Image
            return torch.stack([self.preprocess(Image.fromarray(f)) for f in frames])
        except Exception:
            return torch.zeros((FRAMES_PER_CLIP, 3, 224, 224))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_tensor = self._load_video_frames(row['video_path'])
        text_tokenized = self.tokenizer(row['text'], truncate=True).squeeze(0)
        return video_tensor, text_tokenized

# Lightning Module

class VideoCLIPLightning(pl.LightningModule):
    def __init__(self, model_name="ViT-B/16", lr=5e-6, freeze_backbone=False):
        super().__init__()
        self.save_hyperparameters()
        
        # 1. Load CLIP
        self.model, self.clip_preprocess = clip.load(model_name, device='cpu', jit=False)
        self.model = self.model.float()
        
        # 2. Freeze Logic
        if freeze_backbone:
            print("Freezing Visual and Text Backbones...")
            for param in self.model.parameters():
                param.requires_grad = False
            
            self.model.logit_scale.requires_grad = True
        else:
            print("Full Fine-Tuning Enabled (All weights trainable).")
            pass

        # 3. Losses
        self.loss_img = nn.CrossEntropyLoss()
        self.loss_txt = nn.CrossEntropyLoss()

    def forward(self, video, text):
        # Video: (B, T, C, H, W)
        b, t, c, h, w = video.shape
        
        # Flatten time dim: (B*T, C, H, W)
        video_flat = video.view(-1, c, h, w)
        
        # Get frame embeddings from CLIP: (B*T, Embed_Dim)
        frame_features = self.model.encode_image(video_flat)
        
        # Reshape back to sequence: (B, T, Embed_Dim)
        video_seq = frame_features.view(b, t, -1)
        
        # Mean Pooling
        video_emb = video_seq.mean(dim=1)
        
        # Get Text Embeddings: (B, Embed_Dim)
        text_emb = self.model.encode_text(text)
        
        # Normalize
        video_emb = video_emb / video_emb.norm(dim=-1, keepdim=True)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        
        return video_emb, text_emb, self.model.logit_scale.exp()

    def training_step(self, batch, batch_idx):
        video, text = batch
        video_emb, text_emb, logit_scale = self(video, text)
        
        # Contrastive Loss
        logits = logit_scale * video_emb @ text_emb.t()
        labels = torch.arange(len(video), device=self.device)
        
        loss = (self.loss_img(logits, labels) + self.loss_txt(logits.t(), labels)) / 2
        
        # Accuracy
        pred = logits.argmax(dim=1)
        acc = (pred == labels).float().mean()
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        video, text = batch
        video_emb, text_emb, logit_scale = self(video, text)
        
        logits = logit_scale * video_emb @ text_emb.t()
        labels = torch.arange(len(video), device=self.device)
        
        loss = (self.loss_img(logits, labels) + self.loss_txt(logits.t(), labels)) / 2
        acc = (logits.argmax(dim=1) == labels).float().mean()
        
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Clamp logit scale to prevent gradient explosion
        with torch.no_grad():
            self.model.logit_scale.clamp_(0, math.log(100))

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        
        # Cosine Annealing Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


if __name__ == "__main__":
    print(f"--- Starting Standard Training Script (No LoRA) ---")
    
    # 1. Aggregate Data
    full_df = gather_all_data(EXPERIMENTS_DIR)
    if len(full_df) == 0:
        raise RuntimeError("No data found. Exiting.")

    # 2. Split Data
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
    val_cutoff = int(len(full_df) * VAL_SPLIT)
    val_df = full_df.iloc[:val_cutoff]
    train_df = full_df.iloc[val_cutoff:]
    
    print(f"Train Set: {len(train_df)} clips | Validation Set: {len(val_df)} clips")
    
    # 3. Model
    # NOTE: Set freeze_backbone=False for Full Fine-Tuning (Best performance if careful)
    # Set freeze_backbone=True for Linear Probing (Safest, fastest)
    model = VideoCLIPLightning(
        model_name="ViT-B/16", 
        lr=LEARNING_RATE, 
        freeze_backbone=False 
    )
    
    # 4. Data Loaders
    train_ds = VideoTextDataset(train_df, model.clip_preprocess, clip.tokenize)
    val_ds = VideoTextDataset(val_df, model.clip_preprocess, clip.tokenize)
    
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, num_workers=4, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, num_workers=4, shuffle=False, pin_memory=True
    )

    # 5. Checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="standard-clip-{epoch:02d}-{val_acc:.2f}",
        save_top_k=1,
        monitor="val_acc",
        mode="max" 
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # 6. Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1, 
        precision="16",
        max_epochs=MAX_EPOCHS,
        accumulate_grad_batches=4,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=5,
        enable_progress_bar=True
    )
    
    print("Starting Training...")
    trainer.fit(model, train_loader, val_loader)