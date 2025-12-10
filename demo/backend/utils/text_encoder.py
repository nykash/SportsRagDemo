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
import pandas as pd
import numpy as np
import clip

# config
BATCH_SIZE = 4
FRAMES_PER_CLIP = 16
LEARNING_RATE = 5e-6  
MAX_EPOCHS = 20
VAL_SPLIT = 0.1

checkpoint_path = "best_model_cpu.pt"

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


model = VideoCLIPLightning()
model.model.load_state_dict(torch.load(checkpoint_path))
model.eval()
model.to("cpu")


def encode_text(text: str):
    with torch.no_grad():
        vector = model.model.encode_text(clip.tokenize([text])).detach()[0].numpy().tolist()

    return vector

