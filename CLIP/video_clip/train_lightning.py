import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from decord import VideoReader, cpu
import pandas as pd
import numpy as np
import clip

BATCH_SIZE = 4
FRAMES_PER_CLIP = 16
LEARNING_RATE = 1e-5
MAX_EPOCHS = 20
VAL_SPLIT = 0.1

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.join(SCRIPT_DIR, "/home/richard/Desktop/workspace/SportsRagDemo/experiments")

EXCLUDE_EXACT_FOLDERS = ["clips", "germany_v_japan_clips"]

def gather_all_data(experiments_root):
    """
    Scans experiments folder, finds valid clips folders, pairs them with CSVs.
    """
    all_rows = []
    
    # 1. Find all folders
    if not os.path.exists(experiments_root):
        raise FileNotFoundError(f"Experiments folder not found at: {experiments_root}")

    candidates = [d for d in os.listdir(experiments_root) if os.path.isdir(os.path.join(experiments_root, d))]
    
    print(f"\nScanning {experiments_root}...")
    
    for folder_name in candidates:
        folder_path = os.path.join(experiments_root, folder_name)
        

        if folder_name in EXCLUDE_EXACT_FOLDERS:
            print(f"Skipping excluded folder: {folder_name}")
            continue
        if "germany_v_japan" in folder_name:
            print(f"Skipping excluded match: {folder_name}")
            continue

        print(f"Processing: {folder_name}...")
        
        match_name = folder_name.replace("clips_", "").replace("clips", "")
        
        csv_candidates = glob.glob(os.path.join(experiments_root, f"*{match_name}*.csv"))
        
        if not csv_candidates:
            print(f"  -> WARNING: No matching CSV found for {folder_name} (Looked for *{match_name}*.csv)")
            continue
            
        csv_path = csv_candidates[0] # Take the first match
        print(f"  -> Found CSV: {os.path.basename(csv_path)}")
        
        df = pd.read_csv(csv_path)
        
        id_to_filename = {}
        for f in os.listdir(folder_path):
            if f.endswith(".mp4") and "clip_" in f:
                try:
                    # Filename format: clip_050_... -> ID is 50
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
                
                all_rows.append({
                    "video_path": full_path,
                    "text": text_content
                })
                valid_count += 1
                
        print(f"  -> Added {valid_count} valid clips.")

    if len(all_rows) == 0:
        print("\nCRITICAL WARNING: No clips were added! Check if folder names match CSV names.")
        return pd.DataFrame() # Return empty to fail gracefully later

    print(f"Total Combined Dataset: {len(all_rows)} clips.\n")
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

class VideoCLIPLightning(pl.LightningModule):
    def __init__(self, model_name="ViT-B/16", lr=1e-5, max_epochs=10):
        super().__init__()
        self.save_hyperparameters()
        
        self.model, self.clip_preprocess = clip.load(model_name, device='cpu')
        self.model = self.model.float()
        
        # Freeze Vision backbone (optional, good for speed/memory)
        for param in self.model.visual.parameters():
            param.requires_grad = False
            
        self.loss_img = nn.CrossEntropyLoss()
        self.loss_txt = nn.CrossEntropyLoss()

    def forward(self, video, text):
        b, t, c, h, w = video.shape
        video = video.view(-1, c, h, w)
        
        frame_features = self.model.encode_image(video)
        frame_features = frame_features.view(b, t, -1)
        video_features = frame_features.mean(dim=1)
        text_features = self.model.encode_text(text)
        
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return video_features, text_features, self.model.logit_scale.exp()

    def compute_metrics(self, batch):
        video, text = batch
        video_emb, text_emb, logit_scale = self(video, text)
        
        logits = logit_scale * video_emb @ text_emb.t()
        labels = torch.arange(len(video), device=self.device)
        
        loss = (self.loss_img(logits, labels) + self.loss_txt(logits.t(), labels)) / 2
        
        # Accuracy
        pred = logits.argmax(dim=1)
        acc = (pred == labels).float().mean()
        
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.compute_metrics(batch)
        # prog_bar=True ensures these show up in the terminal progress bar
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.compute_metrics(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)
        return loss
        
    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.hparams.lr)

if __name__ == "__main__":
    print(f"--- Starting Training Script ---")
    
    full_df = gather_all_data(EXPERIMENTS_DIR)
    if len(full_df) == 0:
        raise RuntimeError("No data found. Exiting.")
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle
    val_cutoff = int(len(full_df) * VAL_SPLIT)
    val_df = full_df.iloc[:val_cutoff]
    train_df = full_df.iloc[val_cutoff:]
    
    print(f"Train Set: {len(train_df)} clips | Validation Set: {len(val_df)} clips")
    
    model = VideoCLIPLightning(model_name="ViT-B/16", lr=LEARNING_RATE, max_epochs=MAX_EPOCHS)
    
    train_ds = VideoTextDataset(train_df, model.clip_preprocess, clip.tokenize)
    val_ds = VideoTextDataset(val_df, model.clip_preprocess, clip.tokenize)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=4, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4, shuffle=False, pin_memory=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="multimatch-{epoch:02d}-{val_acc:.2f}",
        save_top_k=1,
        monitor="val_acc",
        mode="max" 
    )
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1, 
        strategy="auto",
        precision=16,
        max_epochs=MAX_EPOCHS,
        accumulate_grad_batches=4,
        callbacks=[checkpoint_callback],
        log_every_n_steps=5,
        enable_progress_bar=True
    )
    
    print("Starting Multi-Match Training...")
    trainer.fit(model, train_loader, val_loader)