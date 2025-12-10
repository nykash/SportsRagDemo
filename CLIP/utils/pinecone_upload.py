import os
import sys
import torch
import clip
import pandas as pd
import numpy as np
import glob
from PIL import Image
from decord import VideoReader, cpu
from tqdm import tqdm
import pytorch_lightning as pl

current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from clients.pinecone_client import PineconeClient

CHECKPOINT_PATH = "/home/richard/Desktop/workspace/SportsRagDemo/backend/video_clip/checkpoints/standard-clip-epoch=14-val_acc=0.93.ckpt"
EXPERIMENTS_DIR = "/home/richard/Desktop/workspace/SportsRagDemo/experiments"
INDEX_NAME = "sports-rag-clip" 
FRAMES_PER_CLIP = 16
BATCH_SIZE = 50

MATCH_ID_MAP = {
    "brazil_v_usa": 1,
    "usa_v_brazil": 1,
    "usa_v_canada": 2,
    "canada_v_usa": 2,
    "usa_v_germany": 3,
    "germany_v_usa": 3,
    "france_v_japan": 4, 
    "frace_v_japan": 4, # Typo handling
    "japan_v_france": 4,
    "japan_v_germany": 5,
    "germany_v_japan": 5,
    "brazil_v_japan": 6, # <--- Added this!
    "japan_v_brazil": 6
}
# Model Definition
class VideoCLIPLightning(pl.LightningModule):
    def __init__(self, model_name="ViT-B/16"):
        super().__init__()
        self.model, self.clip_preprocess = clip.load(model_name, device='cpu', jit=False)
        self.model = self.model.float()

    def forward(self, video_tensor):
        b, t, c, h, w = video_tensor.shape
        video_flat = video_tensor.view(-1, c, h, w)
        with torch.no_grad():
            frame_features = self.model.encode_image(video_flat)
        video_seq = frame_features.view(b, t, -1)
        video_emb = video_seq.mean(dim=1)
        video_emb = video_emb / video_emb.norm(dim=-1, keepdim=True)
        return video_emb

def load_model(ckpt_path):
    print(f"Loading model from {ckpt_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VideoCLIPLightning.load_from_checkpoint(ckpt_path)
    model.eval()
    model.to(device)
    return model, device

# Data Processing
def process_video(video_path, preprocess):
    if not os.path.exists(video_path): return None
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        if len(vr) < FRAMES_PER_CLIP: return None
        indices = np.linspace(0, len(vr) - 1, FRAMES_PER_CLIP).astype(int)
        frames = vr.get_batch(indices).asnumpy()
        frames_tensor = torch.stack([preprocess(Image.fromarray(f)) for f in frames])
        return frames_tensor.unsqueeze(0)
    except Exception:
        return None

def gather_data(root_dir):
    all_rows = []
    folders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    for folder in folders:
        if folder not in ["clips", "germany_v_japan_clips"] and "germany_v_japan" not in folder: continue
        
        folder_path = os.path.join(root_dir, folder)

        # Remove standard prefixes/suffixes
        clean_name = folder.replace("clips_", "").replace("clips", "")
        # Strip trailing/leading underscores (Fixes 'brazil_v_usa_')
        match_name = clean_name.strip("_")
        
        match_id = MATCH_ID_MAP.get(match_name)
        if match_id is None:
            print(f"Warning: No ID found for match '{match_name}' (derived from '{folder}'). Skipping.")
            continue # Skip this folder if we can't identify the match

        csv_files = glob.glob(os.path.join(root_dir, f"*{match_name}*.csv"))
        if not csv_files: 
            print(f"  -> No CSV found for {match_name}")
            continue
        
        df = pd.read_csv(csv_files[0])
        
        def get_col(row, candidates, default=0.0):
            for col in candidates:
                if col in row:
                    return float(row[col])
            return default

        id_to_file = {}
        for f in os.listdir(folder_path):
            if f.endswith(".mp4") and "clip_" in f:
                try:
                    parts = f.split('_')
                    id_to_file[int(parts[1])] = f
                except: continue
        
        valid_count = 0
        for idx, row in df.iterrows():
            if idx in id_to_file:
                start_time = get_col(row, ['start', 'start_time', 'Start'], 0.0)
                end_time = get_col(row, ['end', 'end_time', 'End'], 0.0)

                all_rows.append({
                    "id": f"{match_name}_{idx}",
                    "video_path": os.path.join(folder_path, id_to_file[idx]),
                    "text": str(row.get('text', row.get('transcript', ''))),
                    "video_id": match_id,
                    "start_time": start_time,
                    "end_time": end_time
                })
                valid_count += 1
        # Optional: Print status
        # print(f"  -> '{match_name}': Found {valid_count} clips. (ID: {match_id})")

    return pd.DataFrame(all_rows)

# Main Logic
def main():
    client = PineconeClient(index_name=INDEX_NAME)
    client.create_vector_index(dimension=512)

    model, device = load_model(CHECKPOINT_PATH)
    df = gather_data(EXPERIMENTS_DIR)
    
    if len(df) == 0:
        print("No clips found! Check your paths and folder names.")
        return

    print(f"Found {len(df)} clips to process.")
    
    vectors_to_upsert = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        video_tensor = process_video(row['video_path'], model.clip_preprocess)
        if video_tensor is None: continue
        
        video_tensor = video_tensor.to(device)
        embedding = model(video_tensor)
        
        vectors_to_upsert.append((
            row['id'],
            embedding.cpu().numpy().flatten().tolist(),
            {
                "text": row['text'],
                "video_id": int(row['video_id']),
                "video_path": row['video_path'],
                "start_time": row['start_time'],
                "end_time": row['end_time']
            }
        ))
        
        if len(vectors_to_upsert) >= BATCH_SIZE:
            client.upsert_vectors(vectors_to_upsert)
            vectors_to_upsert = []

    if vectors_to_upsert:
        client.upsert_vectors(vectors_to_upsert)
        
    print(f"Upload Complete to index: {INDEX_NAME}")

if __name__ == "__main__":
    main()