import torch
import pytorch_lightning as pl
import clip
import os

CKPT_PATH = "/home/richard/Desktop/workspace/SportsRagDemo/backend/video_clip/checkpoints/standard-clip-epoch=14-val_acc=0.93.ckpt"
OUTPUT_PATH = "/home/richard/Desktop/workspace/SportsRagDemo/backend/video_clip/checkpoints/best_model_cpu.pt"

class VideoCLIPLightning(pl.LightningModule):
    def __init__(self, model_name="ViT-B/16"):
        super().__init__()
        self.model, _ = clip.load(model_name, device='cpu', jit=False)
        self.model = self.model.float()

def main():
    print(f"Loading checkpoint from: {CKPT_PATH}")
    
    if not os.path.exists(CKPT_PATH):
        print(f"Error: File not found at {CKPT_PATH}")
        return
    pl_model = VideoCLIPLightning.load_from_checkpoint(CKPT_PATH, map_location="cpu")
    
    raw_state_dict = pl_model.state_dict()
    print(f"Loaded state dict with {len(raw_state_dict)} keys.")

    clean_state_dict = {}
    for key, value in raw_state_dict.items():
        if key.startswith("model."):
            new_key = key.replace("model.", "")
            clean_state_dict[new_key] = value
        else:
            clean_state_dict[key] = value

    torch.save(clean_state_dict, OUTPUT_PATH)

if __name__ == "__main__":
    main()