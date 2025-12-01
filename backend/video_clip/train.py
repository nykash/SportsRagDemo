import torch
import torch.nn as nn
import cv2
import random
import numpy as np
from torch.utils.data import Dataset
import open_clip
import torch.nn.functional as F

def read_frame(video: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = video.read()
    if not ret:
        raise ValueError(f"Could not read frame {frame_index}")

    return frame

class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            input_size=512,
            hidden_size=512,
            num_layers=2,
            batch_first=True
        )

        self.mlp = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

    def forward(self, x):               # x: B x seq x 1024
        _, h_n = self.gru(x)           # h_n: num_layers x B x 512
        h_last = h_n[-1]               # B x 512   (top layer's final hidden)
        return self.mlp(h_last)        # B x 1024

class EmbeddingModel(nn.Module):
    def __init__(self, model_name="ViT-B-32"):
        super(EmbeddingModel, self).__init__()
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained='openai')
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.fusion = Fusion()

    def encode_video(self, x):
        og_shape = x.shape
        x = x.view(-1, *x.shape[2:]).contiguous()
        x = self.model.encode_image(x)
        x = x.view(og_shape[0], og_shape[1], *x.shape[1:]).contiguous()

        print("x shape", x.shape)

        x = self.fusion(x)
        return x
    
    def encode_text(self, x):
        x = self.tokenizer(x)
        x = self.model.encode_text(x)
        return x

    def compute_loss(self, video, text):
        video_embedding = self.encode_video(video)
        text_embedding = self.encode_text(text)

        # clip loss
        
        norm_video_embedding = video_embedding / (torch.norm(video_embedding, dim=1) + 1e-8)
        norm_text_embedding = text_embedding / (torch.norm(text_embedding, dim=1) + 1e-8)
        logits = norm_video_embedding @ norm_text_embedding.T
        labels = torch.arange(video_embedding.shape[0]).to(video_embedding.device)

        loss_i = F.cross_entropy(logits, labels)           
        loss_t = F.cross_entropy(logits.t(), labels)

        loss = (loss_i + loss_t) / 2

        return loss

class EmbeddingDataset(Dataset):
    def __init__(self, video_paths, text_paths, tokenizer, keep_frames=20):
        self.video_paths = video_paths # .mp4 files
        self.text_paths = text_paths # .txt files
        self.keep_frames = keep_frames
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        text_path = self.text_paths[idx]

        with open(text_path, 'r') as f:
            text_description = f.read()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        # pick a random frame from every keep_frames frames
        for i in range(total_frames // self.keep_frames):
            frame = read_frame(cap, random.randint(0, self.keep_frames - 1) + i * self.keep_frames)
            frames.append(cv2.resize(frame, (224, 224)))
        cap.release()

        # convert frames to numpy arrays
        frames = np.array(frames)

        # frames should be (frames_count x 3 x height x width)
        frames = frames.transpose(0, 3, 1, 2)

        # normalize frames with mean and std as in pretrained open_clip model mean = [0.48145466, 0.4578275, 0.40821073] and std = [0.26862954, 0.26130258, 0.27577711]
        mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(1, 3, 1, 1)
        std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(1, 3, 1, 1)
        frames = (frames - mean) / std

        frames = torch.tensor(frames, dtype=torch.float32)

        return frames, text_description


device = "cuda" if torch.cuda.is_available() else "cpu"
#model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
#print("hello", tokenizer(["hi"]))
# print(preprocess) --> shows that mean/std are     Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

dataset = EmbeddingDataset(["clip.mp4"], ["sample_text.txt"], tokenizer)
model = EmbeddingModel().to(device)

video = model.encode_video(torch.unsqueeze(dataset[0][0], 0))

text = model.encode_text([dataset[0][1]])
loss = model.compute_loss(torch.unsqueeze(dataset[0][0], 0), [dataset[0][1]])
print(loss)

