import torch
import torch.nn as nn
import cv2
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import open_clip
import torch.nn.functional as F
from tqdm import tqdm

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
        
        norm_video_embedding = video_embedding / (torch.norm(video_embedding, dim=1, keepdim=True) + 1e-8)
        norm_text_embedding = text_embedding / (torch.norm(text_embedding, dim=1, keepdim=True) + 1e-8)
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

        frames = torch.randn(1, 3, 224, 224)
        text_description = random.choice(["This is a test", "This is a test 2", "This is a test 3"])

        return frames, text_description


device = "cuda" if torch.cuda.is_available() else "cpu"
#model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
#print("hello", tokenizer(["hi"]))
# print(preprocess) --> shows that mean/std are     Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

# dataset = EmbeddingDataset(["clip.mp4"], ["sample_text.txt"], tokenizer)
# model = EmbeddingModel().to(device)

# X = dataset[0][0].unsqueeze(0).repeat(2, 1, 1, 1, 1) + torch.randn(2, 1, 3, 224, 224)
# text = [dataset[0][1], "hi bro"]

# print(text)

# video = model.encode_video(X)

# text_embedding = model.encode_text(text[0])
# loss = model.compute_loss(X, text)

# print(video)
# print(text_embedding)
# print(loss)

if __name__ == "__main__":
    batch_size = 16
    test_batch_size = 16
    num_workers = 0
    lr = 1e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = 50
    freeze_clip = False

    patience = 10
    best_loss = float('inf')
    best_epoch = 0
    early_stop_count = 0

    model_name = "ViT-B-32"
    model = EmbeddingModel(model_name).to(device)
    tokenizer = open_clip.get_tokenizer(model_name)

    if freeze_clip:
        for param in model.model.parameters():
            param.requires_grad = False
        for param in model.fusion.parameters():
            param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_video_paths = ["clip.mp4"]*128
    train_text_paths = ["sample_text.txt"]*128
    val_video_paths = ["clip.mp4"]*64
    val_text_paths = ["sample_text.txt"]*64

    train_dataset = EmbeddingDataset(train_video_paths, train_text_paths, tokenizer)
    val_dataset = EmbeddingDataset(val_video_paths, val_text_paths, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}...", total=len(train_loader))
        total_loss = 0
        n = 0
        for i, batch in enumerate(pbar):
            video, text = batch
            loss = model.compute_loss(video, text)
            total_loss += loss.item() * video.shape[0]
            n += video.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Loss: {total_loss / n}")

        print(f"Train loss {epoch+1}: {total_loss / n}")

        model.eval()
        pbar = tqdm(val_loader, desc="Evaluating")
        total_loss = 0
        n = 0
        for i, batch in enumerate(pbar):
            video, text = batch
            loss = model.compute_loss(video, text)
            total_loss += loss.item() * video.shape[0]
            n += video.shape[0]
            pbar.set_description(f"Loss: {total_loss / n}")
        print(f"Val loss {epoch+1}: {total_loss / n}")

        if total_loss / n < best_loss:
            best_loss = total_loss / n
            best_epoch = epoch
            early_stop_count = 0
            torch.save(model.state_dict(), "best_model.pth")
            with open("best_epoch.txt", "w") as f:
                f.write(str(epoch+1) + ", " + str(total_loss / n))
        else:
            early_stop_count += 1

        if early_stop_count >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    torch.save(model.state_dict(), "last_model.pth")

    if epoch != best_epoch:
        print(f"Best epoch: {best_epoch+1}, Best loss: {best_loss}")