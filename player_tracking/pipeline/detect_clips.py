import os
import warnings
import cv2
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import timm
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--video_path", type=str, default="data/4.mp4")
args = parser.parse_args()

# Suppress NNPACK warning (harmless - just means CPU acceleration not available)
warnings.filterwarnings("ignore", message=".*NNPACK.*")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model("efficientnet_b0", pretrained=True)
model.classifier = nn.Sequential(
    nn.Linear(model.classifier.in_features, 1),
    nn.Sigmoid()
)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()  # Set to evaluation mode

model.to(device)

video_path = args.video_path
video = cv2.VideoCapture(video_path)
frame_rate = video.get(cv2.CAP_PROP_FPS)
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frames_per_second = 3
every_n_frames = frame_rate // frames_per_second
first_frames = None

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_frame(raw_frame):
    frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    frame = transform(frame)
    return frame

timestamps = []
threshold = 0.5
on_in_a_row = 0
off_in_a_row = 0
row_threshold = 3
on = False

for frame_idx in tqdm(range(total_frames if first_frames is None else first_frames)):
    ret, frame = video.read()
    if not ret:
        break

    if frame_idx % every_n_frames != 0:
        continue

    with torch.no_grad():
        output = model(load_frame(frame).unsqueeze(0).to(device))
        probability = output.item()
        if probability > threshold:
            on_in_a_row += 1
            off_in_a_row = 0
        else:
            off_in_a_row += 1
            on_in_a_row = 0

    if not on and on_in_a_row >= row_threshold:
        timestamps.append((frame_idx/frame_rate, 1))
        on_in_a_row = 0
        on = True
    elif on and off_in_a_row >= row_threshold:
        timestamps.append((frame_idx/frame_rate, 0))
        off_in_a_row = 0
        on = False

print(timestamps)
json.dump(timestamps, open(f"{args.video_path.split('/')[-1].split('.')[0]}_timestamps.json", "w"))