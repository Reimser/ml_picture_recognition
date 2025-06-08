import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# --- Training & Validierung ---
class HockeyDataset(Dataset):
    def __init__(self, csv_file, frames_root, transform=None, frames_per_clip=100):
        self.labels_df = pd.read_csv(csv_file)
        self.frames_root = frames_root
        self.frames_per_clip = frames_per_clip

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        clip_name = row['clip_name']

        check = row.get('Check', 0)
        neutral = row.get('Neutral', 0)
        schuss = row.get('Schuss', 0)
        tor = row.get('Tor', 0)

        # Neutralität deaktivieren, wenn andere Klassen aktiv sind
        if check == 1 or schuss == 1 or tor == 1:
            neutral = 0

        label = torch.tensor([check, neutral, schuss, tor], dtype=torch.float32)

        frames_dir = os.path.join(self.frames_root, clip_name.replace('.mp4', ''))

        try:
            frames = sorted([
                os.path.join(frames_dir, f)
                for f in os.listdir(frames_dir)
                if f.endswith('.jpg')
            ])
        except FileNotFoundError:
            raise FileNotFoundError(f"Ordner nicht gefunden: {frames_dir}")

        if not frames:
            raise ValueError(f"Keine Frames in Verzeichnis: {frames_dir}")

        if len(frames) >= self.frames_per_clip:
            frames = frames[-self.frames_per_clip:]
        else:
            last_frame = frames[-1]
            while len(frames) < self.frames_per_clip:
                frames.insert(0, last_frame)

        images = [self.transform(Image.open(f).convert("RGB")) for f in frames]
        video_tensor = torch.stack(images)

        return video_tensor, label


# --- Test-Daten ohne Labels ---
class HockeyTestDataset(Dataset):
    def __init__(self, csv_file, frames_root, transform=None, frames_per_clip=100):
        self.df = pd.read_csv(csv_file)
        self.frames_root = frames_root
        self.frames_per_clip = frames_per_clip

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        clip_name = self.df.iloc[idx]['clip_name']
        frames_dir = os.path.join(self.frames_root, clip_name.replace('.mp4', ''))

        try:
            frames = sorted([
                os.path.join(frames_dir, f)
                for f in os.listdir(frames_dir)
                if f.endswith('.jpg')
            ])
        except FileNotFoundError:
            raise FileNotFoundError(f"Ordner nicht gefunden: {frames_dir}")

        if not frames:
            raise ValueError(f"Keine Frames in Verzeichnis: {frames_dir}")

        if len(frames) >= self.frames_per_clip:
            frames = frames[-self.frames_per_clip:]
        else:
            last_frame = frames[-1]
            while len(frames) < self.frames_per_clip:
                frames.insert(0, last_frame)

        images = [self.transform(Image.open(f).convert("RGB")) for f in frames]
        video_tensor = torch.stack(images)

        return video_tensor, clip_name
