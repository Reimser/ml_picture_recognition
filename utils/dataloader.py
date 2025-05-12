import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class HockeyDataset(Dataset):
    def __init__(self, labels_csv, frames_root, transform=None, frames_per_clip=100):
        # CSV-Datei mit clip_name + Multi-Label-Spalten (Check, Neutral, Schuss, Tor) laden
        self.labels_df = pd.read_csv(labels_csv)
        self.frames_root = frames_root
        self.frames_per_clip = frames_per_clip

        # Transformationspipeline festlegen
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        # Clipnamen laden
        clip_name = self.labels_df.iloc[idx]['clip_name']

        # Multi-Label-Vektor aus den vier Spalten extrahieren
        label_vec = self.labels_df.iloc[idx][['Check', 'Neutral', 'Schuss', 'Tor']].values.astype(float)
        label = torch.tensor(label_vec, dtype=torch.float32)

        # Pfad zu den Frames dieses Clips
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

        # Auf gewünschte Frameanzahl zuschneiden oder auffüllen
        if len(frames) >= self.frames_per_clip:
            frames = frames[-self.frames_per_clip:]
        else:
            last_frame = frames[-1]
            while len(frames) < self.frames_per_clip:
                frames.insert(0, last_frame)

        # Frames laden und transformieren
        images = []
        for frame_path in frames:
            image = Image.open(frame_path).convert("RGB")
            image = self.transform(image)
            images.append(image)

        # [T, 3, 224, 224]
        video_tensor = torch.stack(images)

        return video_tensor, label
