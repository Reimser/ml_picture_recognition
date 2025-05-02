import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class HockeyDataset(Dataset):
    def __init__(self, labels_csv, frames_root, transform=None, frames_per_clip=100):
        # CSV-Datei mit Clip-Namen und Labels einlesen
        self.labels_df = pd.read_csv(labels_csv)
        self.frames_root = frames_root
        self.frames_per_clip = frames_per_clip

        # Falls keine Transformation angegeben, Standard verwenden
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

        # Mapping von Textlabel zu Integer
        self.label_map = {
            'Check': 0,
            'Neutral': 1,
            'Schuss': 2,
            'Tor': 3
        }

    def __len__(self):
        # Anzahl der Zeilen in der CSV = Anzahl der Clips
        return len(self.labels_df)

    def __getitem__(self, idx):
        clip_name = self.labels_df.iloc[idx, 0]
        label_str = self.labels_df.iloc[idx, 1]

        # Textlabel in Zahl umwandeln
        label = self.label_map[label_str]

        # Ordnerpfad zu den Frames des Clips
        frames_dir = os.path.join(self.frames_root, clip_name.replace('.mp4', ''))

        # Liste aller Frame-Dateien (nur .jpg) sortieren
        try:
            frames = sorted([
                os.path.join(frames_dir, f)
                for f in os.listdir(frames_dir)
                if f.endswith('.jpg')
            ])
        except FileNotFoundError:
            raise FileNotFoundError(f"Frame-Ordner nicht gefunden: {frames_dir}")

        # Wenn keine Frames gefunden wurden, Fehler ausgeben
        if not frames:
            raise ValueError(f"Keine Frames im Verzeichnis: {frames_dir}")

        # Letzte N Frames verwenden oder vorne mit letztem Frame auffüllen
        if len(frames) >= self.frames_per_clip:
            frames = frames[-self.frames_per_clip:]
        else:
            last_frame = frames[-1]
            while len(frames) < self.frames_per_clip:
                frames.insert(0, last_frame)

        # Bilder laden und transformieren
        images = []
        for frame_path in frames:
            image = Image.open(frame_path).convert('RGB')
            image = self.transform(image)
            images.append(image)

        # Stapel von Bildern ergibt Tensor: [T, 3, 224, 224]
        video_tensor = torch.stack(images)

        return video_tensor, label
