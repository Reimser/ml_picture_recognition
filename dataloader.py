import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class HockeyDataset(Dataset):
    def __init__(self, labels_csv, frames_root, transform=None, frames_per_clip=100):
        # CSV-Datei mit Clipnamen und zugehörigen Labels laden
        self.labels_df = pd.read_csv(labels_csv)
        self.frames_root = frames_root
        self.frames_per_clip = frames_per_clip

        # Wenn kein Transform übergeben wurde, verwende Standard mit Augmentation (für Training)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),                         # Einheitliche Größe
                transforms.RandomHorizontalFlip(),                    # Spiegelung für Varianz
                transforms.RandomRotation(degrees=5),                 # kleine Rotation
                transforms.ColorJitter(brightness=0.2, contrast=0.2), # Helligkeit/Kontrast
                transforms.ToTensor(),                                # PIL -> Tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406],      # Normalisierung wie ImageNet
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform  # z. B. val/test ohne Augmentation

        # Textlabels in numerische Labels umwandeln
        self.label_map = {
            'Check': 0,
            'Neutral': 1,
            'Schuss': 2,
            'Tor': 3
        }

    def __len__(self):
        # Anzahl der Clips in der CSV-Datei
        return len(self.labels_df)

    def __getitem__(self, idx):
        # Hole Clipname und Label aus CSV
        clip_name = self.labels_df.iloc[idx, 0]
        label_str = self.labels_df.iloc[idx, 1]

        # Umwandlung des Textlabels in Integer-Label
        label = self.label_map[label_str]

        # Pfad zum Verzeichnis mit Frames dieses Clips
        frames_dir = os.path.join(self.frames_root, clip_name.replace('.mp4', ''))

        # Lade alle .jpg-Dateien und sortiere sie
        try:
            frames = sorted([
                os.path.join(frames_dir, f)
                for f in os.listdir(frames_dir)
                if f.endswith('.jpg')
            ])
        except FileNotFoundError:
            raise FileNotFoundError(f"Frame-Ordner nicht gefunden: {frames_dir}")

        # Wenn keine Frames vorhanden sind, Fehlermeldung
        if not frames:
            raise ValueError(f"Keine Frames im Verzeichnis: {frames_dir}")

        # Wenn zu viele Frames: schneide von hinten
        # Wenn zu wenige: dupliziere letzten Frame vorne
        if len(frames) >= self.frames_per_clip:
            frames = frames[-self.frames_per_clip:]
        else:
            last_frame = frames[-1]
            while len(frames) < self.frames_per_clip:
                frames.insert(0, last_frame)

        # Lade und transformiere jedes Frame
        images = []
        for frame_path in frames:
            image = Image.open(frame_path).convert('RGB')  # in 3-Kanal RGB konvertieren
            image = self.transform(image)                  # Transform anwenden (z. B. Resize, Tensor)
            images.append(image)

        # Stack zu Tensor der Form [T, 3, 224, 224]
        video_tensor = torch.stack(images)

        return video_tensor, label