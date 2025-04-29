import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class HockeyDataset(Dataset):
    def __init__(self, labels_csv, frames_root, transform=None, frames_per_clip=10):
        self.labels_df = pd.read_csv(labels_csv)
        self.frames_root = frames_root
        self.transform = transform
        self.frames_per_clip = frames_per_clip

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        clip_name = self.labels_df.iloc[idx, 0]
        label = self.labels_df.iloc[idx, 1]

        frames_dir = os.path.join(self.frames_root, clip_name.replace('.mp4', ''))

        frames = sorted([
            os.path.join(frames_dir, f)
            for f in os.listdir(frames_dir)
            if f.endswith('.jpg')
        ])

        # Nur die ersten N Frames nehmen (z.B. 10)
        frames = frames[:self.frames_per_clip]

        images = []
        for frame_path in frames:
            image = Image.open(frame_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)

        return images, label
