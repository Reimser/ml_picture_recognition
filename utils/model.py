import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# CNN-Encoder: Extrahiert visuelle Merkmale aus den einzelnen Frames
class CNNEncoder(nn.Module):
    def __init__(self, embed_size=128, fine_tune_last_block=True):
        super(CNNEncoder, self).__init__()

        # Lade ein vortrainiertes ResNet-18 Modell (auf ImageNet trainiert)
        base_model = models.resnet18(pretrained=True)

        # Alle Parameter einfrieren (kein Training), um als Feature-Extractor zu nutzen
        for param in base_model.parameters():
            param.requires_grad = False

        # Optional: Nur den letzten Block "layer4" freigeben, um Fine-Tuning zu ermöglichen
        if fine_tune_last_block:
            for param in base_model.layer4.parameters():
                param.requires_grad = True

        # Entferne das letzte Klassifikations-Layer und behalte nur die Feature-Extraktion
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])  # Output: [B, 512, 1, 1]

        # Reduktion der Feature-Dimension auf den gewünschten embed_size
        self.fc = nn.Linear(base_model.fc.in_features, embed_size)

    def forward(self, x):
        # Input: x = [B*T, 3, 224, 224]
        features = self.feature_extractor(x)         # Output: [B*T, 512, 1, 1]
        features = features.view(features.size(0), -1)  # Flatten: [B*T, 512]
        features = self.fc(features)                 # Lineare Projektion: [B*T, embed_size]
        return features


# LSTM-Modul zur Modellierung der zeitlichen Abfolge von Frame-Features
class ActionLSTM(nn.Module):
    def __init__(self, embed_size=128, hidden_size=256, num_classes=4, num_layers=1, dropout=0.2):
        super(ActionLSTM, self).__init__()

        # LSTM verarbeitet Sequenzen von Frame-Embeddings
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)

        # Klassifikationskopf: letztes LSTM-Output in Klassenwahrscheinlichkeit umwandeln
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, features):
        # Input: features = [B, T, embed_size]
        lstm_out, _ = self.lstm(features)          # Output: [B, T, hidden_size]
        final_output = lstm_out[:, -1, :]          # Nur letzter Zeitschritt: [B, hidden_size]
        return self.classifier(final_output)       # Output: [B, num_classes]


# Gesamtes Modell: Kombination aus CNN-Encoder und LSTM
class HockeyActionModel(nn.Module):
    def __init__(self, embed_size=128, hidden_size=256, num_classes=4, num_layers=1):
        super(HockeyActionModel, self).__init__()
        self.encoder = CNNEncoder(embed_size)
        self.lstm = ActionLSTM(embed_size, hidden_size, num_classes, num_layers)

    def forward(self, frames_batch):
        # Input: frames_batch = [B, T, 3, 224, 224]
        B, T, C, H, W = frames_batch.shape

        # Reshape zu [B*T, 3, 224, 224] für CNN
        frames_batch = frames_batch.view(B * T, C, H, W)

        # CNN: extrahiere visuelle Merkmale
        features = self.encoder(frames_batch)  # [B*T, embed_size]

        # Reshape zurück zu Sequenz: [B, T, embed_size]
        features = features.view(B, T, -1)

        # LSTM für zeitliche Analyse und Klassifikation
        return self.lstm(features)  # [B, num_classes]
