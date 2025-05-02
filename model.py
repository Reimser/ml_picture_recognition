import torch
import torch.nn as nn
import torchvision.models as models

# CNN-Encoder (ResNet-18 als Feature-Extraktor)
class CNNEncoder(nn.Module):
    def __init__(self, embed_size=128):
        super(CNNEncoder, self).__init__()

        # Vortrainiertes ResNet-18 Modell laden
        base_model = models.resnet18(pretrained=True)
        for param in base_model.parameters():
            param.requires_grad = False  # Feature-Extractor einfrieren (optional)

        # Entferne das letzte Klassifikations-Linear-Layer
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])  # Output: [B, 512, 1, 1]
        self.fc = nn.Linear(base_model.fc.in_features, embed_size)

    def forward(self, x):
        # x: [B*T, 3, 224, 224]
        features = self.feature_extractor(x)  # [B*T, 512, 1, 1]
        features = features.view(features.size(0), -1)  # [B*T, 512]
        features = self.fc(features)  # [B*T, embed_size]
        return features


# LSTM-Modul für zeitliche Abhängigkeiten
class ActionLSTM(nn.Module):
    def __init__(self, embed_size=128, hidden_size=256, num_classes=4, num_layers=1):
        super(ActionLSTM, self).__init__()
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, features):
        # features: [B, T, embed_size]
        lstm_out, _ = self.lstm(features)  # [B, T, hidden_size]
        final_output = lstm_out[:, -1, :]  # nur letzter Zeitschritt: [B, hidden_size]
        return self.fc(final_output)      # [B, num_classes]


# Kombiniertes Gesamtmodell
class HockeyActionModel(nn.Module):
    def __init__(self, embed_size=128, hidden_size=256, num_classes=4, num_layers=1):
        super(HockeyActionModel, self).__init__()
        self.encoder = CNNEncoder(embed_size)
        self.lstm = ActionLSTM(embed_size, hidden_size, num_classes, num_layers)

    def forward(self, frames_batch):
        # frames_batch: [B, T, 3, 224, 224]
        B, T, C, H, W = frames_batch.shape

        # Umformen zu [B*T, 3, 224, 224]
        frames_batch = frames_batch.view(B * T, C, H, W)

        # Feature-Extraktion
        features = self.encoder(frames_batch)  # [B*T, embed_size]

        # Zurückformen in [B, T, embed_size]
        features = features.view(B, T, -1)

        # Klassifikation durch LSTM
        return self.lstm(features)  # [B, num_classes]
