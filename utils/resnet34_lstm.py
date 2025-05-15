import torch
import torch.nn as nn
import torchvision.models as models

class ResNet34_LSTM(nn.Module):
    def __init__(self, hidden_dim, num_classes, lstm_layers=1, bidirectional=False):
        super(ResNet34_LSTM, self).__init__()

        # Lade vortrainiertes ResNet34-Modell
        resnet34 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        
        # Entferne die letzte FC-Schicht (nur Feature-Extractor)
        modules = list(resnet34.children())[:-1]  # alles außer letzter Linear-Layer
        self.resnet34 = nn.Sequential(*modules)
        
        # Größe der ResNet34-Featuremaps (meist 512)
        self.feature_dim = resnet34.fc.in_features

        # LSTM, um zeitliche Dynamik der Frames zu lernen
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Klassifikationskopf
        self.fc = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, x):
        batch_size, seq_length, C, H, W = x.shape

        # Initialisiere Output-Liste
        features = []

        for t in range(seq_length):
            # Extrahiere Features für jedes Frame
            frame = x[:, t, :, :, :]
            feature = self.resnet34(frame)  # Output: (B, 512, 1, 1)
            feature = feature.view(batch_size, -1)  # (B, 512)
            features.append(feature)

        # Stapeln der Features entlang Zeitachse
        features = torch.stack(features, dim=1)  # (B, T, 512)

        # LSTM drüberlaufen lassen
        lstm_out, _ = self.lstm(features)

        # Nur den letzten Zeitschritt nehmen
        out = lstm_out[:, -1, :]  # (B, hidden_dim)

        out = self.fc(out)  # (B, num_classes)

        return out
