import torch
import torch.nn as nn
import torchvision.models as models

class CNNEncoder(nn.Module):
    def __init__(self, embed_size=128):
        super(CNNEncoder, self).__init__()
        # Kleines CNN bauen (ResNet-18 Backbone nehmen)
        base_model = models.resnet18(pretrained=True)
        # Entferne die letzte Schicht (Classification Layer)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        self.fc = nn.Linear(base_model.fc.in_features, embed_size)

    def forward(self, x):
        # x: (Batch*10, 3, 224, 224)
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)  # Flach machen
        features = self.fc(features)  # In eingebetteten Raum bringen
        return features  # (Batch*10, 128)

class ActionLSTM(nn.Module):
    def __init__(self, embed_size=128, hidden_size=256, num_classes=4, num_layers=1):
        super(ActionLSTM, self).__init__()
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, features):
        # features: (Batch, 10, 128)
        lstm_out, _ = self.lstm(features)
        # Nur den letzten Zeitschritt nehmen
        final_output = lstm_out[:, -1, :]  # (Batch, hidden_size)
        output = self.fc(final_output)
        return output  # (Batch, num_classes)

class HockeyActionModel(nn.Module):
    def __init__(self, embed_size=128, hidden_size=256, num_classes=4, num_layers=1):
        super(HockeyActionModel, self).__init__()
        self.encoder = CNNEncoder(embed_size)
        self.lstm = ActionLSTM(embed_size, hidden_size, num_classes, num_layers)

    def forward(self, frames_batch):
        # frames_batch: (Batch, 100, 3, 224, 224)
        batch_size, seq_len, C, H, W = frames_batch.size()

        # Erst alle Frames einzeln durch CNN-Encoder schicken
        frames_batch = frames_batch.view(batch_size * seq_len, C, H, W)
        features = self.encoder(frames_batch)

        # Features wieder in (Batch, Seq_len, Feature_dim) bringen
        features = features.view(batch_size, seq_len, -1)

        # Sequenz durch LSTM
        output = self.lstm(features)

        return output
