import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.models.video as video_models

# ---------------------------
# CNN-Encoder für ResNet18
# ---------------------------
class CNNEncoderResNet18(nn.Module):
    def __init__(self, embed_size=256, fine_tune_last_block=True):
        super(CNNEncoderResNet18, self).__init__()

        base_model = models.resnet18(pretrained=True)

        for param in base_model.parameters():
            param.requires_grad = False

        if fine_tune_last_block:
            for param in base_model.layer3.parameters():
                param.requires_grad = True
            for param in base_model.layer4.parameters():
                param.requires_grad = True

        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        self.fc = nn.Linear(base_model.fc.in_features, embed_size)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features

# ---------------------------
# CNN-Encoder für ResNet34
# ---------------------------
class CNNEncoderResNet34(nn.Module):
    def __init__(self, embed_size=256, fine_tune_last_block=True):
        super(CNNEncoderResNet34, self).__init__()

        base_model = models.resnet34(pretrained=True)

        for param in base_model.parameters():
            param.requires_grad = False

        if fine_tune_last_block:
            for param in base_model.layer3.parameters():
                param.requires_grad = True
            for param in base_model.layer4.parameters():
                param.requires_grad = True

        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        self.fc = nn.Linear(base_model.fc.in_features, embed_size)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features

# ---------------------------
# LSTM-Modul
# ---------------------------
class ActionLSTM(nn.Module):
    def __init__(self, embed_size=256, hidden_size=256, num_classes=4, num_layers=1, dropout=0.2):
        super(ActionLSTM, self).__init__()
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, features):
        lstm_out, _ = self.lstm(features)
        final_output = lstm_out[:, -1, :]
        return self.classifier(final_output)

# ---------------------------
# Komplette Modelle
# ---------------------------
class HockeyActionModelResNet18(nn.Module):
    def __init__(self, embed_size=256, hidden_size=256, num_classes=4, num_layers=1):
        super(HockeyActionModelResNet18, self).__init__()
        self.encoder = CNNEncoderResNet18(embed_size)
        self.lstm = ActionLSTM(embed_size, hidden_size, num_classes, num_layers)

    def forward(self, frames_batch):
        B, T, C, H, W = frames_batch.shape
        frames_batch = frames_batch.view(B * T, C, H, W)
        features = self.encoder(frames_batch)
        features = features.view(B, T, -1)
        return self.lstm(features)

class HockeyActionModelResNet34(nn.Module):
    def __init__(self, embed_size=256, hidden_size=256, num_classes=4, num_layers=1):
        super(HockeyActionModelResNet34, self).__init__()
        self.encoder = CNNEncoderResNet34(embed_size)
        self.lstm = ActionLSTM(embed_size, hidden_size, num_classes, num_layers)

    def forward(self, frames_batch):
        B, T, C, H, W = frames_batch.shape
        frames_batch = frames_batch.view(B * T, C, H, W)
        features = self.encoder(frames_batch)
        features = features.view(B, T, -1)
        return self.lstm(features)

# --- R3D-18 Modell ---
class R3D18Model(nn.Module):
    def __init__(self, num_classes=4):
        super(R3D18Model, self).__init__()
        self.model = video_models.r3d_18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
