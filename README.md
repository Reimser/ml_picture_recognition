# Eishockey-Video-Analyse mit Machine Learning

Dieses Projekt untersucht die automatische Erkennung von spielrelevanten Aktionen in Eishockeyspielen auf Basis eigener Videodaten.

## Projektziel

- Verarbeitung von eigenen Spielvideos
- Extraktion von Spielszenen (Clips)
- Extraktion von Einzelbildern (Frames)
- Optional: Anwendung von Pose Estimation (MediaPipe)
- Training eines Machine-Learning-Modells zur Erkennung von Aktionen (z. B. Tor, Check, Pass)

## Verzeichnisstruktur

```bash
eishockey_projekt/
├── videos_roh/     # Originale Spielvideos (nicht in GitHub)
├── clips/          # Ausgeschnittene Spielszenen (nicht in GitHub)
├── frames/         # Einzelbilder aus Clips (nicht in GitHub)
├── poses/          # Pose-Daten (nicht in GitHub)
├── models/         # Modell-Dateien (nicht in GitHub)
├── extract_frames.py  # Script zur Frame-Extraktion
├── train_model.py     # Modell-Training (noch aufzubauen)
├── labels.csv         # CSV-Datei mit Clip-Labels
├── README.md          # Projektbeschreibung
└── .gitignore         # Ausgeschlossene Ordner/Dateien

