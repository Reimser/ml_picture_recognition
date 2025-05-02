# Action Recognition im Eishockey mit CNN + LSTM
Dieses Projekt beschäftigt sich mit der Klassifikation von Eishockey-Aktionen wie Tor, Schuss, Check und Neutral auf Basis von Videoaufnahmen.
Dazu werden Videoclips in einzelne Frames aufgeteilt, über ein CNN encodiert und anschließend sequenziell mit einem LSTM ausgewertet.

## Projektstruktur

```text
ml_picture_recognition/
├── clips/                # Geschnittene Videoclips (.mp4)
├── frames/               # Extrahierte Einzelbilder aus Clips
├── models/               # Gespeicherte Modelle (.pth)
├── videos_roh/           # Rohmaterial (komplette Videos)
├── model.py              # CNN + LSTM Modellarchitektur
├── dataloader.py         # Dataset-Klasse für Frame-Input
├── train.ipynb           # Training des Modells
├── eval.ipynb            # Vorhersagen auf Testdaten
├── visualize.ipynb       # Anzeige der Frames + Vorhersage
├── schneide_clips.sh     # Automatisches Clip-Schneiden
├── extract_frames.sh     # Extrahieren von Frames (mit ffmpeg)
├── create_labels.sh      # Generierung der labels.csv
├── labels.csv            # Zuordnung von Clipnamen zu Labels
├── README.md
├── .gitignore
```
Voraussetzungen
Python 3.8+

PyTorch

torchvision

Pillow

ffmpeg (für Frame-Extraktion)

matplotlib (für Visualisierung)

## Installation
bash
Kopieren
Bearbeiten
pip install torch torchvision pillow matplotlib
Ablauf
Clips erstellen:
Videos in kurze Clips (ca. 10–15 Sek.) schneiden mit schneide_clips.sh.

Frames extrahieren:
Aus jedem Clip 10 Bilder extrahieren (z. B. mit 5–10 FPS) via extract_frames.sh.

Labels generieren:
Die Datei labels.csv mit Clipnamen und Aktionen vorbereiten (z. B. mit create_labels.sh).

Training:
Modelltraining über train.ipynb.

Evaluation:
Modellbewertung auf Testdaten mit eval.ipynb.

Visualisierung:
Frames und Vorhersage anzeigen mit visualize.ipynb.

## Modellarchitektur
CNN (ResNet-18, vortrainiert) extrahiert pro Frame ein Feature.

LSTM analysiert die zeitliche Abfolge der Features.

Klassifikationskopf mit 4 Ausgängen: Tor, Schuss, Check, Neutral.

## Ziel
Die Aktion eines Spielers in einem kurzen Clip automatisch erkennen – insbesondere auf realem Amateur-Videomaterial, bei dem mehrere Spieler gleichzeitig sichtbar sind.

## Lizenz
Dieses Projekt steht unter einer freien Forschungs- und Lernlizenz. Bei Verwendung des Codes oder der Datenstruktur bitte Quelle angeben.
