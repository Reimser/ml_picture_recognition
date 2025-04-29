#!/bin/bash

# Ordner, in dem deine Clips liegen
clips_dir="clips"

# Zielordner für die Frames
frames_dir="frames"

# Zielordner erstellen, falls er noch nicht existiert
mkdir -p "$frames_dir"

# Schleife über alle Clips
for clip in "$clips_dir"/*.mp4; do
    # Clipname ohne Verzeichnis und ohne Dateiendung
    filename=$(basename "$clip" .mp4)

    # Eigenen Ordner für Frames dieses Clips erstellen
    mkdir -p "$frames_dir/$filename"

    # Frames extrahieren (z.B. 10 Frame pro Sekunde)
    ffmpeg -i "$clip" -vf fps=10 "$frames_dir/$filename/frame_%04d.jpg"
done

echo "✅ Alle Frames wurden erfolgreich extrahiert und gespeichert unter ./frames/"
