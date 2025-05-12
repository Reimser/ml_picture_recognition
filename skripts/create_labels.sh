#!/bin/bash

# Pfad zu deinen Clips
clips_dir="clips"

# Ziel-Labels-Datei
labels_file="labels.csv"

# Labels-Datei neu anlegen (und Kopfzeile schreiben)
echo "clip_name,label" > "$labels_file"

# Alle Clips auflisten und als "Neutral" labeln
for clip in "$clips_dir"/*.mp4; do
    clipname=$(basename "$clip")
    echo "$clipname,Neutral" >> "$labels_file"
done

echo "✅ labels.csv wurde erfolgreich erstellt!"
