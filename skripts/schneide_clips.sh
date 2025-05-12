#!/bin/bash

# Arbeitsverzeichnis: ml_picture_recognition/
# Dieses Skript geht davon aus, dass das Eingabevideo im Ordner videos_roh/ liegt.

# Name des Eingabevideos (ohne Endung)
video_name="neutral"
input_video="videos_roh/${video_name}.mp4"

# Zielverzeichnis für die Clips
output_dir="clips"
mkdir -p "$output_dir"

# Clipdauer in Sekunden
clip_length=12

# Gesamtlänge des Videos in Sekunden ermitteln
duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$input_video")
duration=${duration%.*}  # Nur ganzzahlige Sekunden

# Clips in Schleife schneiden
start=0
clip_index=1

while [ "$start" -lt "$duration" ]; do
    end=$((start + clip_length))
    output_clip="${output_dir}/${video_name}_clip${clip_index}.mp4"

    ffmpeg -loglevel error -y -i "$input_video" -ss "$start" -t "$clip_length" -c copy "$output_clip"

    echo "✅ Erstellt: $output_clip"

    start=$end
    ((clip_index++))
done

echo "🎬 Alle Clips wurden aus $input_video in $output_dir/ gespeichert."
