#!/bin/bash

# Arbeitsverzeichnis: ml_picture_recognition/
# Dieses Skript geht davon aus, dass neutral.mp4 im Ordner videos_roh/ liegt.

# Pfad zum Ursprungs-Video
input_video="videos_roh/neutral.mp4"

# Clips-Ordner anlegen (falls nicht vorhanden)
mkdir -p clips

# Clips schneiden (immer neuer Clipname)
# Start- und Endzeiten der Schnitte sind angegeben

ffmpeg -i "$input_video" -ss 00:00:00 -to 00:00:12 -c copy clips/neutral_clip1.mp4
ffmpeg -i "$input_video" -ss 00:00:12 -to 00:00:25 -c copy clips/neutral_clip2.mp4
ffmpeg -i "$input_video" -ss 00:00:25 -to 00:00:39 -c copy clips/neutral_clip3.mp4
ffmpeg -i "$input_video" -ss 00:00:39 -to 00:00:50 -c copy clips/neutral_clip4.mp4
ffmpeg -i "$input_video" -ss 00:00:50 -to 00:01:01 -c copy clips/neutral_clip5.mp4
ffmpeg -i "$input_video" -ss 00:01:01 -to 00:01:12 -c copy clips/neutral_clip6.mp4
ffmpeg -i "$input_video" -ss 00:01:12 -to 00:01:27 -c copy clips/neutral_clip7.mp4
ffmpeg -i "$input_video" -ss 00:01:27 -to 00:01:39 -c copy clips/neutral_clip8.mp4
ffmpeg -i "$input_video" -ss 00:01:39 -to 00:01:53 -c copy clips/neutral_clip9.mp4
ffmpeg -i "$input_video" -ss 00:01:53 -to 00:02:04 -c copy clips/neutral_clip10.mp4
ffmpeg -i "$input_video" -ss 00:02:04 -to 00:02:18 -c copy clips/neutral_clip11.mp4
ffmpeg -i "$input_video" -ss 00:02:18 -to 00:02:31 -c copy clips/neutral_clip12.mp4
ffmpeg -i "$input_video" -ss 00:02:31 -to 00:02:43 -c copy clips/neutral_clip13.mp4
ffmpeg -i "$input_video" -ss 00:02:43 -to 00:02:53 -c copy clips/neutral_clip14.mp4
ffmpeg -i "$input_video" -ss 00:02:53 -to 00:03:06 -c copy clips/neutral_clip15.mp4
ffmpeg -i "$input_video" -ss 00:03:06 -to 00:03:22 -c copy clips/neutral_clip16.mp4
ffmpeg -i "$input_video" -ss 00:03:22 -to 00:03:38 -c copy clips/neutral_clip17.mp4
ffmpeg -i "$input_video" -ss 00:03:38 -to 00:03:50 -c copy clips/neutral_clip18.mp4
ffmpeg -i "$input_video" -ss 00:03:50 -to 00:04:00 -c copy clips/neutral_clip19.mp4
ffmpeg -i "$input_video" -ss 00:04:00 -to 00:04:14 -c copy clips/neutral_clip20.mp4
ffmpeg -i "$input_video" -ss 00:04:14 -to 00:04:27 -c copy clips/neutral_clip21.mp4
ffmpeg -i "$input_video" -ss 00:04:27 -to 00:04:38 -c copy clips/neutral_clip22.mp4
ffmpeg -i "$input_video" -ss 00:04:38 -to 00:04:48 -c copy clips/neutral_clip23.mp4
ffmpeg -i "$input_video" -ss 00:04:48 -to 00:05:01 -c copy clips/neutral_clip24.mp4
ffmpeg -i "$input_video" -ss 00:05:01 -to 00:05:12 -c copy clips/neutral_clip25.mp4

echo "✅ Alle Clips wurden erfolgreich erstellt und gespeichert unter ./clips/"
