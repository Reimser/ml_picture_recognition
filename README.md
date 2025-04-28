# Eishockey-Action-Recognition auf eigenen Spieldaten

Dieses Projekt untersucht die automatische Erkennung von spielrelevanten Aktionen in Eishockeyspielen auf Basis eigens geschnittener Videoclips.

## Projektziel

- Verarbeitung von eigenen Spielvideos (ganze Spiele)
- Erstellung kurzer Clips (5–10 Sekunden) mit spezifischen Aktionen
- Erstellung eines sauberen Datensatzes (`labels.csv`)
- Extraktion von Frames zur Vorbereitung auf Modelltraining
- Aufbau eines Machine-Learning-Modells zur Klassifikation von Aktionen

## Klassifizierte Aktionen

Nur folgende vier Aktionen werden betrachtet:

- **Tor** (z. B. Schuss ins Tor)
- **Check** (Körperchecks)
- **Schuss** (Torschüsse ohne Torerfolg)
- **Neutral** (normale Spielszenen ohne spezielle Aktion)

## Verzeichnisstruktur

```bash
eishockey_projekt/
├── videos_roh/     # Originale vollständige Spielvideos (nicht in GitHub)
├── clips/          # Kurz geschnittene Clips einzelner Aktionen (nicht in GitHub)
├── frames/         # Extrahierte Einzelbilder aus Clips (nicht in GitHub)
├── poses/          # Keypoints nach Pose Estimation (optional, nicht in GitHub)
├── labels.csv      # Tabelle mit Clipnamen und Labels
├── extract_frames.py  # Script zur Extraktion von Frames
├── train_model.py     # Script für Modelltraining (folgt)
├── README.md       # Projektbeschreibung
├── .gitignore      # Ausschlussregeln für Git
