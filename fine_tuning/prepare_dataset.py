"""
Préparation du dataset pour le fine-tuning de Whisper.
=====================================================

Ce script prend tes paires audio/transcription et les transforme
en un dataset Hugging Face prêt pour l'entraînement.

Structure attendue dans fine_tuning/data/ :
-------------------------------------------
Option 1 : CSV + fichiers audio
    data/
        audio/
            enregistrement_001.wav
            enregistrement_002.wav
            ...
        transcriptions.csv     (colonnes: audio_file, transcription)

Option 2 : Dossiers split (si tu veux séparer train/test toi-même)
    data/
        train/
            audio/
                001.wav
            transcriptions.csv
        test/
            audio/
                001.wav
            transcriptions.csv

Le CSV doit avoir ce format :
    audio_file,transcription
    enregistrement_001.wav,"Bonjour, je teste le fine-tuning de Whisper."
    enregistrement_002.wav,"OpenSIPs est un serveur SIP open source."

Usage :
    python fine_tuning/prepare_dataset.py
    python fine_tuning/prepare_dataset.py --test_size 0.1
    python fine_tuning/prepare_dataset.py --csv data/mes_transcriptions.csv --audio_dir data/mes_audios
"""

import argparse
import csv
import os
import sys

from datasets import Audio, Dataset, DatasetDict


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset_prepared")


def load_from_csv(csv_path: str, audio_dir: str) -> list[dict]:
    """Charge les paires audio/transcription depuis un fichier CSV."""
    entries = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Vérifier les colonnes
        if "audio_file" not in reader.fieldnames or "transcription" not in reader.fieldnames:
            print("[ERREUR] Le CSV doit contenir les colonnes 'audio_file' et 'transcription'.")
            print(f"  Colonnes trouvées : {reader.fieldnames}")
            sys.exit(1)

        for row in reader:
            audio_file = row["audio_file"].strip()
            transcription = row["transcription"].strip()

            # Chercher le fichier audio
            audio_path = os.path.join(audio_dir, audio_file)
            if not os.path.isfile(audio_path):
                print(f"  [SKIP] Audio introuvable : {audio_path}")
                continue

            if not transcription:
                print(f"  [SKIP] Transcription vide pour : {audio_file}")
                continue

            entries.append({
                "audio": audio_path,
                "sentence": transcription,
            })

    return entries


def create_dataset(entries: list[dict], test_size: float = 0.1) -> DatasetDict:
    """Crée un DatasetDict (train/test) à partir des entrées."""
    if not entries:
        print("[ERREUR] Aucune entrée valide trouvée !")
        sys.exit(1)

    print(f"\n  Total : {len(entries)} paires audio/transcription")

    # Créer le dataset
    dataset = Dataset.from_dict({
        "audio": [e["audio"] for e in entries],
        "sentence": [e["sentence"] for e in entries],
    })

    # Configurer la colonne audio pour le chargement automatique
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Séparer train/test
    if test_size > 0 and len(entries) > 5:
        split = dataset.train_test_split(test_size=test_size, seed=42)
        print(f"  Train : {len(split['train'])} exemples")
        print(f"  Test  : {len(split['test'])} exemples")
        return split
    else:
        print(f"  (Pas de split test — trop peu de données ou test_size=0)")
        return DatasetDict({"train": dataset})


def main():
    parser = argparse.ArgumentParser(description="Prépare le dataset pour le fine-tuning Whisper")
    parser.add_argument("--csv", type=str, default=None,
                        help="Chemin vers le CSV (défaut: data/transcriptions.csv)")
    parser.add_argument("--audio_dir", type=str, default=None,
                        help="Dossier contenant les fichiers audio (défaut: data/audio/)")
    parser.add_argument("--test_size", type=float, default=0.1,
                        help="Proportion du dataset pour le test (défaut: 0.1)")
    parser.add_argument("--output", type=str, default=None,
                        help="Dossier de sortie (défaut: fine_tuning/dataset_prepared/)")
    args = parser.parse_args()

    # Chemins par défaut
    csv_path = args.csv or os.path.join(DATA_DIR, "transcriptions.csv")
    audio_dir = args.audio_dir or os.path.join(DATA_DIR, "audio")
    output_dir = args.output or OUTPUT_DIR

    print("=" * 60)
    print("  Préparation du dataset Whisper Fine-Tuning")
    print("=" * 60)
    print(f"\n  CSV          : {csv_path}")
    print(f"  Audio dir    : {audio_dir}")
    print(f"  Test size    : {args.test_size}")
    print(f"  Output       : {output_dir}")

    # Vérifications
    if not os.path.isfile(csv_path):
        print(f"\n[ERREUR] CSV introuvable : {csv_path}")
        print("\nCrée un fichier transcriptions.csv dans fine_tuning/data/ avec le format :")
        print("  audio_file,transcription")
        print('  001.wav,"Ma transcription ici."')
        sys.exit(1)

    if not os.path.isdir(audio_dir):
        print(f"\n[ERREUR] Dossier audio introuvable : {audio_dir}")
        print("\nCrée le dossier fine_tuning/data/audio/ et place-y tes fichiers audio.")
        sys.exit(1)

    # Charger les données
    print("\n  Chargement des données...")
    entries = load_from_csv(csv_path, audio_dir)

    # Créer le dataset
    print("\n  Création du dataset Hugging Face...")
    dataset = create_dataset(entries, test_size=args.test_size)

    # Sauvegarder
    print(f"\n  Sauvegarde dans : {output_dir}")
    dataset.save_to_disk(output_dir)

    print("\n" + "=" * 60)
    print("  Dataset prêt !")
    print(f"  Prochaine étape : python fine_tuning/train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
