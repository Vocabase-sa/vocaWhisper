"""
Conversion du modèle fine-tuné vers CTranslate2 (faster-whisper).
=================================================================

Ce script convertit le modèle Whisper fine-tuné (format Hugging Face)
en format CTranslate2, compatible avec faster-whisper utilisé par Vocabase.

Prérequis :
    1. Avoir fine-tuné le modèle : python fine_tuning/train.py
    2. Le modèle final se trouve dans fine_tuning/output/final/

Usage :
    python fine_tuning/convert_to_ct2.py
    python fine_tuning/convert_to_ct2.py --quantization float16
    python fine_tuning/convert_to_ct2.py --model fine_tuning/output/final --output models/mon-whisper

Après conversion :
    Le modèle CTranslate2 sera dans fine_tuning/model_ct2/
    Tu peux le sélectionner dans les paramètres de Vocabase comme modèle personnalisé.
"""

import argparse
import os
import shutil
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(BASE_DIR, "output", "final")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "model_ct2")


def check_dependencies():
    """Vérifie que les dépendances sont installées."""
    try:
        import ctranslate2
        print(f"  CTranslate2 version : {ctranslate2.__version__}")
    except ImportError:
        print("[ERREUR] CTranslate2 n'est pas installé.")
        print("  pip install ctranslate2")
        sys.exit(1)

    try:
        import transformers
        print(f"  Transformers version : {transformers.__version__}")
    except ImportError:
        print("[ERREUR] Transformers n'est pas installé.")
        print("  pip install transformers")
        sys.exit(1)


def convert_model(model_dir: str, output_dir: str, quantization: str = "float16", copy_files: bool = True):
    """Convertit le modèle HF en format CTranslate2."""
    import ctranslate2

    print(f"\n  Conversion en cours...")
    print(f"  Source        : {model_dir}")
    print(f"  Destination   : {output_dir}")
    print(f"  Quantization  : {quantization}")

    # Utiliser le convertisseur CTranslate2
    converter = ctranslate2.converters.TransformersConverter(
        model_name_or_path=model_dir,
    )
    converter.convert(
        output_dir=output_dir,
        quantization=quantization,
        force=True,
    )

    print(f"\n  Modèle CTranslate2 créé dans : {output_dir}")

    # Copier les fichiers nécessaires pour faster-whisper
    if copy_files:
        files_to_copy = [
            "tokenizer.json",
            "tokenizer_config.json",
            "preprocessor_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "normalizer.json",
            "vocab.json",
            "merges.txt",
        ]
        copied = 0
        for filename in files_to_copy:
            src = os.path.join(model_dir, filename)
            if os.path.isfile(src):
                dst = os.path.join(output_dir, filename)
                shutil.copy2(src, dst)
                copied += 1

        print(f"  Fichiers copiés : {copied} fichiers tokenizer/config")


def verify_model(output_dir: str):
    """Vérifie que le modèle converti fonctionne avec faster-whisper."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("\n  [INFO] faster-whisper non installé, vérification ignorée.")
        return False

    print(f"\n  Vérification avec faster-whisper...")
    try:
        model = WhisperModel(output_dir, device="cpu", compute_type="int8")
        print("  [OK] Le modèle se charge correctement avec faster-whisper !")
        del model
        return True
    except Exception as e:
        print(f"  [ERREUR] Échec du chargement : {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convertir le modèle fine-tuné en CTranslate2")
    parser.add_argument("--model", type=str, default=None,
                        help=f"Modèle HF à convertir (défaut: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--output", type=str, default=None,
                        help=f"Dossier de sortie CTranslate2 (défaut: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--quantization", type=str, default="float16",
                        choices=["float16", "float32", "int8", "int8_float16", "int8_bfloat16"],
                        help="Type de quantization (défaut: float16)")
    parser.add_argument("--skip_verify", action="store_true",
                        help="Ne pas vérifier le modèle après conversion")
    args = parser.parse_args()

    model_dir = args.model or DEFAULT_MODEL_DIR
    output_dir = args.output or DEFAULT_OUTPUT_DIR

    print("=" * 60)
    print("  Conversion Whisper -> CTranslate2 (faster-whisper)")
    print("=" * 60)

    # Vérifications
    check_dependencies()

    if not os.path.isdir(model_dir):
        print(f"\n[ERREUR] Modèle introuvable : {model_dir}")
        print("Lance d'abord : python fine_tuning/train.py")
        sys.exit(1)

    # Convertir
    convert_model(model_dir, output_dir, quantization=args.quantization)

    # Vérifier
    if not args.skip_verify:
        verify_model(output_dir)

    # Calculer la taille
    total_size = 0
    for dirpath, _, filenames in os.walk(output_dir):
        for f in filenames:
            total_size += os.path.getsize(os.path.join(dirpath, f))

    print("\n" + "=" * 60)
    print("  Conversion terminée !")
    print(f"  Taille du modèle : {total_size / 1e9:.2f} Go")
    print(f"  Chemin : {os.path.abspath(output_dir)}")
    print()
    print("  Pour utiliser ce modèle dans Vocabase :")
    print(f'    1. Ouvre les Paramètres')
    print(f'    2. Dans le champ "Modèle personnalisé", entre :')
    print(f"       {os.path.abspath(output_dir)}")
    print(f'    3. Redémarre Vocabase')
    print("=" * 60)


if __name__ == "__main__":
    main()
