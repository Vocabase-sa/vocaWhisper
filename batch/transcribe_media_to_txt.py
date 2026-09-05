#!/usr/bin/env python3
"""
Transcrit un fichier audio/video en texte avec faster-whisper.

Exemples:
    python batch/transcribe_media_to_txt.py "C:\video.mp4"
    python batch/transcribe_media_to_txt.py "C:\video.mp4" --output "C:\video.txt"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "config.json"
VOCAB_PATH = BASE_DIR / "vocabulaire.txt"
CORRECTIONS_PATH = BASE_DIR / "corrections.txt"

VIDEO_EXTENSIONS = {
    ".mp4",
    ".m4v",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".wmv",
    ".flv",
    ".mpeg",
    ".mpg",
}


def emit(message: str, log_handle, verbose: bool = False, error: bool = False) -> None:
    if log_handle is not None:
        log_handle.write(message + "\n")
        log_handle.flush()
    if verbose:
        stream = sys.stderr if error else sys.stdout
        print(message, file=stream)


def load_config() -> dict:
    defaults = {
        "model_size": "large-v3",
        "custom_model_path": "",
        "device": "cuda",
        "compute_type": "float16",
        "language": "fr",
    }
    if not CONFIG_PATH.exists():
        return defaults

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    for key, value in defaults.items():
        config.setdefault(key, value)
    return config


def load_vocabulary() -> str:
    if not VOCAB_PATH.exists():
        return ""

    words: list[str] = []
    with VOCAB_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                words.append(line)
    return ", ".join(words)


def load_corrections() -> list[tuple[re.Pattern[str], str]]:
    if not CORRECTIONS_PATH.exists():
        return []

    corrections: list[tuple[re.Pattern[str], str]] = []
    with CORRECTIONS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or " -> " not in line:
                continue
            source, target = line.split(" -> ", 1)
            source = source.strip()
            target = target.strip()
            if source and target:
                corrections.append((re.compile(re.escape(source), re.IGNORECASE), target))
    return corrections


def apply_corrections(text: str, corrections: list[tuple[re.Pattern[str], str]]) -> str:
    for pattern, replacement in corrections:
        text = pattern.sub(replacement, text)
    return text


def build_output_path(input_path: Path, output_arg: str | None) -> Path:
    if output_arg:
        return Path(output_arg).expanduser().resolve()
    return input_path.with_suffix(".txt")


def extract_audio_with_ffmpeg(input_path: Path) -> Path:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg est introuvable dans le PATH.")

    temp_dir = Path(tempfile.mkdtemp(prefix="vocawhisper_transcribe_"))
    audio_path = temp_dir / f"{input_path.stem}.wav"

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(audio_path),
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not audio_path.exists():
        raise RuntimeError(f"ffmpeg a echoue:\n{result.stderr.strip()}")
    return audio_path


def transcribe_file(
    media_path: Path,
    output_path: Path,
    language: str | None,
    model_name: str,
    device: str,
    compute_type: str,
    use_vad: bool,
    verbose: bool,
) -> None:
    from faster_whisper import WhisperModel

    vocabulary = load_vocabulary()
    corrections = load_corrections()
    log_path = output_path.with_suffix(".log")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = output_path.with_suffix(output_path.suffix + ".partial")
    temp_log_path = log_path.with_suffix(log_path.suffix + ".partial")

    media_for_whisper = media_path
    temp_audio_path: Path | None = None
    with temp_log_path.open("w", encoding="utf-8") as log_handle:
        emit(f"Source: {media_path}", log_handle, verbose)
        emit(f"Texte: {output_path}", log_handle, verbose)
        emit(f"Modele: {model_name}", log_handle, verbose)
        emit(f"Device demande: {device} ({compute_type})", log_handle, verbose)

        try:
            emit("Chargement du modele...", log_handle, verbose)
            try:
                model = WhisperModel(model_name, device=device, compute_type=compute_type)
            except Exception as exc:
                if device == "cpu":
                    raise
                emit(f"Echec sur {device}: {exc}", log_handle, verbose)
                emit("Fallback sur CPU (int8)...", log_handle, verbose)
                model = WhisperModel(model_name, device="cpu", compute_type="int8")

            if media_path.suffix.lower() in VIDEO_EXTENSIONS:
                emit("Extraction audio via ffmpeg...", log_handle, verbose)
                temp_audio_path = extract_audio_with_ffmpeg(media_path)
                media_for_whisper = temp_audio_path

            start_time = time.time()
            kwargs = {
                "beam_size": 5,
                "vad_filter": use_vad,
                "initial_prompt": vocabulary or None,
            }
            if language and language.lower() != "auto":
                kwargs["language"] = language

            emit("Transcription en cours...", log_handle, verbose)
            segments, info = model.transcribe(str(media_for_whisper), **kwargs)
            lines_written = 0

            with temp_output_path.open("w", encoding="utf-8") as handle:
                for segment in segments:
                    chunk = apply_corrections(segment.text.strip(), corrections).strip()
                    if not chunk:
                        continue
                    if lines_written:
                        handle.write("\n")
                    handle.write(chunk)
                    handle.flush()
                    lines_written += 1

            elapsed = time.time() - start_time
            emit(f"Langue detectee: {info.language} ({info.language_probability:.0%})", log_handle, verbose)
            emit(f"Lignes ecrites: {lines_written}", log_handle, verbose)
            emit(f"Transcription terminee en {elapsed / 60:.1f} min", log_handle, verbose)

            temp_output_path.replace(output_path)
            temp_log_path.replace(log_path)
        finally:
            if temp_audio_path is not None:
                shutil.rmtree(temp_audio_path.parent, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcrit un fichier audio/video en texte avec faster-whisper."
    )
    parser.add_argument("input", help="Chemin du fichier media (mp3, wav, mp4, mov, mkv, etc.)")
    parser.add_argument(
        "-o",
        "--output",
        help="Chemin du fichier texte de sortie. Par defaut: meme nom que l'entree avec .txt",
    )
    parser.add_argument(
        "--language",
        help="Langue forcee (ex: fr, en). Par defaut: valeur de config.json",
    )
    parser.add_argument(
        "--model",
        help="Modele Whisper a utiliser. Par defaut: valeur de config.json",
    )
    parser.add_argument(
        "--device",
        help="Peripherique (cuda, cpu). Par defaut: valeur de config.json",
    )
    parser.add_argument(
        "--compute-type",
        help="Precision (float16, float32, int8). Par defaut: valeur de config.json",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Desactive le VAD filter.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Affiche aussi les details dans la console.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Fichier introuvable: {input_path}", file=sys.stderr)
        return 1

    output_path = build_output_path(input_path, args.output)

    language = args.language if args.language is not None else config.get("language", "fr")
    model_name = args.model or config.get("custom_model_path") or config.get("model_size", "large-v3")
    device = args.device or config.get("device", "cuda")
    compute_type = args.compute_type or config.get("compute_type", "float16")

    try:
        transcribe_file(
            media_path=input_path,
            output_path=output_path,
            language=language,
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            use_vad=not args.no_vad,
            verbose=args.verbose,
        )
    except KeyboardInterrupt:
        print("\nTranscription interrompue.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Erreur: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
