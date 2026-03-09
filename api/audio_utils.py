"""Validation et conversion de fichiers WAV pour Whisper."""

import io
import wave
import numpy as np

SAMPLE_RATE = 16000
MIN_DURATION = 0.3  # secondes


def wav_to_numpy(file_bytes: bytes) -> np.ndarray:
    """Convertit un fichier WAV (bytes) en array float32 16kHz mono.

    Args:
        file_bytes: contenu brut du fichier .wav

    Returns:
        np.ndarray float32, mono, 16kHz

    Raises:
        ValueError: si le fichier est invalide ou le sample rate incorrect.
    """
    try:
        with wave.open(io.BytesIO(file_bytes), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
    except wave.Error as e:
        raise ValueError(f"Fichier WAV invalide : {e}")

    # Resample si nécessaire (ex: 8kHz VoIP -> 16kHz)
    needs_resample = framerate != SAMPLE_RATE

    # Conversion bytes -> numpy float32 selon la largeur d'échantillon
    if sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Largeur d'échantillon non supportée : {sampwidth} octets")

    # Conversion stéréo -> mono
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    # Resample vers 16kHz si nécessaire
    if needs_resample:
        original_len = len(audio)
        target_len = int(original_len * SAMPLE_RATE / framerate)
        indices = np.linspace(0, original_len - 1, target_len)
        audio = np.interp(indices, np.arange(original_len), audio).astype(np.float32)

    # Vérification durée minimale
    duration = len(audio) / SAMPLE_RATE
    if duration < MIN_DURATION:
        raise ValueError(f"Audio trop court ({duration:.2f}s < {MIN_DURATION}s)")

    return audio
