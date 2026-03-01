"""Test de transcription sur le fichier test_audio.wav"""
from faster_whisper import WhisperModel
import numpy as np
import time

print("Chargement du modèle large-v3 sur CUDA...")
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
print("Modèle chargé.")

print()
print("=== Test 1 : AVEC vad_filter (comme le programme principal) ===")
t0 = time.perf_counter()
segments, info = model.transcribe(
    "test_audio.wav",
    language="fr",
    beam_size=5,
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500),
)
text = " ".join(s.text.strip() for s in segments).strip()
print(f"  Temps : {time.perf_counter() - t0:.1f}s")
print(f"  Langue : {info.language} ({info.language_probability:.0%})")
print(f"  Résultat : '{text}'")

print()
print("=== Test 2 : SANS vad_filter ===")
t0 = time.perf_counter()
segments, info = model.transcribe(
    "test_audio.wav",
    language="fr",
    beam_size=5,
    vad_filter=False,
)
text = " ".join(s.text.strip() for s in segments).strip()
print(f"  Temps : {time.perf_counter() - t0:.1f}s")
print(f"  Langue : {info.language} ({info.language_probability:.0%})")
print(f"  Résultat : '{text}'")

print()
print("=== Test 3 : SANS vad_filter + audio amplifié x10 ===")
import wave
with wave.open("test_audio.wav", 'r') as wf:
    raw = wf.readframes(wf.getnframes())
    audio_int16 = np.frombuffer(raw, dtype=np.int16)
    audio_float = audio_int16.astype(np.float32) / 32768.0

audio_amplified = np.clip(audio_float * 10.0, -1.0, 1.0)
print(f"  Niveau original max : {np.abs(audio_float).max():.4f}")
print(f"  Niveau amplifié max : {np.abs(audio_amplified).max():.4f}")

t0 = time.perf_counter()
segments, info = model.transcribe(
    audio_amplified,
    language="fr",
    beam_size=5,
    vad_filter=False,
)
text = " ".join(s.text.strip() for s in segments).strip()
print(f"  Temps : {time.perf_counter() - t0:.1f}s")
print(f"  Langue : {info.language} ({info.language_probability:.0%})")
print(f"  Résultat : '{text}'")
