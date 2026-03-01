"""Test rapide du micro - enregistre 5 secondes et affiche le niveau sonore."""
import sounddevice as sd
import numpy as np
import time

SAMPLE_RATE = 16000
DURATION = 5

print("=== Test du micro ===")
print()

# Afficher tous les périphériques
print("Périphériques audio disponibles :")
print(sd.query_devices())
print()

default = sd.query_devices(kind='input')
print(f"Micro par défaut : {default['name']}")
print(f"  Canaux max : {default['max_input_channels']}")
print(f"  Sample rate par défaut : {default['default_samplerate']}")
print()

print(f"Enregistrement de {DURATION}s... PARLE MAINTENANT !")
print()

audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
for i in range(DURATION):
    time.sleep(1)
    # Afficher le niveau en temps réel
    recorded_so_far = audio[:int((i + 1) * SAMPLE_RATE)]
    level = np.abs(recorded_so_far).max()
    bar_len = int(level * 200)
    bar = "#" * min(bar_len, 50)
    print(f"  Seconde {i+1}/{DURATION} : niveau max = {level:.4f} |{bar}|")

sd.wait()

print()
print(f"Enregistrement terminé.")

level_max = np.abs(audio).max()
level_mean = np.abs(audio).mean()

print(f"  Niveau max  : {level_max:.6f}")
print(f"  Niveau moyen: {level_mean:.6f}")
print()

if level_max < 0.001:
    print(">>> PROBLÈME : Le micro ne capte quasiment rien !")
    print("    - Vérifie que le bon micro est sélectionné dans Windows")
    print("    - Vérifie que le volume du micro n'est pas à zéro")
    print("    - Essaie un autre périphérique (voir liste ci-dessus)")
elif level_max < 0.01:
    print(">>> ATTENTION : Niveau très faible, le micro capte peu.")
    print("    Monte le volume du micro dans les paramètres Windows.")
else:
    print(">>> OK : Le micro capte du son correctement !")

# Sauvegarder pour tester avec whisper
import wave
import struct

filename = "test_audio.wav"
with wave.open(filename, 'w') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)
    audio_int16 = (audio.flatten() * 32767).astype(np.int16)
    wf.writeframes(audio_int16.tobytes())

print(f"\nAudio sauvegardé dans '{filename}' pour vérification.")
