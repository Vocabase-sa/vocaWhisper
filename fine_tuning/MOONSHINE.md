# Moteur Moonshine — intégration et fine-tuning

Branche `moonshine_dev`.

## Pourquoi Moonshine

Moonshine est une architecture ASR conçue pour le temps réel. Sa particularité :
là où Whisper padde tout l'audio à 30 secondes, l'encodeur Moonshine accepte des
fenêtres de longueur variable. Sur un énoncé de 2 s — le format exact du standard
téléphonique (« Je voudrais le docteur X ») — le coût de calcul est proportionnel
à la durée réelle, pas à une fenêtre fixe.

Mesuré sur ce corpus, modèle `tiny` (27M) sur **CPU** : RTF 0,108, soit ~150 ms
par énoncé. Pour la voie RTP, cela ouvre la possibilité de servir des sessions
sans mobiliser de VRAM, là où `api/whisper_pool.py` réserve ~3 Go par instance.

## L'état du français

À connaître avant d'investir : **aucun modèle Moonshine officiel ne couvre le
français**.

| Génération | Date | Langues | Streaming |
|---|---|---|---|
| v1 | 2024 | anglais | non |
| « Flavors of Moonshine » | sept. 2025 | ar, zh, ja, ko, uk, vi (+ es) | non |
| v2 | fév. 2026 | anglais uniquement | oui (50–258 ms) |

Le seul français disponible est un fine-tune communautaire de l'architecture
**v1 tiny** : [`Cornebidouil/moonshine-tiny-fr`](https://huggingface.co/Cornebidouil/moonshine-tiny-fr),
entraîné sur Multilingual LibriSpeech, ~21,8 % de WER.

Conséquence pratique : **le français et le streaming v2 sont disjoints**. On ne
peut pas avoir les deux aujourd'hui. C'est le point à réévaluer si Moonshine AI
publie un modèle français.

## Limite structurelle : pas d'`initial_prompt`

Moonshine n'accepte pas de prompt de conditionnement. Le contenu de
`vocabulaire.txt` ne peut donc **pas** être injecté pour biaiser la
reconnaissance, contrairement au moteur `local` (faster-whisper).

Les corrections restent appliquées en aval — `corrections.txt` puis
`fuzzy_correction.py` — mais le biasing amont est perdu. C'est la contrepartie
directe du gain de latence, et la raison pour laquelle le fine-tuning n'est pas
optionnel ici : ce que le prompt faisait, il faut désormais l'inscrire dans les
poids.

## Utilisation

### Installation

```bash
pip install -r requirements-moonshine.txt
```

### Activer le moteur

Dans l'interface de configuration, « Moteur STT » → `moonshine`. Ou dans
`config.json` :

```json
"stt_engine": "moonshine",
"moonshine_model": "",
"moonshine_backend": "torch",
"moonshine_fallback_local": false
```

- `moonshine_model` vide → `Cornebidouil/moonshine-tiny-fr`. Sinon, chemin du
  modèle fine-tuné.
- `moonshine_backend` : `torch` (défaut, CUDA ou CPU) ou `onnx` (CPU rapide,
  nécessite `optimum`).
- `moonshine_fallback_local` : charge Whisper en secours si Moonshine échoue.

### Fine-tuner sur le corpus

Le dataset préparé pour Whisper est réutilisé **tel quel** — `prepare_dataset.py`
produit des colonnes `audio` + `sentence` brutes, sans featurisation spécifique.

```bash
python fine_tuning/train_moonshine.py --epochs 12 --batch_size 32
```

Sur RTX 4090, 2097 exemples × 12 époques ≈ 6 minutes.

### Évaluer

```bash
python utils/test_moonshine.py --limit 100
python utils/test_moonshine.py --moonshine_model fine_tuning/output_moonshine/final
```

Le banc d'essai rapporte le WER global, la **latence**, et surtout le taux de
reconnaissance des **noms propres** — les tokens en majuscules de la référence.
C'est la métrique qui compte : un standard qui comprend « je voudrais le
docteur » mais rate le nom ne sert à rien.

### Exporter en ONNX (optionnel)

```bash
pip install "optimum[onnxruntime]" onnx
python fine_tuning/export_moonshine_onnx.py --quantize
```

## Résultats mesurés

Corpus : 100 échantillons du split test, 3 minutes d'audio, 104 noms propres.
Moonshine `tiny` fine-tuné 12 époques (LR 3e-5, batch 32) contre le modèle
CTranslate2 fine-tuné du projet. Annuaire fuzzy de 101 noms, seuil 65.

| | Moonshine FT | + fuzzy | Whisper FT | + fuzzy |
|---|---|---|---|---|
| WER | 17,08 % | **8,32 %** | 3,28 % | **0,95 %** |
| Noms propres | 13,5 % | **62,5 %** | 80,8 % | **97,1 %** |
| Latence médiane | 237 ms (GPU) / 212 ms (CPU) | | 372 ms (GPU) | |
| RTF | 0,122 | | 0,209 | |

Le fine-tuning fait passer le WER de 31,4 % (zéro-shot) à 17,1 %, et apprend la
convention de casse du corpus (noms en majuscules). Le fuzzy matching rattrape
ensuite les quasi-correspondances — `CORLAD` → CORLAT, `MARCHETA` → MARCHETTA,
`HORBAN` → ORBAN — pour +49 points sur les noms propres.

### Le learning rate était le facteur limitant

Ce premier run (LR 3e-5) **sous-apprenait** : sur 8 exemples du split *train*,
donc déjà vus à l'entraînement, le modèle n'en restituait que 4 exactement.
Un modèle à sa capacité aurait dû être quasi parfait sur ses propres données.

Second run à **LR 1e-4 sur 30 époques** (`output_moonshine_v2`) :

| | v1 (LR 3e-5, 12 ép.) | v2 (LR 1e-4, 30 ép.) |
|---|---|---|
| eval WER | 19,97 % | **14,13 %** |
| WER + fuzzy (50 éch.) | 8,32 % | **3,2 %** |
| Noms propres + fuzzy | 62,5 % | **84,6 %** |

La conclusion en est renversée : Moonshine correctement entraîné n'est pas
disqualifié. Comparé à Whisper `base` fine-tuné sur le même corpus, en CPU :

| | WER | WER+fuzzy | Noms+fuzzy | RTF |
|---|---|---|---|---|
| whisper base FT int8 (80 Mo) | 11,6 % | 4,0 % | **94,2 %** | 0,271 |
| moonshine v2 FT (107 Mo) | **10,6 %** | **3,2 %** | 84,6 % | **0,111** |

Moonshine gagne sur le WER brut et va 2,4× plus vite ; Whisper `base` gagne les
10 points de noms propres qui décident du routage d'appels. L'arbitrage dépend
donc du facteur limitant : qualité du routage, ou densité de sessions par cœur.

Pistes restant ouvertes :
- modèle `base` (61M) plutôt que `tiny` (27M) — anglais uniquement, à
  re-spécialiser entièrement
- Moonshine en pré-filtre (détection d'activité, barge-in) devant faster-whisper

## Piège d'exécution : conflit cuDNN entre torch et CTranslate2

Moonshine tourne sur **CPU** dans l'application, et ce n'est pas un choix par
défaut paresseux.

`whisper_dictation.py` importe `faster_whisper` au chargement du module, ce qui
charge CTranslate2 et ses bibliothèques cuDNN. Si torch charge ensuite les
siennes dans le même processus, le premier appel CUDA échoue :

```
Could not load symbol cudnnGetLibConfig. Error code 127
```

C'est un crash **natif**, pas une exception Python : aucun `try/except` ne le
rattrape, et sous `pythonw.exe` — le lanceur `run_silent.vbs` — le processus
meurt sans le moindre message. Symptôme observé : l'application ne démarre plus,
sans erreur ni entrée de log.

`MoonshineEngine._resolve_device()` détecte donc CTranslate2 dans `sys.modules`
et retient CPU. Le coût est nul, et même négatif : Moonshine sur CPU (212 ms)
est plus rapide que faster-whisper large-v3 sur RTX 4090 (372 ms), tout en
laissant le GPU libre.

Un device explicite (`"moonshine_device": "cuda"`) reste respecté — à réserver
aux processus où faster-whisper n'est pas chargé, comme `utils/test_moonshine.py`
qui charge torch en premier.

## Piège d'implémentation : le double décalage des labels

À retenir avant de toucher à `train_moonshine.py`.

`MoonshineForConditionalGeneration.forward()` applique `shift_tokens_right()`
quand on lui passe `labels` sans `decoder_input_ids`. Mais sa fonction de perte
est `ForCausalLMLoss`, qui décale **une seconde fois** (`logits[:-1]` contre
`labels[1:]`).

Passer `labels` seul — le schéma standard, celui que suit `train.py` pour
Whisper — produit donc un double décalage. Mesuré sur un exemple que le modèle
de base transcrit pourtant correctement :

| Schéma | Loss |
|---|---|
| `labels` seul (schéma Whisper) | 10,88 |
| `decoder_input_ids` explicite + labels identiques | **3,82** |

Le premier run d'entraînement, fait avec le schéma standard, donnait
`eval_wer: 95,5 %` dès la première époque — pire que le modèle non entraîné.

Le collator fournit donc `decoder_input_ids` explicitement, ce qui court-circuite
le shift du `forward` et ne laisse que celui de la loss.

Second point : **le tokenizer Moonshine ajoute le BOS mais pas l'EOS**. Sans EOS
dans les labels, le modèle n'apprend jamais à s'arrêter. `prepare_dataset_entry()`
l'ajoute explicitement.

## Fichiers

| Fichier | Rôle |
|---|---|
| [moonshine_engine.py](../moonshine_engine.py) | Moteur d'inférence (torch / onnx), découpage des audios longs |
| [fine_tuning/train_moonshine.py](train_moonshine.py) | Fine-tuning sur le corpus |
| [fine_tuning/export_moonshine_onnx.py](export_moonshine_onnx.py) | Export ONNX + quantification INT8 |
| [utils/test_moonshine.py](../utils/test_moonshine.py) | Banc d'essai comparatif |
| [requirements-moonshine.txt](../requirements-moonshine.txt) | Dépendances |

Points d'intégration dans l'existant : `transcribe()` et `load_model()` de
[whisper_dictation.py](../whisper_dictation.py), sélecteur de moteur de
[config_ui.py](../config_ui.py).

La voie RTP (`api/rtp_listener.py`, `api/whisper_pool.py`) n'est **pas** touchée :
elle continue d'utiliser le pool faster-whisper. C'est le prochain chantier si
les mesures sont concluantes.
