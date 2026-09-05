"""
Fine-tuning de Moonshine (ASR edge, 27M params) sur ton corpus.
================================================================

Moonshine est une architecture ASR conçue pour le temps réel / faible latence.
Contrairement à Whisper qui padde tout l'audio à 30 secondes, Moonshine accepte
des fenêtres de longueur variable : sur un segment de 2 s, tu paies 2 s de calcul.
C'est exactement le profil des énoncés du standard téléphonique
("Je voudrais le docteur X").

Différences avec train.py (Whisper) :
    - entrée = audio brut 16 kHz (input_values), PAS de log-Mel spectrogram
    - modèle MONOLINGUE : pas de token <|fr|> ni de tâche "transcribe"
    - le padding se fait sur les échantillons audio, pas sur des frames fixes
    - modèle 50x plus petit -> batch bien plus gros et learning rate plus élevé

Le dataset préparé par prepare_dataset.py est réutilisé TEL QUEL
(colonnes : audio 16 kHz + sentence). Aucune re-préparation nécessaire.

Prérequis :
    1. python fine_tuning/prepare_dataset.py
    2. pip install -r requirements-moonshine.txt

Usage :
    python fine_tuning/train_moonshine.py
    python fine_tuning/train_moonshine.py --epochs 15 --batch_size 64
    python fine_tuning/train_moonshine.py --base_model UsefulSensors/moonshine-base

Conseil RTX 4090 (24 Go VRAM) :
    - tiny (27M) : batch_size=64 confortable
    - base (61M) : batch_size=32
    - Sur un corpus fermé (formule de phrase quasi fixe), 10-20 époques valent
      mieux que 3 : le modèle doit surtout mémoriser le lexique des noms propres.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from datasets import DatasetDict, load_from_disk
from transformers import (
    AutoProcessor,
    MoonshineForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_prepared")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_moonshine")

# Modèle de départ : le seul checkpoint français publié à ce jour.
# Les modèles officiels Moonshine (v1 "Flavors", v2 streaming) ne couvrent PAS
# le français -> on part du fine-tune communautaire MLS-French.
DEFAULT_BASE_MODEL = "Cornebidouil/moonshine-tiny-fr"

SAMPLE_RATE = 16000


# =============================================================================
# Data Collator pour Moonshine
# =============================================================================
@dataclass
class DataCollatorMoonshine:
    """Collator qui pad l'audio brut et les labels texte.

    Whisper pad tout à 3000 frames (30 s). Ici on pad seulement à la longueur
    du plus long échantillon du batch : c'est la principale source du gain de
    vitesse de Moonshine sur des énoncés courts.

    ATTENTION — décalage des labels :
        MoonshineForConditionalGeneration.forward() applique shift_tokens_right()
        quand on lui passe `labels` sans `decoder_input_ids`. Mais sa loss est
        ForCausalLMLoss, qui décale une SECONDE fois (logits[:-1] vs labels[1:]).
        Passer `labels` seul produit donc un double décalage et un entraînement
        dégénéré (loss ~10.5 au lieu de ~3.6 sur un exemple pourtant bien
        transcrit par le modèle de base).

        On fournit donc `decoder_input_ids` explicitement pour court-circuiter
        le shift du forward, avec des labels identiques (BOS compris) : le seul
        décalage restant est celui de ForCausalLMLoss, qui est le bon.
    """
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        # Audio brut -> padding à la longueur max du batch
        input_features = [{"input_values": f["input_values"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Texte -> padding
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        input_ids = labels_batch["input_ids"]

        # decoder_input_ids : padding avec pad_token_id (indices d'embedding valides)
        batch["decoder_input_ids"] = input_ids

        # labels : mêmes tokens, mais padding masqué à -100 (ignoré par la loss)
        batch["labels"] = input_ids.masked_fill(labels_batch.attention_mask.ne(1), -100)

        return batch


# =============================================================================
# Préparation des données
# =============================================================================
def prepare_dataset_entry(batch, processor, eos_token_id):
    """Prépare une entrée : audio brut -> input_values, texte -> tokens."""
    audio = batch["audio"]

    # Moonshine consomme la forme d'onde directement (pas de spectrogramme)
    features = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
    )
    batch["input_values"] = features.input_values[0]
    batch["n_samples"] = len(batch["input_values"])

    # Le tokenizer Moonshine ajoute le BOS mais PAS l'EOS. Sans EOS dans les
    # labels, le modèle n'apprend jamais à s'arrêter et la génération part en
    # boucle : on l'ajoute explicitement.
    ids = processor.tokenizer(batch["sentence"]).input_ids
    if not ids or ids[-1] != eos_token_id:
        ids = ids + [eos_token_id]
    batch["labels"] = ids
    return batch


# =============================================================================
# Métriques
# =============================================================================
def compute_metrics(pred, tokenizer, metric):
    """Calcule le WER (Word Error Rate) sur les prédictions."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Remplacer -100 par le token de padding avant décodage
    label_ids = np.where(label_ids == -100, tokenizer.pad_token_id, label_ids)

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# =============================================================================
# Entraînement
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Fine-tuning Moonshine")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL,
                        help=f"Modèle de base (défaut: {DEFAULT_BASE_MODEL})")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Chemin du dataset préparé (défaut: fine_tuning/dataset_prepared/)")
    parser.add_argument("--output", type=str, default=None,
                        help="Dossier de sortie (défaut: fine_tuning/output_moonshine/)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Nombre d'époques (défaut: 10)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Taille de batch par GPU (défaut: 32)")
    parser.add_argument("--gradient_accumulation", type=int, default=1,
                        help="Étapes d'accumulation de gradient (défaut: 1)")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                        help="Learning rate (défaut: 3e-5)")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Étapes de warmup (défaut: 100)")
    parser.add_argument("--max_duration", type=float, default=30.0,
                        help="Durée audio max en secondes (défaut: 30)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Reprendre depuis un checkpoint")
    parser.add_argument("--no_fp16", action="store_true",
                        help="Désactiver la précision mixte")
    args = parser.parse_args()

    dataset_dir = args.dataset or DATASET_DIR
    output_dir = args.output or OUTPUT_DIR

    print("=" * 60)
    print("  Fine-tuning Moonshine")
    print("=" * 60)
    print(f"\n  Modèle de base : {args.base_model}")
    print(f"  Dataset        : {dataset_dir}")
    print(f"  Output         : {output_dir}")
    print(f"  Époques        : {args.epochs}")
    print(f"  Batch size     : {args.batch_size}")
    print(f"  Grad accum     : {args.gradient_accumulation}")
    print(f"  Learning rate  : {args.learning_rate}")

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print(f"  Device         : {torch.cuda.get_device_name(0)} (CUDA)")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM           : {vram:.1f} Go")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("  Device         : Apple Silicon (MPS)")
    else:
        print("  Device         : CPU (l'entraînement sera lent)")
    print()

    # --- Charger le dataset ---
    print("  Chargement du dataset...")
    if not os.path.isdir(dataset_dir):
        print(f"\n[ERREUR] Dataset introuvable : {dataset_dir}")
        print("Lance d'abord : python fine_tuning/prepare_dataset.py")
        sys.exit(1)

    dataset = load_from_disk(dataset_dir)
    if not isinstance(dataset, DatasetDict):
        dataset = DatasetDict({"train": dataset})
    print(f"  Train : {len(dataset['train'])} exemples")
    if "test" in dataset:
        print(f"  Test  : {len(dataset['test'])} exemples")

    # --- Charger le processeur et le modèle ---
    print(f"\n  Chargement du processeur depuis {args.base_model}...")
    try:
        processor = AutoProcessor.from_pretrained(args.base_model)
    except Exception as e:
        print(f"\n[ERREUR] Impossible de charger le processeur : {e}")
        sys.exit(1)

    print(f"  Chargement du modèle {args.base_model}...")
    model = MoonshineForConditionalGeneration.from_pretrained(args.base_model)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Paramètres : {n_params:.1f}M")

    # Moonshine est monolingue : contrairement à Whisper, il n'y a pas de
    # forced_decoder_ids (langue + tâche) à neutraliser avant l'entraînement.
    model.generation_config.max_new_tokens = 128

    # --- Préparer le dataset ---
    print("\n  Préparation des features audio (forme d'onde brute)...")
    dataset = dataset.map(
        lambda b: prepare_dataset_entry(b, processor, model.config.eos_token_id),
        remove_columns=dataset["train"].column_names,
        num_proc=1,
        desc="Extraction audio",
    )

    # --- Filtrer les audios trop longs ---
    max_samples = int(args.max_duration * SAMPLE_RATE)
    before = len(dataset["train"])
    dataset = dataset.filter(lambda n: n <= max_samples, input_columns=["n_samples"])
    dropped = before - len(dataset["train"])
    if dropped:
        print(f"  {dropped} exemple(s) écarté(s) : durée > {args.max_duration}s")

    durations = np.array(dataset["train"]["n_samples"]) / SAMPLE_RATE
    print(f"  Durée audio : moyenne {durations.mean():.1f}s, "
          f"max {durations.max():.1f}s, total {durations.sum() / 3600:.2f}h")

    dataset = dataset.remove_columns(["n_samples"])

    # --- Collator ---
    data_collator = DataCollatorMoonshine(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # --- Métrique WER ---
    import evaluate
    metric = evaluate.load("wer")

    has_eval = "test" in dataset and len(dataset["test"]) > 0

    # --- Arguments d'entraînement ---
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        lr_scheduler_type="linear",
        fp16=use_cuda and not args.no_fp16,
        eval_strategy="epoch" if has_eval else "no",
        save_strategy="epoch",
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=128,
        logging_steps=25,
        load_best_model_at_end=has_eval,
        metric_for_best_model="wer" if has_eval else None,
        greater_is_better=False,
        remove_unused_columns=False,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"] if has_eval else None,
        data_collator=data_collator,
        compute_metrics=(
            (lambda p: compute_metrics(p, processor.tokenizer, metric))
            if has_eval else None
        ),
        processing_class=processor,
    )

    # --- Entraîner ---
    print("\n" + "=" * 60)
    print("  Démarrage de l'entraînement")
    print("=" * 60 + "\n")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # --- Sauvegarder ---
    final_dir = os.path.join(output_dir, "final")
    print(f"\n  Sauvegarde du modèle dans {final_dir}...")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)

    print("\n" + "=" * 60)
    print("  Entraînement terminé")
    print("=" * 60)
    print(f"\n  Modèle : {final_dir}")

    if has_eval:
        print("\n  Évaluation finale...")
        metrics = trainer.evaluate()
        print(f"  WER : {metrics.get('eval_wer', float('nan')):.2f}%")

    print("\n  Étapes suivantes :")
    print(f"    1. Comparer à Whisper : python utils/test_moonshine.py --moonshine_model {final_dir}")
    print(f"    2. Exporter en ONNX   : python fine_tuning/export_moonshine_onnx.py --model {final_dir}")
    print()


if __name__ == "__main__":
    main()
