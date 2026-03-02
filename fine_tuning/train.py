"""
Fine-tuning de Whisper large-v3 avec Hugging Face Transformers.
================================================================

Entraîne le modèle Whisper sur ton dataset personnalisé pour améliorer
la reconnaissance de ton vocabulaire spécifique (VoIP, IoT, noms propres, etc.).

Prérequis :
    1. Avoir préparé le dataset : python fine_tuning/prepare_dataset.py
    2. Avoir installé les dépendances : pip install -r fine_tuning/requirements.txt

Usage :
    python fine_tuning/train.py
    python fine_tuning/train.py --epochs 5 --batch_size 4
    python fine_tuning/train.py --base_model bofenghuang/whisper-large-v3-french
    python fine_tuning/train.py --resume_from_checkpoint fine_tuning/output/checkpoint-500

Conseil RTX 4090 (24 Go VRAM) :
    - batch_size=8 avec gradient_accumulation=2 fonctionne bien
    - Pour large-v3, fp16 est recommandé
    - L'entraînement prend ~1-3h pour quelques heures d'audio
"""

import argparse
import os
import sys

import evaluate
import torch
from datasets import DatasetDict, load_from_disk
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)
from dataclasses import dataclass
from typing import Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_prepared")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


# =============================================================================
# Data Collator pour Whisper
# =============================================================================
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Collator qui pad les features audio et les labels texte."""
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        # Séparer audio (input_features) et texte (labels)
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad les labels
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Remplacer le padding par -100 (ignoré par la loss)
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Retirer le token de début s'il est présent (ajouté automatiquement)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# =============================================================================
# Préparation des données
# =============================================================================
def prepare_dataset_entry(batch, processor):
    """Prépare une entrée du dataset : audio -> features, texte -> tokens."""
    audio = batch["audio"]

    # Extraire les features audio (log-Mel spectrogram)
    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
    ).input_features[0]

    # Tokeniser la transcription
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch


# =============================================================================
# Métriques
# =============================================================================
def compute_metrics(pred, tokenizer, metric):
    """Calcule le WER (Word Error Rate) sur les prédictions."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Remplacer -100 par le token de padding
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Décoder
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Calculer WER
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# =============================================================================
# Entraînement
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Fine-tuning Whisper")
    parser.add_argument("--base_model", type=str, default="openai/whisper-large-v3",
                        help="Modèle de base (défaut: openai/whisper-large-v3)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Chemin du dataset préparé (défaut: fine_tuning/dataset_prepared/)")
    parser.add_argument("--output", type=str, default=None,
                        help="Dossier de sortie (défaut: fine_tuning/output/)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Nombre d'époques (défaut: 3)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Taille de batch par GPU (défaut: 8)")
    parser.add_argument("--gradient_accumulation", type=int, default=2,
                        help="Étapes d'accumulation de gradient (défaut: 2)")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate (défaut: 1e-5)")
    parser.add_argument("--warmup_steps", type=int, default=50,
                        help="Étapes de warmup (défaut: 50)")
    parser.add_argument("--language", type=str, default="fr",
                        help="Langue cible (défaut: fr)")
    parser.add_argument("--task", type=str, default="transcribe",
                        help="Tâche Whisper (défaut: transcribe)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Reprendre depuis un checkpoint")
    args = parser.parse_args()

    dataset_dir = args.dataset or DATASET_DIR
    output_dir = args.output or OUTPUT_DIR

    print("=" * 60)
    print("  Fine-tuning Whisper")
    print("=" * 60)
    print(f"\n  Modèle de base : {args.base_model}")
    print(f"  Dataset        : {dataset_dir}")
    print(f"  Output         : {output_dir}")
    print(f"  Époques        : {args.epochs}")
    print(f"  Batch size     : {args.batch_size}")
    print(f"  Grad accum     : {args.gradient_accumulation}")
    print(f"  Learning rate  : {args.learning_rate}")
    print(f"  Langue         : {args.language}")
    if torch.cuda.is_available():
        print(f"  Device         : {torch.cuda.get_device_name(0)} (CUDA)")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM           : {vram:.1f} Go")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("  Device         : Apple Silicon (MPS)")
    else:
        print("  Device         : CPU")
    print()

    # --- Charger le dataset ---
    print("  Chargement du dataset...")
    if not os.path.isdir(dataset_dir):
        print(f"\n[ERREUR] Dataset introuvable : {dataset_dir}")
        print("Lance d'abord : python fine_tuning/prepare_dataset.py")
        sys.exit(1)

    dataset = load_from_disk(dataset_dir)
    if isinstance(dataset, DatasetDict):
        print(f"  Train : {len(dataset['train'])} exemples")
        if "test" in dataset:
            print(f"  Test  : {len(dataset['test'])} exemples")
    else:
        print(f"  Total : {len(dataset)} exemples")

    # --- Charger le processeur Whisper ---
    print(f"\n  Chargement du processeur depuis {args.base_model}...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.base_model)
    tokenizer = WhisperTokenizer.from_pretrained(
        args.base_model,
        language=args.language,
        task=args.task,
    )
    processor = WhisperProcessor.from_pretrained(
        args.base_model,
        language=args.language,
        task=args.task,
    )

    # --- Préparer le dataset ---
    print("\n  Préparation des features audio...")
    dataset = dataset.map(
        lambda batch: prepare_dataset_entry(batch, processor),
        remove_columns=dataset["train"].column_names if isinstance(dataset, DatasetDict) else dataset.column_names,
        num_proc=1,  # Pas de multiprocessing pour éviter les problèmes audio
    )

    # --- Charger le modèle ---
    print(f"\n  Chargement du modèle {args.base_model}...")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.float32
    )

    # Configurer pour le français
    model.generation_config.language = args.language
    model.generation_config.task = args.task
    model.generation_config.forced_decoder_ids = None

    # Geler l'encoder pour éviter le catastrophic forgetting (surtout avec peu de données)
    model.freeze_encoder()
    print("  [INFO] Encoder gelé — seul le decoder sera fine-tuné")

    # --- Data Collator ---
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # --- Métriques ---
    wer_metric = evaluate.load("wer")

    # --- Configuration de l'entraînement ---
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        fp16=False,  # Désactivé : fp16 cause des erreurs de gradient scaling
        bf16=False,  # Note : bf16 corrompt les poids à la sauvegarde sur certaines configs
        eval_strategy="epoch" if "test" in dataset else "no",
        save_strategy="epoch",
        logging_steps=25,
        load_best_model_at_end="test" in dataset,
        metric_for_best_model="wer" if "test" in dataset else None,
        greater_is_better=False,
        push_to_hub=False,
        report_to=["tensorboard"],
        predict_with_generate=True,
        generation_max_length=225,
        save_total_limit=3,
        dataloader_num_workers=0,  # 0 = plus stable sur toutes les plateformes
        remove_unused_columns=False,
    )

    # --- Trainer ---
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"] if isinstance(dataset, DatasetDict) else dataset,
        eval_dataset=dataset.get("test") if isinstance(dataset, DatasetDict) else None,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, tokenizer, wer_metric),
        processing_class=processor.feature_extractor,
    )

    # --- Lancer l'entraînement ---
    print("\n" + "=" * 60)
    print("  Lancement de l'entraînement...")
    print("=" * 60)
    print("  (TensorBoard : tensorboard --logdir fine_tuning/output/)")
    print()

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # --- Sauvegarder le modèle final ---
    final_dir = os.path.join(output_dir, "final")
    print(f"\n  Sauvegarde du modèle final dans : {final_dir}")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    print("\n" + "=" * 60)
    print("  Fine-tuning terminé !")
    print(f"  Modèle sauvegardé dans : {final_dir}")
    print(f"\n  Prochaine étape : python fine_tuning/convert_to_ct2.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
