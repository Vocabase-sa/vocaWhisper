"""
Fine-tuning de wav2vec2 (CTC) sur ton corpus.
==============================================

wav2vec2 est un modèle CTC : il classe chaque trame audio en caractère, sans
décodeur autorégressif. Le fine-tuning en est plus simple et plus stable que
celui de Whisper — pas de décalage de labels, pas de risque de boucle de
génération — mais il impose une contrainte forte sur les transcriptions.

CONTRAINTE — le vocabulaire est fermé :
    Le tokenizer CTC ne connaît que les caractères de son vocabulaire, fixé à
    l'entraînement du modèle de base. Pour jonatasgrosman/wav2vec2-large-xlsr-53-french,
    c'est 59 tokens : minuscules, accents français, apostrophe, tiret.
    Ni majuscules, ni ponctuation, ni chiffres.

    Les transcriptions sont donc normalisées automatiquement (minuscules,
    ponctuation retirée). « Je voudrais le docteur AGGOUR. » devient
    « je voudrais le docteur aggour ». Les caractères hors vocabulaire sont
    signalés puis supprimés — c'est normal et sans gravité, mais surveillez le
    compte : s'il est élevé, le modèle de base ne convient pas à votre corpus.

Le dataset préparé par prepare_dataset.py est réutilisé tel quel.

Usage :
    python fine_tuning/train_wav2vec2.py
    python fine_tuning/train_wav2vec2.py --epochs 20 --batch_size 8
    python fine_tuning/train_wav2vec2.py --base_model bofenghuang/asr-wav2vec2-ctc-french

Conseil RTX 4090 :
    Le modèle fait 315M paramètres et consomme beaucoup de VRAM sur les
    séquences longues. batch_size=8 avec gradient_accumulation=2 est un bon
    point de départ ; réduire à 4 si la mémoire sature.
"""

import argparse
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from datasets import DatasetDict, load_from_disk
from transformers import (
    AutoModelForCTC,
    Trainer,
    TrainingArguments,
    Wav2Vec2Processor,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_prepared")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_wav2vec2")

DEFAULT_BASE_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-french"

SAMPLE_RATE = 16000


# =============================================================================
# Normalisation des transcriptions
# =============================================================================
def build_normalizer(tokenizer):
    """Construit un normaliseur adapté au vocabulaire du tokenizer CTC.

    Retourne (fonction de normalisation, ensemble des caractères autorisés).
    """
    vocab = {k for k in tokenizer.get_vocab() if len(k) == 1}
    has_upper = any(c.isupper() for c in vocab)

    def normalize(text: str) -> str:
        if not has_upper:
            text = text.lower()

        # Uniformiser les apostrophes et tirets typographiques
        text = text.replace("’", "'").replace("‘", "'")
        text = text.replace("–", "-").replace("—", "-")

        # Retirer les caractères absents du vocabulaire, en gardant les espaces
        kept = []
        for char in text:
            if char == " " or char in vocab:
                kept.append(char)
            elif unicodedata.category(char).startswith("P"):
                kept.append(" ")  # ponctuation -> séparateur
            # sinon : caractère inconnu, supprimé

        return re.sub(r"\s+", " ", "".join(kept)).strip()

    return normalize, vocab


# =============================================================================
# Data Collator CTC
# =============================================================================
@dataclass
class DataCollatorCTC:
    """Pad l'audio et les labels indépendamment.

    Bien plus simple qu'un collator seq2seq : CTC aligne seul les trames sur
    les caractères, il n'y a donc ni decoder_input_ids ni décalage à gérer.
    """
    processor: Any

    def __call__(self, features: list[dict]) -> dict:
        input_features = [{"input_values": f["input_values"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt", return_attention_mask=True,
        )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # -100 = ignoré par la CTC loss
        batch["labels"] = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        return batch


# =============================================================================
# Préparation et métriques
# =============================================================================
def prepare_entry(batch, processor, normalize):
    """Audio -> input_values, texte normalisé -> ids de caractères."""
    audio = batch["audio"]
    batch["input_values"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"],
    ).input_values[0]
    batch["n_samples"] = len(batch["input_values"])

    text = normalize(batch["sentence"])
    batch["labels"] = processor.tokenizer(text).input_ids
    batch["normalized"] = text
    return batch


def compute_metrics(pred, processor, metric):
    """WER sur les prédictions décodées en CTC greedy."""
    pred_ids = np.argmax(pred.predictions, axis=-1)

    label_ids = pred.label_ids
    label_ids = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.tokenizer.batch_decode(label_ids, group_tokens=False)

    return {"wer": 100 * metric.compute(predictions=pred_str, references=label_str)}


# =============================================================================
# Entraînement
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Fine-tuning wav2vec2 (CTC)")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--max_duration", type=float, default=20.0)
    parser.add_argument("--freeze_encoder", action="store_true", default=True,
                        help="Geler l'extracteur convolutionnel (recommandé)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--no_fp16", action="store_true")
    args = parser.parse_args()

    dataset_dir = args.dataset or DATASET_DIR
    output_dir = args.output or OUTPUT_DIR

    print("=" * 60)
    print("  Fine-tuning wav2vec2 (CTC)")
    print("=" * 60)
    print(f"\n  Modèle de base : {args.base_model}")
    print(f"  Dataset        : {dataset_dir}")
    print(f"  Output         : {output_dir}")
    print(f"  Époques        : {args.epochs}")
    print(f"  Batch size     : {args.batch_size}")
    print(f"  Learning rate  : {args.learning_rate}")

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print(f"  Device         : {torch.cuda.get_device_name(0)} (CUDA)")
    else:
        print("  Device         : CPU (l'entraînement sera très lent)")
    print()

    # --- Dataset ---
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

    # --- Processeur et modèle ---
    print(f"\n  Chargement du processeur depuis {args.base_model}...")
    try:
        processor = Wav2Vec2Processor.from_pretrained(args.base_model)
    except Exception as e:
        print(f"\n[ERREUR] Processeur illisible : {e}")
        print("Astuce : les modèles avec décodeur à LM (Wav2Vec2ProcessorWithLM)")
        print("ne sont pas gérés ici — choisissez un modèle CTC simple.")
        sys.exit(1)

    normalize, vocab = build_normalizer(processor.tokenizer)
    print(f"  Vocabulaire CTC : {len(vocab)} caractères")

    print(f"  Chargement du modèle...")
    model = AutoModelForCTC.from_pretrained(
        args.base_model,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    print(f"  Paramètres : {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

    # L'extracteur convolutionnel encode des traits acoustiques génériques :
    # le regeler évite de le dégrader sur un corpus de quelques milliers
    # d'exemples, et accélère nettement l'entraînement.
    if args.freeze_encoder:
        model.freeze_feature_encoder()
        print("  Extracteur convolutionnel gelé")

    # --- Préparation ---
    print("\n  Préparation des features (forme d'onde brute)...")
    original_columns = dataset["train"].column_names
    dataset = dataset.map(
        lambda b: prepare_entry(b, processor, normalize),
        remove_columns=original_columns,
        num_proc=1,
        desc="Extraction audio",
    )

    # Contrôle de la normalisation : montrer ce que le modèle va réellement voir
    print("\n  Exemples après normalisation :")
    for i in range(min(3, len(dataset["train"]))):
        print(f"    {dataset['train'][i]['normalized']!r}")

    # --- Filtrage des audios trop longs ---
    max_samples = int(args.max_duration * SAMPLE_RATE)
    before = len(dataset["train"])
    dataset = dataset.filter(lambda n: n <= max_samples, input_columns=["n_samples"])
    dropped = before - len(dataset["train"])
    if dropped:
        print(f"\n  {dropped} exemple(s) écarté(s) : durée > {args.max_duration}s")

    durations = np.array(dataset["train"]["n_samples"]) / SAMPLE_RATE
    print(f"  Durée audio : moyenne {durations.mean():.1f}s, "
          f"total {durations.sum() / 3600:.2f}h")

    dataset = dataset.remove_columns(["n_samples", "normalized"])

    # --- Métrique ---
    import evaluate
    metric = evaluate.load("wer")
    has_eval = "test" in dataset and len(dataset["test"]) > 0

    training_args = TrainingArguments(
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
        logging_steps=25,
        load_best_model_at_end=has_eval,
        metric_for_best_model="wer" if has_eval else None,
        greater_is_better=False,
        # group_by_length n'existe plus dans transformers 5.x. Sans lui, chaque
        # batch est padé à son plus long élément ; sur ce corpus (1,8 s de
        # moyenne, faible dispersion) le surcoût reste marginal.
        remove_unused_columns=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"] if has_eval else None,
        data_collator=DataCollatorCTC(processor=processor),
        compute_metrics=(
            (lambda p: compute_metrics(p, processor, metric)) if has_eval else None
        ),
        processing_class=processor,
    )

    print("\n" + "=" * 60)
    print("  Démarrage de l'entraînement")
    print("=" * 60 + "\n")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_dir = os.path.join(output_dir, "final")
    print(f"\n  Sauvegarde du modèle dans {final_dir}...")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)

    print("\n" + "=" * 60)
    print("  Entraînement terminé")
    print("=" * 60)
    print(f"\n  Modèle : {final_dir}")

    if has_eval:
        metrics = trainer.evaluate()
        print(f"  WER : {metrics.get('eval_wer', float('nan')):.2f}%")

    print("\n  Pour l'utiliser : Paramètres > Moteur STT = wav2vec2,")
    print(f"  puis « Modèle wav2vec2 » = {final_dir}")
    print()


if __name__ == "__main__":
    main()
