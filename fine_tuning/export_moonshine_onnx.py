"""
Export d'un modèle Moonshine fine-tuné vers ONNX.
==================================================

L'export ONNX sert le déploiement CPU : sur la voie RTP, il permet de servir
des sessions sans mobiliser de VRAM, là où le pool faster-whisper réserve
~3 Go par instance.

Prérequis :
    pip install "optimum[onnxruntime]" onnx

Usage :
    python fine_tuning/export_moonshine_onnx.py
    python fine_tuning/export_moonshine_onnx.py --model fine_tuning/output_moonshine/final
    python fine_tuning/export_moonshine_onnx.py --quantize   # INT8, ~4x plus léger

Le modèle exporté se charge ensuite via :
    config.json -> "moonshine_backend": "onnx"
                   "moonshine_model": "<dossier de sortie>"
"""

import argparse
import os
import shutil
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(BASE_DIR, "output_moonshine", "final")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "output_moonshine", "onnx")


def check_dependencies():
    """Vérifie qu'optimum et onnx sont disponibles."""
    missing = []
    try:
        import optimum  # noqa: F401
    except ImportError:
        missing.append("optimum[onnxruntime]")
    try:
        import onnx  # noqa: F401
    except ImportError:
        missing.append("onnx")

    if missing:
        print("\n[ERREUR] Dépendances manquantes pour l'export ONNX :")
        for m in missing:
            print(f"    pip install {m}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Export Moonshine vers ONNX")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Modèle à exporter (défaut: {DEFAULT_MODEL})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"Dossier de sortie (défaut: {DEFAULT_OUTPUT})")
    parser.add_argument("--quantize", action="store_true",
                        help="Quantifier en INT8 après l'export")
    parser.add_argument("--opset", type=int, default=17,
                        help="Version de l'opset ONNX (défaut: 17)")
    args = parser.parse_args()

    check_dependencies()

    if not os.path.isdir(args.model) and "/" not in args.model:
        print(f"\n[ERREUR] Modèle introuvable : {args.model}")
        print("Lance d'abord : python fine_tuning/train_moonshine.py")
        sys.exit(1)

    print("=" * 60)
    print("  Export Moonshine -> ONNX")
    print("=" * 60)
    print(f"\n  Modèle : {args.model}")
    print(f"  Sortie : {args.output}")
    print(f"  Opset  : {args.opset}")
    print(f"  INT8   : {'oui' if args.quantize else 'non'}\n")

    from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
    from transformers import AutoProcessor

    print("  Export en cours (quelques minutes)...")
    model = ORTModelForSpeechSeq2Seq.from_pretrained(args.model, export=True)
    model.save_pretrained(args.output)

    # Le processeur doit accompagner le modèle pour l'inférence
    processor = AutoProcessor.from_pretrained(args.model)
    processor.save_pretrained(args.output)
    print(f"  Export terminé : {args.output}")

    # --- Quantification INT8 optionnelle ---
    if args.quantize:
        print("\n  Quantification INT8...")
        try:
            from optimum.onnxruntime import ORTQuantizer
            from optimum.onnxruntime.configuration import AutoQuantizationConfig

            qconfig = AutoQuantizationConfig.avx512_vnni(
                is_static=False, per_channel=False
            )
            quant_dir = args.output + "_int8"
            os.makedirs(quant_dir, exist_ok=True)

            # Un quantizer par sous-graphe (encodeur / décodeur)
            onnx_files = [f for f in os.listdir(args.output) if f.endswith(".onnx")]
            for onnx_file in onnx_files:
                quantizer = ORTQuantizer.from_pretrained(args.output, file_name=onnx_file)
                quantizer.quantize(save_dir=quant_dir, quantization_config=qconfig)
                print(f"    {onnx_file} quantifié")

            # Recopier la configuration et le tokenizer
            for name in os.listdir(args.output):
                if not name.endswith(".onnx"):
                    src = os.path.join(args.output, name)
                    if os.path.isfile(src):
                        shutil.copy2(src, os.path.join(quant_dir, name))

            print(f"  Modèle INT8 : {quant_dir}")
        except Exception as e:
            print(f"  [WARN] Quantification échouée : {e}")
            print("  Le modèle ONNX non quantifié reste utilisable.")

    # --- Récapitulatif des tailles ---
    def dir_size(path):
        return sum(
            os.path.getsize(os.path.join(path, f))
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        )

    print("\n  Tailles :")
    if os.path.isdir(args.model):
        print(f"    PyTorch : {dir_size(args.model) / 1e6:.0f} Mo")
    print(f"    ONNX    : {dir_size(args.output) / 1e6:.0f} Mo")
    quant_dir = args.output + "_int8"
    if os.path.isdir(quant_dir):
        print(f"    INT8    : {dir_size(quant_dir) / 1e6:.0f} Mo")

    print("\n  Pour l'utiliser, dans config.json :")
    print('    "stt_engine": "moonshine",')
    print('    "moonshine_backend": "onnx",')
    print(f'    "moonshine_model": "{args.output.replace(os.sep, "/")}"')
    print()


if __name__ == "__main__":
    main()
