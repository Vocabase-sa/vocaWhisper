"""Endpoint Modal pour VocaWhisper fine-tuné — API compatible OpenAI/Groq.

Expose une API HTTP qui imite l'API Groq Whisper :

    POST /v1/audio/transcriptions   (multipart: file, model, language,
                                      prompt, temperature, response_format)
    -> {"text": ..., "language": ..., "duration": ...}

But : remplacer l'appel Groq dans stt-api par un appel à ce service, sans
toucher au reste (VAD Silero, buffering, simili-streaming, fuzzy + corrections
restent côté stt-api).

Le modèle CTranslate2 fine-tuné (fine_tuning/model_ct2) est servi via
faster-whisper sur GPU, avec scale-to-zero pour ne payer qu'à l'usage.

Déploiement :
    pip install modal
    modal setup
    modal volume create vocawhisper-model
    modal volume put vocawhisper-model ./fine_tuning/model_ct2 /model_ct2
    modal secret create vocawhisper-token API_TOKEN=<un-token-secret>
    modal deploy modal_app.py
"""
import os

import modal

app = modal.App("vocawhisper")

# Image CUDA + cuDNN (requis par CTranslate2 / faster-whisper sur GPU)
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.11"
    )
    .pip_install(
        "faster-whisper==1.0.3",
        "fastapi[standard]",
        "python-multipart",
    )
)

# Volume contenant le modèle uploadé (cf. `modal volume put`)
model_volume = modal.Volume.from_name("vocawhisper-model")
MODEL_DIR = "/models/model_ct2"

# Secret contenant API_TOKEN=... pour l'authentification Bearer
secrets = [modal.Secret.from_name("vocawhisper-token")]

# Paramètres de transcription — identiques au local (whisper_dictation.py)
BEAM_SIZE = 5


@app.cls(
    gpu="T4",                 # ~0,6 $/h ; passer à "L4" si trop lent
    image=image,
    volumes={"/models": model_volume},
    secrets=secrets,
    scaledown_window=60,      # scale-to-zero après 60 s d'inactivité
    timeout=600,
)
@modal.concurrent(max_inputs=4)   # plusieurs requêtes par conteneur GPU
class VocaWhisper:
    @modal.enter()
    def load(self):
        """Charge le modèle une seule fois au démarrage du conteneur."""
        from faster_whisper import WhisperModel

        self.model = WhisperModel(
            MODEL_DIR, device="cuda", compute_type="float16"
        )

    @modal.asgi_app()
    def web(self):
        import io

        from fastapi import (
            FastAPI, File, Form, Header, HTTPException, UploadFile,
        )

        api = FastAPI(title="VocaWhisper", version="1.0")
        expected_token = os.environ.get("API_TOKEN", "")

        def _check_auth(authorization: str | None):
            """Vérifie le header Authorization: Bearer <token>."""
            if not expected_token:
                return  # pas de token configuré -> auth désactivée
            token = (authorization or "").removeprefix("Bearer ").strip()
            if token != expected_token:
                raise HTTPException(status_code=401, detail="Invalid token")

        @api.get("/health")
        def health():
            return {"status": "ok", "model": "vocawhisper-ct2"}

        @api.post("/v1/audio/transcriptions")
        async def transcribe(
            file: UploadFile = File(...),
            model: str = Form("vocawhisper"),
            language: str = Form("fr"),
            prompt: str | None = Form(None),
            temperature: float = Form(0.0),
            response_format: str = Form("json"),
            authorization: str | None = Header(None),
        ):
            _check_auth(authorization)

            audio_bytes = await file.read()

            # faster-whisper décode le WAV via PyAV (inclus) ; vad_filter=False
            # car la VAD Silero est déjà faite côté stt-api.
            segments, info = self.model.transcribe(
                io.BytesIO(audio_bytes),
                language=language,
                beam_size=BEAM_SIZE,
                vad_filter=False,
                initial_prompt=prompt or None,
                temperature=temperature,
            )

            text = " ".join(seg.text.strip() for seg in segments).strip()

            return {
                "text": text,
                "language": info.language,
                "duration": info.duration,
            }

        return api
