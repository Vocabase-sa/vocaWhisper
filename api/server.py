"""Serveur Flask pour l'API HTTP de transcription VocaWhisper."""

import logging
import threading

from flask import Flask, request, jsonify

from api.audio_utils import wav_to_numpy

logger = logging.getLogger(__name__)

# Références injectées par start_api_server()
_state = None
_config = None
_transcribe_fn = None
_server_thread = None
_rtp_initialized = False


def _create_app():
    """Crée et configure l'application Flask."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB max

    @app.route("/health", methods=["GET"])
    def health():
        model_loaded = _state is not None and _state.model is not None
        return jsonify({
            "status": "ok",
            "model_loaded": model_loaded,
            "language": _config.get("language", "fr") if _config else None,
        })

    @app.route("/transcribe", methods=["POST"])
    def transcribe_endpoint():
        # Vérifier que le modèle est chargé
        if _state is None or _state.model is None:
            return jsonify({"error": "Modèle pas encore chargé"}), 503

        # Vérifier la présence du fichier
        if "file" not in request.files:
            return jsonify({
                "error": "Pas de champ 'file' dans la requête. "
                         "Envoyez un fichier WAV en multipart/form-data avec la clé 'file'."
            }), 400

        uploaded = request.files["file"]
        if uploaded.filename == "":
            return jsonify({"error": "Nom de fichier vide"}), 400

        # Lire et convertir le WAV
        try:
            file_bytes = uploaded.read()
            audio = wav_to_numpy(file_bytes)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.exception("Erreur traitement audio")
            return jsonify({"error": f"Erreur traitement audio : {e}"}), 400

        # Transcrire (avec lock, le modèle n'est pas thread-safe)
        try:
            with _state.lock:
                text = _transcribe_fn(audio)
        except Exception as e:
            logger.exception("Erreur de transcription")
            return jsonify({"error": f"Erreur de transcription : {e}"}), 500

        return jsonify({
            "text": text,
            "language": _config.get("language", "fr"),
        })

    return app


def start_api_server(state, config, transcribe_fn):
    """Démarre le serveur Flask dans un thread daemon.

    Args:
        state: instance AppState (fournit state.model et state.lock)
        config: dict de configuration (api_host, api_port)
        transcribe_fn: fonction transcribe(audio) -> str
    """
    global _state, _config, _transcribe_fn, _server_thread

    if not config.get("api_enabled", False):
        logger.info("API HTTP désactivée dans la config.")
        return

    _state = state
    _config = config
    _transcribe_fn = transcribe_fn

    host = config.get("api_host", "0.0.0.0")
    port = config.get("api_port", 5000)

    app = _create_app()

    # --- Initialiser et enregistrer le module RTP si activé ---
    _init_rtp_module(app, config)

    def _run():
        try:
            # use_reloader=False est indispensable dans un thread
            app.run(host=host, port=int(port), debug=False, use_reloader=False)
        except OSError as e:
            logger.error(f"Impossible de démarrer l'API sur {host}:{port} — {e}")
        except Exception as e:
            logger.exception(f"Erreur serveur API : {e}")

    _server_thread = threading.Thread(target=_run, name="api-server", daemon=True)
    _server_thread.start()
    logger.info(f"API HTTP démarrée sur {host}:{port}")


def _init_rtp_module(app, config):
    """Initialise le module RTP : config, pool Whisper, Blueprint, auto-start."""
    global _rtp_initialized

    if _rtp_initialized:
        return

    try:
        # 1. Appliquer la config RTP
        from api.rtp_config import apply_config
        apply_config(config)

        # 2. Initialiser le pool de modèles Whisper (si RTP activé)
        if config.get("rtp_enabled", False):
            from api.whisper_pool import initialize_pool
            logger.info("[RTP] Initialisation du pool de modèles Whisper pour RTP...")
            success = initialize_pool(config)
            if success:
                from api.whisper_pool import get_pool_stats
                stats = get_pool_stats()
                logger.info(
                    f"[RTP] Pool initialisé : {stats['total']} modèle(s), "
                    f"{stats['available']} disponible(s)"
                )
            else:
                logger.error("[RTP] Échec de l'initialisation du pool Whisper")

        # 3. Enregistrer le Blueprint RTP
        from api.rtp_routes import rtp_bp
        app.register_blueprint(rtp_bp)
        logger.info("[RTP] Blueprint /rtp enregistré")

        # 4. Auto-démarrer le listener si rtp_enabled=True
        if config.get("rtp_enabled", False):
            from api.rtp_routes import auto_start_listener
            auto_start_listener(config)

        _rtp_initialized = True

    except Exception as e:
        logger.error(f"[RTP] Erreur d'initialisation du module RTP : {e}")
        import traceback
        logger.error(traceback.format_exc())


def stop_api_server():
    """Arrête le serveur API. Le thread daemon sera tué à la fermeture du process."""
    global _server_thread
    if _server_thread is not None:
        logger.info("Arrêt du serveur API.")
        _server_thread = None
