"""Serveur Flask pour l'API HTTP de transcription VocaWhisper."""

import logging
import queue
import threading
import time

from flask import Flask, request, jsonify

from api.audio_utils import wav_to_numpy

logger = logging.getLogger(__name__)

# Références injectées par start_api_server()
_state = None
_config = None
_transcribe_fn = None
_server_thread = None
_rtp_initialized = False

# --- Suivi de l'inférence, pour que /health teste le chemin réel ---
_meta_lock = threading.Lock()
_inflight_since = None   # timestamp de début de l'inférence en cours
_last_ok = None          # timestamp de la dernière transcription réussie
_started_at = time.time()

# Au-delà, on considère le moteur bloqué (boucle de répétition Whisper).
STUCK_AFTER_S = 180
# Au-delà, on refuse la requête plutôt que d'attendre indéfiniment le verrou.
LOCK_TIMEOUT_S = 120

# --- Worker d'inférence : UN SEUL thread pour toutes les transcriptions ---
# Werkzeug traite chaque requête dans un thread neuf, et CTranslate2/CUDA
# attachent au thread appelant des ressources (contexte, handles cuBLAS,
# workspaces) que la mort du thread Python ne libère pas. Mesuré : +24 % de RTF
# par millier de requêtes en thread neuf, contre -0,2 % en thread unique.
# En routant toutes les inférences vers un thread stable, l'accumulation cesse.
_infer_queue = queue.Queue(maxsize=32)
_worker_thread = None


def _inference_worker():
    """Exécute toutes les inférences dans un thread unique et stable."""
    global _inflight_since, _last_ok
    while True:
        job = _infer_queue.get()
        if job is None:
            break
        audio, box, done = job
        # Le client a renoncé (timeout HTTP) : ne pas gaspiller le GPU.
        if box.get("abandoned"):
            done.set()
            continue
        try:
            with _state.lock:          # cohérence avec la dictée micro
                with _meta_lock:
                    _inflight_since = time.time()
                box["text"] = _transcribe_fn(audio)
                with _meta_lock:
                    _last_ok = time.time()
        except Exception as e:
            box["error"] = e
        finally:
            with _meta_lock:
                _inflight_since = None
            done.set()


def _create_app():
    """Crée et configure l'application Flask."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB max

    @app.route("/health", methods=["GET"])
    def health():
        engine = _config.get("stt_engine", "local") if _config else "local"
        if engine == "groq":
            model_ready = bool(_config.get("groq_api_key", "").strip())
        else:
            model_ready = _state is not None and _state.model is not None
        now = time.time()
        with _meta_lock:
            started, last_ok = _inflight_since, _last_ok

        # Une inférence qui dure au-delà du seuil = moteur parti en boucle.
        # C'est ce cas qui a laissé l'API répondre « ok » pendant 10 h alors
        # qu'aucune transcription n'aboutissait (incident 2026-09-04 21:47 UTC).
        busy_for = (now - started) if started else 0.0
        stuck = busy_for > STUCK_AFTER_S
        healthy = model_ready and not stuck

        return jsonify({
            "status": "stuck" if stuck else ("ok" if model_ready else "loading"),
            "model_loaded": model_ready,
            "stt_engine": engine,
            "language": _config.get("language", "fr") if _config else None,
            "inference_busy_s": round(busy_for, 1),
            "last_success_age_s": round(now - last_ok, 1) if last_ok else None,
            "uptime_s": round(now - _started_at, 1),
        }), (200 if healthy else 503)

    @app.route("/transcribe", methods=["POST"])
    def transcribe_endpoint():
        # Vérifier que le moteur STT est prêt
        is_groq = _config and _config.get("stt_engine") == "groq"
        if _state is None or (_state.model is None and not is_groq):
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

        # Soumettre au thread d'inférence unique. La file remplace le verrou :
        # elle sérialise naturellement, et surtout l'inférence s'exécute toujours
        # dans le même thread (voir _inference_worker).
        box, done = {}, threading.Event()
        try:
            _infer_queue.put_nowait((audio, box, done))
        except queue.Full:
            logger.error("File d'inférence saturée (%d en attente)", _infer_queue.maxsize)
            return jsonify({"error": "File d'inférence saturée, réessayez plus tard."}), 503

        if not done.wait(timeout=LOCK_TIMEOUT_S):
            # Marquer abandonné : si le worker n'a pas encore pris ce job, il
            # le sautera au lieu de calculer un résultat que personne n'attend.
            box["abandoned"] = True
            logger.error("Inférence non terminée après %ss", LOCK_TIMEOUT_S)
            return jsonify({
                "error": "Moteur de transcription occupé, réessayez plus tard."
            }), 503

        if "error" in box:
            logger.error("Erreur de transcription : %s", box["error"])
            return jsonify({"error": f"Erreur de transcription : {box['error']}"}), 500
        text = box["text"]

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

    # --- Thread d'inférence unique (évite l'accumulation de ressources CUDA) ---
    global _worker_thread
    if _worker_thread is None:
        _worker_thread = threading.Thread(target=_inference_worker,
                                          name="inference-worker", daemon=True)
        _worker_thread.start()
        logger.info("Thread d'inférence unique démarré.")

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
    """Initialise le module RTP : config, pool Whisper, Blueprint, auto-start.

    Ne fait rien si rtp_enabled=False pour éviter des imports inutiles.
    """
    global _rtp_initialized

    if _rtp_initialized:
        return

    # Ne pas charger les modules RTP si RTP est désactivé
    if not config.get("rtp_enabled", False):
        logger.info("[RTP] Module RTP désactivé, import ignoré.")
        return

    try:
        # 1. Appliquer la config RTP
        from api.rtp_config import apply_config
        apply_config(config)

        # 2. Initialiser le pool de modèles Whisper
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

        # 4. Auto-démarrer le listener
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
