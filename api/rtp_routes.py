"""Blueprint Flask pour le controle de l'ecouteur RTP.

Endpoints :
  POST /rtp/start    - Demarre l'ecouteur
  POST /rtp/stop     - Arrete l'ecouteur
  GET  /rtp/status   - Statut de l'ecouteur
  GET  /rtp/sessions - Sessions actives
  POST /rtp/close_session/<client_key> - Ferme une session
  GET  /rtp/pool     - Stats du pool Whisper

Inspire de stt-api/routes/rtp_routes.py (FastAPI -> Flask Blueprint).
"""

import threading
import time

from flask import Blueprint, jsonify, request

from api.rtp_config import RTP_PORT, logger
from api.rtp_listener import RTPListener
from api.whisper_pool import get_pool_stats

rtp_bp = Blueprint("rtp", __name__, url_prefix="/rtp")

# --- Instance globale du listener ---
_rtp_listener = None
_listener_lock = threading.Lock()


# =====================================================================
# Fonctions internes
# =====================================================================
def _start_listener(webhook_url=None, client_id=None, language="fr"):
    """Demarre un ecouteur RTP dans un thread daemon."""
    global _rtp_listener

    with _listener_lock:
        if _rtp_listener and _rtp_listener.is_running:
            _rtp_listener.stop()
            time.sleep(1)

        _rtp_listener = RTPListener()

        listener_thread = threading.Thread(
            target=_rtp_listener.start,
            args=(webhook_url, client_id, language),
            name="rtp-listener",
            daemon=True,
        )
        listener_thread.start()

    logger.info(
        f"Ecouteur RTP demarre (webhook={webhook_url}, language={language})"
    )
    return {
        "status": "started",
        "port": RTP_PORT,
        "multi_client": True,
        "language": language,
    }


def _stop_listener():
    """Arrete l'ecouteur RTP."""
    global _rtp_listener

    with _listener_lock:
        if _rtp_listener:
            _rtp_listener.stop()
            _rtp_listener = None
            return {"status": "stopped"}
    return {"status": "not_running"}


def _get_active_sessions():
    """Liste des sessions actives."""
    if not _rtp_listener:
        return []

    sessions = []
    with _rtp_listener.sessions_lock:
        for client_key, session in _rtp_listener.sessions.items():
            sessions.append({
                "client_key": client_key,
                "client_id": session.client_id,
                "session_id": session.session_id,
                "last_activity": session.last_activity,
                "wav_path": session.wav_path if hasattr(session, "wav_path") else None,
                "buffer_size": len(session.audio_buffer),
                "speech_buffer_duration": session.speech_buffer_duration,
            })
    return sessions


# =====================================================================
# Auto-start (appele par server.py)
# =====================================================================
def auto_start_listener(config):
    """Demarre automatiquement le listener si rtp_enabled=True."""
    if not config.get("rtp_enabled", False):
        logger.info("[RTP] rtp_enabled=False, listener non demarre")
        return False

    webhook_url = config.get("rtp_webhook_url", "")
    language = config.get("rtp_language", "fr")

    result = _start_listener(webhook_url=webhook_url, language=language)
    logger.info(f"[RTP] Auto-start: {result}")
    return True


# =====================================================================
# Endpoints Flask
# =====================================================================
@rtp_bp.route("/start", methods=["POST"])
def start_endpoint():
    """Demarre l'ecouteur RTP."""
    data = request.get_json(silent=True) or {}
    webhook_url = data.get("webhook_url")
    language = data.get("language", "fr")
    result = _start_listener(webhook_url=webhook_url, language=language)
    return jsonify(result)


@rtp_bp.route("/stop", methods=["POST"])
def stop_endpoint():
    """Arrete l'ecouteur RTP."""
    result = _stop_listener()
    return jsonify(result)


@rtp_bp.route("/sessions", methods=["GET"])
def sessions_endpoint():
    """Liste des sessions actives."""
    sessions = _get_active_sessions()
    return jsonify({
        "status": "success" if _rtp_listener else "not_running",
        "active_sessions": sessions,
        "count": len(sessions),
    })


@rtp_bp.route("/status", methods=["GET"])
def status_endpoint():
    """Statut de l'ecouteur RTP."""
    if not _rtp_listener or not _rtp_listener.is_running:
        return jsonify({"status": "not_running"})

    session_count = 0
    with _rtp_listener.sessions_lock:
        session_count = len(_rtp_listener.sessions)

    return jsonify({
        "status": "running",
        "port": RTP_PORT,
        "active_sessions": session_count,
        "uptime_seconds": time.time() - _rtp_listener.last_activity,
    })


@rtp_bp.route("/close_session/<client_key>", methods=["POST"])
def close_session_endpoint(client_key):
    """Ferme une session specifique."""
    if not _rtp_listener:
        return jsonify({
            "status": "error",
            "message": "L'ecouteur RTP n'est pas en cours d'execution",
        }), 400

    with _rtp_listener.sessions_lock:
        if client_key in _rtp_listener.sessions:
            session = _rtp_listener.sessions[client_key]

            # Finaliser le buffer
            if len(session.audio_buffer) > 0:
                _rtp_listener.process_buffer(session, force_final=True)

            session.cleanup()
            del _rtp_listener.sessions[client_key]

            return jsonify({
                "status": "success",
                "message": f"Session {client_key} fermee",
            })
        else:
            return jsonify({
                "status": "error",
                "message": f"Session {client_key} non trouvee",
            }), 404


@rtp_bp.route("/pool", methods=["GET"])
def pool_endpoint():
    """Stats du pool de modeles Whisper."""
    stats = get_pool_stats()
    return jsonify(stats)
