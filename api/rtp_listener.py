"""Ecouteur RTP multi-clients avec transcription Whisper.

Port direct de stt-api/routes/rtp_listener.py, adapte pour faster-whisper
avec architecture borrow-style (le modele est emprunte au pool uniquement
pendant la transcription ~0.3s).

Pipeline audio identique a stt-api :
  RTP UDP -> strip header -> amplify_audio_early -> WAV recording
  -> accumulate buffer -> detect silence/speech -> transcribe_speech
  -> apply_corrections + text2num -> send_webhook
"""

import json
import os
import re
import socket
import threading
import time

import numpy as np
import requests

from api.rtp_config import (
    BUFFER_THRESHOLD,
    CHANNELS,
    RECORD_WAV,
    RTP_PORT,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    SAVE_DIR,
    SILENCE_THRESHOLD_MEAN,
    WEBHOOK_URL,
    logger,
)
from api.rtp_session import RTPSession
from api.whisper_pool import whisper_pool, get_pool_stats

# --- text2num (optionnel) ---
try:
    from text_to_num import alpha2digit
    TEXT2NUM_AVAILABLE = True
    logger.info("Module text2num (alpha2digit) charge avec succes")
except ImportError:
    TEXT2NUM_AVAILABLE = False
    logger.warning(
        "Module text2num non installe. "
        "Conversion des nombres ecrits en chiffres desactivee."
    )

    def alpha2digit(text, lang="fr"):  # type: ignore[misc]
        return text


# --- Corrections post-transcription ---
CORRECTIONS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "corrections.txt",
)


def _load_corrections():
    """Charge les corrections depuis corrections.txt."""
    if not os.path.exists(CORRECTIONS_FILE):
        return []
    corrections = []
    with open(CORRECTIONS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if " -> " not in line:
                continue
            parts = line.split(" -> ", 1)
            erreur = parts[0].strip()
            correction = parts[1].strip()
            if erreur and correction:
                pattern = re.compile(re.escape(erreur), re.IGNORECASE)
                corrections.append((pattern, correction))
    return corrections


def apply_corrections(text: str) -> str:
    """Applique les corrections post-transcription."""
    corrections = _load_corrections()
    if not corrections:
        return text
    for pattern, replacement in corrections:
        text = pattern.sub(replacement, text)
    return text


# --- Vocabulaire (prompt Whisper) ---
VOCAB_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "vocabulaire.txt",
)


def _load_vocabulary() -> str:
    """Charge le vocabulaire et le transforme en prompt initial pour Whisper."""
    if not os.path.exists(VOCAB_FILE):
        return ""
    words = []
    with open(VOCAB_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                words.append(line)
    return ", ".join(words)


# =====================================================================
# Amplification audio precoce (IDENTIQUE a stt-api)
# =====================================================================
def amplify_audio_early(audio_array, client_key="unknown"):
    """Amplifier l'audio faible AVANT la detection de signal.

    Retourne (audio_amplifie, moyenne, a_ete_amplifie).
    """
    orig_mean = np.mean(np.abs(audio_array))

    if orig_mean > 500:
        return audio_array, orig_mean, False

    target_mean = 1000
    gain = target_mean / max(orig_mean, 1)
    gain = min(gain, 50.0)

    logger.debug(
        f"[AMPLIFY] Client {client_key}: Audio faible "
        f"(moyenne={orig_mean:.2f}), gain={gain:.2f}"
    )
    amplified = audio_array * gain
    max_val = np.iinfo(np.int16).max
    amplified = np.clip(amplified, -max_val, max_val)
    amplified_array = amplified.astype(np.int16)

    amp_mean = np.mean(np.abs(amplified_array))
    return amplified_array, amp_mean, True


# =====================================================================
# RTPListener
# =====================================================================
class RTPListener:
    """Gestionnaire d'ecoute RTP multi-clients avec transcription Whisper."""

    def __init__(self, port=None, save_dir=None):
        self.port = port or RTP_PORT
        self.save_dir = save_dir or SAVE_DIR
        self.sock = None
        self.is_running = False
        self.webhook_url = WEBHOOK_URL
        self.language = "fr"

        # Sessions par addr:port
        self.sessions = {}
        self.sessions_lock = threading.Lock()

        # Compteurs / timers
        self.last_activity = time.time()
        self.weak_signal_start_times = {}
        self.buffer_process_counts = {}
        self.force_partial_check_interval = 5
        self.max_weak_signal_duration = 3.0

        # Fichier / duree max
        self.max_file_duration = 60 * 5  # 5 min

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------
    def get_or_create_session(self, client_addr, client_port,
                              client_id=None, language="fr"):
        """Recupere ou cree une session pour ce client."""
        client_key = f"{client_addr}:{client_port}"

        with self.sessions_lock:
            if client_key in self.sessions:
                session = self.sessions[client_key]
                session.last_activity = time.time()
                # Assurer la compat pour les attributs ajoutes dynamiquement
                if not hasattr(session, "consecutive_timeouts"):
                    session.consecutive_timeouts = 0
                if not hasattr(session, "has_audio_data"):
                    session.has_audio_data = False
                if not hasattr(session, "has_pending_partial"):
                    session.has_pending_partial = False
                return session

            # Creer une nouvelle session (pas de modele permanent)
            session = RTPSession(
                client_addr=client_addr,
                client_port=client_port,
                save_dir=self.save_dir,
                webhook_url=self.webhook_url,
                client_id=client_id,
                language=language,
            )
            self.sessions[client_key] = session
            logger.info(f"Nouvelle session creee pour {client_key}")
            return session

    # ------------------------------------------------------------------
    # Communication (webhook + RTP response)
    # ------------------------------------------------------------------
    def send_webhook(self, session, data):
        """Envoie les donnees par RTP direct puis par webhook HTTP."""
        sent_rtp = self.send_rtp_response(session, data)

        if not sent_rtp and session.webhook_url:
            try:
                data["client_id"] = session.client_id
                data["session_id"] = session.session_id
                response = requests.post(
                    session.webhook_url, json=data, timeout=2
                )
                if response.status_code != 200:
                    logger.warning(
                        f"Echec webhook pour {session.client_key}: "
                        f"{response.status_code}"
                    )
            except Exception as e:
                logger.error(
                    f"Erreur webhook pour {session.client_key}: {e}"
                )

    def send_rtp_response(self, session, data):
        """Envoie une reponse JSON via paquet RTP (PT=123)."""
        try:
            if not session.client_addr or not session.client_port:
                return False

            response_data = json.dumps(data).encode("utf-8")

            rtp_header = bytearray(12)
            rtp_header[0] = 0x80  # version 2
            rtp_header[1] = 123   # payload type JSON

            seq = int(time.time() * 1000) & 0xFFFF
            rtp_header[2] = (seq >> 8) & 0xFF
            rtp_header[3] = seq & 0xFF

            ts = int(time.time() * 1000)
            rtp_header[4] = (ts >> 24) & 0xFF
            rtp_header[5] = (ts >> 16) & 0xFF
            rtp_header[6] = (ts >> 8) & 0xFF
            rtp_header[7] = ts & 0xFF

            ssrc = 0x12345678
            rtp_header[8] = (ssrc >> 24) & 0xFF
            rtp_header[9] = (ssrc >> 16) & 0xFF
            rtp_header[10] = (ssrc >> 8) & 0xFF
            rtp_header[11] = ssrc & 0xFF

            packet = bytes(rtp_header) + response_data
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.sendto(packet, (session.client_addr, session.client_port))
            logger.debug(
                f"Reponse RTP envoyee a {session.client_key} ({len(packet)} o)"
            )
            return True
        except Exception as e:
            logger.error(f"Erreur envoi RTP a {session.client_key}: {e}")
            return False

    def send_audio_status(self, session, has_speech):
        """Envoie un status audio au client Node-RED."""
        data = {
            "type": "audio_status",
            "has_speech": has_speech,
            "timestamp": time.time(),
        }
        return self.send_rtp_response(session, data)

    # ------------------------------------------------------------------
    # text2num
    # ------------------------------------------------------------------
    def convert_text_to_numbers(self, text, language="fr"):
        """Convertit les nombres ecrits en chiffres via alpha2digit."""
        if not text or not TEXT2NUM_AVAILABLE:
            return text
        try:
            base_lang = language.split("-")[0].lower() if "-" in language else language.lower()
            supported = ["fr", "en", "es", "pt", "de", "nl", "it"]
            if base_lang not in supported:
                base_lang = "fr"
            converted = alpha2digit(text, base_lang)
            if converted != text:
                logger.debug(f"[ALPHA2DIGIT] '{text}' -> '{converted}'")
            return converted
        except Exception as e:
            logger.debug(f"[ALPHA2DIGIT] Erreur (texte conserve): {e}")
            return text

    # ------------------------------------------------------------------
    # Transcription Whisper (borrow-style)
    # ------------------------------------------------------------------
    def transcribe_speech(self, session):
        """Transcrit le speech_buffer accumule en empruntant un modele du pool.

        Retourne le texte transcrit ou None.
        """
        if not session.speech_buffer:
            return None

        # Concatener le buffer de parole
        speech_audio = np.concatenate(session.speech_buffer)
        duration = len(speech_audio) / SAMPLE_RATE

        if duration < 0.3:
            logger.debug(
                f"[TRANSCRIBE] {session.client_key}: Audio trop court "
                f"({duration:.2f}s), ignore"
            )
            session.clear_speech_buffer()
            return None

        logger.info(
            f"[TRANSCRIBE] {session.client_key}: Transcription de "
            f"{duration:.2f}s d'audio..."
        )

        # Emprunter un modele du pool
        caller_id = f"transcribe_{session.client_key}_{int(time.time()*1000)}"
        model_index, model = whisper_pool.acquire_model(caller_id, timeout=5.0)

        if model is None:
            logger.error(
                f"[TRANSCRIBE] {session.client_key}: Aucun modele disponible"
            )
            return None

        try:
            t0 = time.perf_counter()

            prompt = _load_vocabulary()
            segments, info = model.transcribe(
                speech_audio,
                language=session.language.split("-")[0].lower(),
                beam_size=5,
                vad_filter=False,
                initial_prompt=prompt if prompt else None,
            )

            text_parts = [s.text.strip() for s in segments]
            text = " ".join(text_parts).strip()

            elapsed = time.perf_counter() - t0
            logger.info(
                f"[TRANSCRIBE] {session.client_key} [Model #{model_index}]: "
                f"'{text}' ({elapsed:.2f}s, {duration:.1f}s audio)"
            )

            # Appliquer corrections + text2num
            if text:
                text = apply_corrections(text)
                text = self.convert_text_to_numbers(text, session.language)

            return text

        except Exception as e:
            logger.error(
                f"[TRANSCRIBE] {session.client_key}: Erreur : {e}"
            )
            import traceback
            logger.error(traceback.format_exc())
            return None

        finally:
            # TOUJOURS rendre le modele, meme en cas d'erreur
            whisper_pool.release_model(caller_id)
            session.clear_speech_buffer()

    # ------------------------------------------------------------------
    # Traitement du buffer audio
    # ------------------------------------------------------------------
    def process_buffer(self, session, force_final=False):
        """Traite le buffer audio accumule.

        Detecte silence / parole, accumule dans speech_buffer,
        et declenche la transcription quand necessaire.
        """
        if len(session.audio_buffer) == 0:
            return

        try:
            audio_data = bytes(session.audio_buffer)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_mean = np.mean(np.abs(audio_array))

            is_speech = audio_mean >= SILENCE_THRESHOLD_MEAN

            if is_speech:
                # Convertir int16 -> float32 normalise pour Whisper
                audio_f32 = audio_array.astype(np.float32) / 32768.0
                session.speech_buffer.append(audio_f32)
                session.speech_buffer_duration += len(audio_f32) / SAMPLE_RATE

                # Signaler la parole a Node-RED
                self.send_audio_status(session, has_speech=True)

                logger.debug(
                    f"[BUFFER] {session.client_key}: Parole detectee "
                    f"(mean={audio_mean:.1f}), buffer parole="
                    f"{session.speech_buffer_duration:.2f}s"
                )
            else:
                # Silence detecte
                self.send_audio_status(session, has_speech=False)

                # Si on avait de la parole avant le silence -> transcrire
                if session.speech_buffer and not force_final:
                    logger.info(
                        f"[BUFFER] {session.client_key}: Silence apres parole, "
                        f"transcription de {session.speech_buffer_duration:.2f}s"
                    )
                    text = self.transcribe_speech(session)
                    if text:
                        self._send_transcription(session, text, final=True)

            # Force final (fin de session / timeout)
            if force_final and session.speech_buffer:
                logger.info(
                    f"[BUFFER] {session.client_key}: Finalisation forcee, "
                    f"transcription de {session.speech_buffer_duration:.2f}s"
                )
                text = self.transcribe_speech(session)
                if text:
                    self._send_transcription(
                        session, text, final=True, forced=True
                    )
                elif session.partial_text:
                    # Utiliser le dernier partiel comme final
                    self._send_transcription(
                        session, session.partial_text, final=True, forced=True
                    )

            # Reinitialiser le buffer audio (int16)
            session.audio_buffer = bytearray()

        except Exception as e:
            logger.error(
                f"[BUFFER] {session.client_key}: Erreur : {e}"
            )
            import traceback
            logger.error(traceback.format_exc())
            session.audio_buffer = bytearray()

    def _send_transcription(self, session, text, final=False, forced=False):
        """Envoie une transcription via webhook."""
        webhook_data = {
            "type": "transcription",
            "text": text,
            "final": final,
            "timestamp": time.time(),
            "confidence": None,
            "forced_by_inactivity": forced,
        }

        self.send_webhook(session, webhook_data)

        if final:
            logger.info(
                f"[TRANSCRIPTION] {session.client_key} FINALE"
                f"{' (forcee)' if forced else ''}: '{text}'"
            )
            session.partial_text = ""
            session.has_pending_partial = False
            session.final_transcription_sent = True
            session.final_transcription_time = time.time()
            session.reset_state()
        else:
            session.partial_text = text

    # ------------------------------------------------------------------
    # Boucle principale
    # ------------------------------------------------------------------
    def start(self, webhook_url=None, client_id=None, language="fr"):
        """Demarre l'ecouteur RTP."""
        if webhook_url:
            self.webhook_url = webhook_url
        self.language = language
        self.is_running = True

        # Socket UDP avec buffer agrandi
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        SO_RCVBUF = 10 * 1024 * 1024  # 10 MB
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SO_RCVBUF)
        actual_buf = self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        logger.info(f"[SOCKET] Buffer UDP : demande={SO_RCVBUF}, effectif={actual_buf}")

        self.sock.bind(("0.0.0.0", self.port))
        logger.info(f"Ecouteur RTP multi-clients demarre sur port {self.port}")

        if TEXT2NUM_AVAILABLE:
            logger.info(f"Module text2num actif, langue={self.language}")

        # Stats du pool
        stats = get_pool_stats()
        logger.info(
            f"[POOL] {stats['total']} modele(s), "
            f"{stats['available']} disponible(s)"
        )

        self.listen_loop()

    def listen_loop(self):
        """Boucle principale d'ecoute RTP."""
        self.last_activity = time.time()
        force_transcription_timeout = 10.0
        quick_finalization_timeout = 0.3
        max_consecutive_timeouts = 4

        last_audio_times = {}

        try:
            while self.is_running:
                try:
                    self.sock.settimeout(quick_finalization_timeout)
                    data, addr = self.sock.recvfrom(4096)
                    current_time = time.time()

                    client_addr, client_port = addr
                    client_key = f"{client_addr}:{client_port}"

                    # --- Commande PT=124 ---
                    if len(data) >= 12:
                        payload_type = data[1] & 0x7F
                        if payload_type == 124:
                            self._handle_command(data, client_key)
                            continue

                    # --- Fin de session (marker bit) ---
                    is_end_of_session = False
                    if len(data) >= 12:
                        marker_bit = (data[1] & 0x80) != 0
                        payload_size = len(data) - 12
                        if marker_bit or payload_size == 0:
                            is_end_of_session = True
                            logger.info(
                                f"[RTP-END] Signal de fin pour {client_key} "
                                f"(marker={marker_bit}, payload={payload_size})"
                            )

                    # Obtenir ou creer la session
                    session = self.get_or_create_session(
                        client_addr, client_port, language=self.language
                    )
                    if session is None:
                        logger.warning(
                            f"[RTP] Paquet ignore pour {client_key} "
                            f"(impossible de creer la session)"
                        )
                        continue

                    # --- Fin de session immediate ---
                    if is_end_of_session:
                        logger.info(
                            f"[RTP-END] Traitement fin de session {client_key}"
                        )
                        with self.sessions_lock:
                            if len(session.audio_buffer) > 0:
                                self.process_buffer(session, force_final=True)
                            elif session.speech_buffer:
                                text = self.transcribe_speech(session)
                                if text:
                                    self._send_transcription(
                                        session, text, final=True, forced=True
                                    )
                            session.cleanup()
                            if client_key in self.sessions:
                                del self.sessions[client_key]
                            logger.info(
                                f"[RTP-END] Session {client_key} fermee"
                            )
                        continue

                    # Mise a jour activite
                    session.last_activity = current_time
                    last_audio_times[client_key] = current_time
                    session.consecutive_timeouts = 0
                    session.has_audio_data = True

                    # Nouveau fichier WAV si necessaire
                    file_duration = current_time - session.current_file_start
                    if ((current_time - session.last_activity > 10)
                            or file_duration >= self.max_file_duration):
                        if len(session.audio_buffer) > 0:
                            self.process_buffer(session, force_final=True)
                        if RECORD_WAV:
                            session.init_new_wav_file()
                        else:
                            session.current_file_start = current_time

                    # --- Extraction et traitement audio ---
                    if len(data) > 12:
                        audio_data = data[12:]
                        if len(audio_data) == 0:
                            continue

                        try:
                            audio_array = np.frombuffer(audio_data, dtype=np.int16)
                            if len(audio_array) == 0:
                                continue

                            # Amplification precoce (IDENTIQUE stt-api)
                            audio_array, audio_mean, was_amplified = (
                                amplify_audio_early(audio_array, client_key)
                            )

                            # Ecriture WAV
                            if RECORD_WAV and session.wav_file:
                                try:
                                    session.wav_file.writeframes(
                                        audio_array.tobytes()
                                    )
                                except Exception as e:
                                    logger.error(
                                        f"Erreur WAV {client_key}: {e}"
                                    )

                            # Accumuler dans le buffer
                            session.audio_buffer.extend(audio_array.tobytes())

                            # Traiter quand le buffer atteint le seuil
                            if len(session.audio_buffer) >= BUFFER_THRESHOLD:
                                self.process_buffer(session)

                        except Exception as e:
                            logger.error(
                                f"Erreur traitement audio {client_key}: {e}"
                            )

                except socket.timeout:
                    # --- Gestion des timeouts ---
                    current_time = time.time()

                    with self.sessions_lock:
                        for client_key, session in list(self.sessions.items()):
                            session.consecutive_timeouts += 1
                            current_buffer_size = len(session.audio_buffer)

                            # Marquer partiel en attente
                            if (session.consecutive_timeouts == 1
                                    and session.partial_text
                                    and not session.has_pending_partial):
                                session.has_pending_partial = True

                            # --- Finalisation par timeout ---
                            min_timeouts_tts = 2       # 600ms pour TTS
                            min_timeouts_voice = max_consecutive_timeouts  # 1.2s

                            timeout_threshold = (
                                min_timeouts_tts
                                if current_buffer_size > 0
                                else min_timeouts_voice
                            )

                            has_content = (
                                current_buffer_size > 0
                                or session.speech_buffer
                                or session.has_pending_partial
                            )

                            if (session.has_audio_data
                                    and session.consecutive_timeouts >= timeout_threshold
                                    and has_content):
                                logger.info(
                                    f"[FINALISATION] Arret paquets RTP "
                                    f"{client_key} apres "
                                    f"{session.consecutive_timeouts} timeouts"
                                )

                                if current_buffer_size > 0:
                                    self.process_buffer(
                                        session, force_final=True
                                    )
                                elif session.speech_buffer:
                                    text = self.transcribe_speech(session)
                                    if text:
                                        self._send_transcription(
                                            session, text,
                                            final=True, forced=True,
                                        )
                                elif (session.has_pending_partial
                                      and session.partial_text):
                                    final_text = self.convert_text_to_numbers(
                                        session.partial_text, session.language
                                    )
                                    self._send_transcription(
                                        session, final_text,
                                        final=True, forced=True,
                                    )

                                session.consecutive_timeouts = 0
                                session.has_audio_data = False
                                session.has_pending_partial = False
                                continue

                            # --- Inactivite prolongee (fallback) ---
                            if (current_buffer_size > 0
                                    and current_time - last_audio_times.get(
                                        client_key, 0
                                    ) > force_transcription_timeout):
                                logger.info(
                                    f"[FINALISATION-FALLBACK] Inactivite "
                                    f"prolongee {client_key}"
                                )
                                self.process_buffer(
                                    session, force_final=True
                                )
                                session.consecutive_timeouts = 0
                                session.has_audio_data = False
                                continue

                            # --- Auto-cloture apres transcription finale ---
                            if (session.final_transcription_sent
                                    and session.final_transcription_time
                                    and current_time
                                    - session.final_transcription_time > 3.0):
                                logger.info(
                                    f"[AUTO-CLOSE] Fermeture auto "
                                    f"{client_key}"
                                )
                                if len(session.audio_buffer) > 0:
                                    self.process_buffer(
                                        session, force_final=True
                                    )
                                session.cleanup()
                                del self.sessions[client_key]
                                continue

                            # --- Sessions inactives (>30s) ---
                            if current_time - session.last_activity > 30:
                                logger.info(
                                    f"[AUTO-CLOSE] Session inactive "
                                    f"{client_key} ({current_time - session.last_activity:.0f}s)"
                                )
                                if len(session.audio_buffer) > 0:
                                    self.process_buffer(
                                        session, force_final=True
                                    )
                                session.cleanup()
                                del self.sessions[client_key]

                    continue

                except Exception as e:
                    logger.error(f"Erreur boucle d'ecoute: {e}")
                    time.sleep(1)

        finally:
            # Finaliser toutes les sessions
            with self.sessions_lock:
                for client_key, session in list(self.sessions.items()):
                    if len(session.audio_buffer) > 0:
                        self.process_buffer(session, force_final=True)
                    elif session.speech_buffer:
                        text = self.transcribe_speech(session)
                        if text:
                            self._send_transcription(
                                session, text, final=True, forced=True
                            )
                    session.cleanup()
                    logger.info(f"Session nettoyee : {client_key}")

            if self.sock:
                self.sock.close()
            logger.info("Ecouteur RTP arrete")

    # ------------------------------------------------------------------
    # Commandes RTP (PT=124)
    # ------------------------------------------------------------------
    def _handle_command(self, data, client_key):
        """Traite un paquet de commande RTP (PT=124)."""
        try:
            json_payload = data[12:].decode("utf-8")
            command_data = json.loads(json_payload)

            if command_data.get("type") != "command":
                return

            cmd = command_data.get("command")
            cmd_client_id = command_data.get("client_id", "unknown")

            if cmd == "reset":
                logger.info(
                    f"[RTP-CMD] RESET recu pour {cmd_client_id} "
                    f"depuis {client_key}"
                )
                with self.sessions_lock:
                    session = self.sessions.get(client_key)
                    if session:
                        session.audio_buffer = bytearray()
                        session.reset_state()
                        if client_key in self.weak_signal_start_times:
                            del self.weak_signal_start_times[client_key]
                        if client_key in self.buffer_process_counts:
                            del self.buffer_process_counts[client_key]
                        logger.info(
                            f"[RTP-CMD] Session {client_key} reinitialisee"
                        )
                    else:
                        logger.warning(
                            f"[RTP-CMD] Pas de session pour {client_key}"
                        )
            else:
                logger.warning(f"[RTP-CMD] Commande inconnue: {cmd}")

        except json.JSONDecodeError as e:
            logger.error(f"[RTP-CMD] JSON invalide: {e}")
        except Exception as e:
            logger.error(f"[RTP-CMD] Erreur: {e}")

    # ------------------------------------------------------------------
    # Stop
    # ------------------------------------------------------------------
    def stop(self):
        """Arrete l'ecouteur RTP."""
        self.is_running = False
        logger.info("Arret demande pour l'ecouteur RTP")
