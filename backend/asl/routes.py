"""
ASL Flask Routes (Blueprint)
============================
Exposes dedicated ASL recognition endpoints:
  - GET  /asl/status
  - POST /asl/start
  - POST /asl/frame
  - POST /asl/commit
  - POST /asl/undo
  - POST /asl/clear
  - POST /asl/stop
"""

import logging
from flask import Blueprint, request, jsonify

from .asl_service import ASLService

logger = logging.getLogger(__name__)
asl_bp = Blueprint("asl", __name__, url_prefix="/asl")


def _get_service() -> ASLService:
    return ASLService.get_instance()


@asl_bp.route("/status", methods=["GET"])
def asl_status():
    """
    Returns model readiness, MediaPipe status, active compute device,
    and active session counts.
    """
    try:
        service = _get_service()
        status_info = service.get_status()
        return jsonify({"success": True, **status_info}), 200
    except Exception as e:
        logger.error("Failed to query ASL status: %s", e)
        return jsonify({
            "success": False,
            "ready": False,
            "error": "Failed to inspect ASL subsystem status"
        }), 500


@asl_bp.route("/start", methods=["POST"])
def asl_start():
    """
    Initializes or registers a new recognition session.
    Payload (optional): { "session_id": "...", "mode": "alphabet"|"word" }
    """
    try:
        data = request.get_json(silent=True) or {}
        session_id = data.get("session_id", "").strip() or None
        mode = data.get("mode", "alphabet").strip().lower()
        if mode not in ("alphabet", "word"):
            mode = "alphabet"

        service = _get_service()
        session = service.session_manager.create_session(session_id=session_id, mode=mode)

        return jsonify({
            "success": True,
            "session_id": session.session_id,
            "mode": session.mode,
            "sentence": session.sentence_builder.get_sentence(),
            "message": "ASL session initialized successfully"
        }), 200
    except Exception as e:
        logger.error("Failed to start ASL session: %s", e)
        return jsonify({
            "success": False,
            "error": "Failed to start recognition session"
        }), 500


@asl_bp.route("/frame", methods=["POST"])
def asl_frame():
    """
    Receives an in-flight webcam frame and returns live prediction and sentence updates.
    Payload: { "session_id": "...", "image": "data:image/jpeg;base64,...", "mode": "alphabet" }
    """
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"success": False, "error": "Invalid JSON body"}), 400

        session_id = data.get("session_id", "").strip()
        if not session_id:
            return jsonify({"success": False, "error": "Missing session_id"}), 400

        image_b64 = data.get("image", "")
        if not image_b64:
            return jsonify({"success": False, "error": "Missing image frame data"}), 400

        mode = data.get("mode", "alphabet").strip().lower()
        if mode not in ("alphabet", "word"):
            mode = "alphabet"

        service = _get_service()
        frame_rgb = service.decode_frame(image_b64)
        if frame_rgb is None:
            return jsonify({"success": False, "error": "Invalid or unparseable image data"}), 400

        result = service.process_frame(session_id=session_id, frame_rgb=frame_rgb, mode=mode)
        return jsonify(result), 200
    except Exception as e:
        logger.error("Error processing ASL frame: %s", e)
        return jsonify({
            "success": False,
            "error": "Internal error processing frame"
        }), 500


@asl_bp.route("/commit", methods=["POST"])
def asl_commit():
    """
    Explicitly commits the current letters or word override into the sentence.
    Payload: { "session_id": "...", "word": "override" (optional) }
    """
    try:
        data = request.get_json(silent=True) or {}
        session_id = data.get("session_id", "").strip()
        word_override = data.get("word", "").strip() or None

        service = _get_service()
        session = service.session_manager.get_session(session_id)
        if not session:
            return jsonify({"success": False, "error": "Session not found"}), 404

        session.sentence_builder.commit(word_override=word_override)
        return jsonify({
            "success": True,
            "sentence": session.sentence_builder.get_sentence(),
            "current_word": session.sentence_builder.get_current_word()
        }), 200
    except Exception as e:
        logger.error("Error committing ASL token: %s", e)
        return jsonify({"success": False, "error": "Commit failed"}), 500


@asl_bp.route("/undo", methods=["POST"])
def asl_undo():
    """
    Removes the last accepted token (letter or word) from the current sentence.
    Payload: { "session_id": "..." }
    """
    try:
        data = request.get_json(silent=True) or {}
        session_id = data.get("session_id", "").strip()

        service = _get_service()
        session = service.session_manager.get_session(session_id)
        if not session:
            return jsonify({"success": False, "error": "Session not found"}), 404

        session.sentence_builder.undo()
        return jsonify({
            "success": True,
            "sentence": session.sentence_builder.get_sentence(),
            "current_word": session.sentence_builder.get_current_word()
        }), 200
    except Exception as e:
        logger.error("Error undoing ASL token: %s", e)
        return jsonify({"success": False, "error": "Undo failed"}), 500


@asl_bp.route("/clear", methods=["POST"])
def asl_clear():
    """
    Clears all buffers and recognized text in the session.
    Payload: { "session_id": "..." }
    """
    try:
        data = request.get_json(silent=True) or {}
        session_id = data.get("session_id", "").strip()

        service = _get_service()
        session = service.session_manager.get_session(session_id)
        if not session:
            return jsonify({"success": False, "error": "Session not found"}), 404

        session.sentence_builder.clear()
        session.seq_buffer.clear()
        return jsonify({
            "success": True,
            "sentence": "",
            "current_word": ""
        }), 200
    except Exception as e:
        logger.error("Error clearing ASL session: %s", e)
        return jsonify({"success": False, "error": "Clear failed"}), 500


@asl_bp.route("/stop", methods=["POST"])
def asl_stop():
    """
    Releases the recognition session.
    Payload: { "session_id": "..." }
    """
    try:
        data = request.get_json(silent=True) or {}
        session_id = data.get("session_id", "").strip()

        service = _get_service()
        if session_id:
            service.session_manager.remove_session(session_id)

        return jsonify({
            "success": True,
            "message": "ASL session stopped and released successfully"
        }), 200
    except Exception as e:
        logger.error("Error stopping ASL session: %s", e)
        return jsonify({"success": False, "error": "Stop failed"}), 500

