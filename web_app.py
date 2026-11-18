import math
import uuid
from functools import lru_cache
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename
import joblib
import numpy as np

from test_script import HRV16_ORDER, HRV16_UNITS, extract_ppg

PROJECT_ROOT = Path(__file__).resolve().parent
RHYTHM_MODEL_PATH = PROJECT_ROOT / "rhythm_rf.joblib"
ALLOWED_VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
RHYTHM_LABELS = {0: "Regular", 1: "Irregular rhythm"}

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["UPLOAD_FOLDER"] = PROJECT_ROOT / "uploads"


class RhythmModelError(Exception):
    """Raised when the saved rhythm model cannot be used safely."""


@lru_cache(maxsize=1)
def load_rhythm_model():
    if not RHYTHM_MODEL_PATH.is_file():
        raise RhythmModelError(f"Rhythm model not found: {RHYTHM_MODEL_PATH.name}")

    try:
        bundle = joblib.load(RHYTHM_MODEL_PATH)
    except (OSError, EOFError, ImportError, ValueError) as exc:
        raise RhythmModelError(f"Could not load {RHYTHM_MODEL_PATH.name}: {exc}") from exc

    if not isinstance(bundle, dict) or not {"model", "features", "threshold"} <= bundle.keys():
        raise RhythmModelError("The rhythm model bundle must contain model, features, and threshold.")

    model = bundle["model"]
    features = bundle["features"]
    threshold = bundle["threshold"]
    classes = getattr(model, "classes_", None)
    if not isinstance(features, list) or not features:
        raise RhythmModelError("The rhythm model has no valid feature list.")
    if not callable(getattr(model, "predict_proba", None)):
        raise RhythmModelError("The rhythm model must support probability predictions.")
    if classes is None or {str(value) for value in classes} != {"0", "1"}:
        raise RhythmModelError("The rhythm model must have numeric classes 0 and 1.")
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise RhythmModelError("The rhythm model threshold must be between 0 and 1.")

    return bundle


def classify_rhythm(metrics):
    bundle = load_rhythm_model()
    feature_names = bundle["features"]
    missing_features = [name for name in feature_names if name not in metrics]
    if missing_features:
        raise RhythmModelError(
            "HRV analysis did not provide model features: " + ", ".join(missing_features)
        )

    feature_values = np.asarray(
        [[float(metrics[name]) for name in feature_names]],
        dtype=np.float64,
    )
    try:
        probabilities = bundle["model"].predict_proba(feature_values)[0]
    except (TypeError, ValueError) as exc:
        raise RhythmModelError(f"Rhythm model prediction failed: {exc}") from exc
    classes = [int(value) for value in bundle["model"].classes_]
    if len(probabilities) != len(classes) or not np.all(np.isfinite(probabilities)):
        raise RhythmModelError("The rhythm model returned invalid class probabilities.")
    probability_by_class = dict(zip(classes, probabilities))
    probability_class_1 = float(probability_by_class[1])
    class_id = 1 if probability_class_1 >= float(bundle["threshold"]) else 0
    probability = float(probability_by_class[class_id])

    return {
        "class_id": class_id,
        "label": RHYTHM_LABELS[class_id],
        "probability": probability,
        "class_probabilities": [
            {"label": RHYTHM_LABELS[class_id], "probability": float(probability_by_class[class_id])}
            for class_id in (0, 1)
        ],
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload_file():
    uploaded = request.files.get("file")
    if uploaded is None:
        return jsonify({"error": "Choose a video to upload."}), 400
    if not uploaded.filename:
        return jsonify({"error": "Choose a video to upload."}), 400

    filename = secure_filename(uploaded.filename)
    if not filename:
        return jsonify({"error": "The selected filename is not valid."}), 400
    if Path(filename).suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        return jsonify({"error": "Choose a supported video file (MP4, MOV, AVI, MKV, M4V, or WebM)."}), 400

    upload_folder = Path(app.config["UPLOAD_FOLDER"])
    upload_folder.mkdir(parents=True, exist_ok=True)
    saved_path = upload_folder / f"{uuid.uuid4().hex}_{filename}"
    uploaded.save(saved_path)

    try:
        result = extract_ppg(str(saved_path))
    except (FileNotFoundError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 422

    try:
        classification = classify_rhythm(result.metrics)
    except RhythmModelError as exc:
        app.logger.error("Rhythm classification unavailable: %s", exc)
        return jsonify({"error": "Could not classify this recording: " + str(exc)}), 500

    sample_step = max(1, math.ceil(len(result.t) / 2200))
    chart_indices = slice(None, None, sample_step)
    metrics = [
        {
            "name": name,
            "label": name.replace("_", " "),
            "value": (
                float(result.metrics[name])
                if math.isfinite(float(result.metrics[name]))
                else None
            ),
            "unit": HRV16_UNITS[name],
        }
        for name in HRV16_ORDER
    ]
    mean_bpm = result.metrics.get("Mean_HR")
    spectral_bpm = result.metrics.get("HR_spectral_bpm")

    return jsonify({
        "filename": filename,
        "sampling_rate_hz": float(result.fs),
        "duration_seconds": float(result.t[-1]) if len(result.t) else 0.0,
        "mean_bpm": float(mean_bpm) if mean_bpm is not None and math.isfinite(float(mean_bpm)) else None,
        "spectral_bpm": float(spectral_bpm) if spectral_bpm is not None and math.isfinite(float(spectral_bpm)) else None,
        "classification": classification,
        "metrics": metrics,
        "warnings": result.warnings,
        "charts": {
            "time": result.t[chart_indices].tolist(),
            "raw": result.raw[chart_indices].tolist(),
            "filtered": result.filtered[chart_indices].tolist(),
            "peak_time": result.peak_times.tolist(),
            "peak_value": result.filtered[result.peaks].tolist(),
        },
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
