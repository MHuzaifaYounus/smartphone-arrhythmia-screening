from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from web_app import RhythmModelError, app, classify_rhythm


@pytest.mark.parametrize("filename", ["sample.mp4", "camera-recording.webm"])
def test_upload_runs_analysis_and_returns_results(tmp_path: Path, monkeypatch, filename):
    app.config["TESTING"] = True
    app.config["UPLOAD_FOLDER"] = tmp_path
    analyzed_paths = []

    def fake_extract_ppg(video_path):
        analyzed_paths.append(Path(video_path))
        return type("PPGResult", (), {
            "fs": 30.0,
            "t": np.array([0.0, 0.1, 0.2]),
            "raw": np.array([-100.0, -101.0, -99.0]),
            "filtered": np.array([0.0, 0.5, 0.1]),
            "peaks": np.array([1]),
            "peak_times": np.array([0.1]),
            "warnings": ["Sample quality note."],
            "metrics": {
                "Mean_HR": 72.0,
                "HR_spectral_bpm": 70.0,
                "Max_HR": 75.0,
                "Min_HR": 69.0,
                "SDNN": 10.0,
                "RMSSD": 8.0,
                "NN50": 0.0,
                "pNN50": 0.0,
                "VLF": 1.0,
                "LF": 2.0,
                "HF": 3.0,
                "TP": 6.0,
                "LFnu": 40.0,
                "HFnu": 60.0,
                "LF_HF": 0.67,
                "SD1": 5.0,
                "SD2": 12.0,
            },
        })()

    monkeypatch.setattr("web_app.extract_ppg", fake_extract_ppg)
    monkeypatch.setattr("web_app.classify_rhythm", lambda metrics: {
        "class_id": 0,
        "label": "Regular",
        "probability": 0.91,
        "class_probabilities": [
            {"label": "Regular", "probability": 0.91},
            {"label": "Irregular rhythm", "probability": 0.09},
        ],
    })
    response = app.test_client().post(
        "/api/upload",
        data={"file": (BytesIO(b"video contents"), filename)},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.json["filename"] == filename
    assert response.json["mean_bpm"] == 72.0
    assert response.json["spectral_bpm"] == 70.0
    assert response.json["classification"]["label"] == "Regular"
    assert response.json["classification"]["probability"] == 0.91
    assert response.json["metrics"][0]["label"] == "Mean HR"
    assert response.json["charts"]["filtered"] == [0.0, 0.5, 0.1]
    assert response.json["warnings"] == ["Sample quality note."]
    assert len(analyzed_paths) == 1
    assert analyzed_paths[0].is_file()
    assert analyzed_paths[0].read_bytes() == b"video contents"
    saved_files = list(tmp_path.iterdir())
    assert len(saved_files) == 1
    assert saved_files[0].read_bytes() == b"video contents"


def test_home_page_has_camera_recording_controls():
    response = app.test_client().get("/")

    assert response.status_code == 200
    assert b'id="cameraSelect"' in response.data
    assert b'id="startRecordingBtn"' in response.data
    assert b'id="stopRecordingBtn"' in response.data


def test_upload_requires_a_file():
    response = app.test_client().post("/api/upload", data={})

    assert response.status_code == 400
    assert response.json["error"] == "Choose a video to upload."


def test_upload_rejects_non_video_files(tmp_path: Path):
    app.config["TESTING"] = True
    app.config["UPLOAD_FOLDER"] = tmp_path

    response = app.test_client().post(
        "/api/upload",
        data={"file": (BytesIO(b"text"), "notes.txt")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "supported video file" in response.json["error"]
    assert list(tmp_path.iterdir()) == []


def test_classification_uses_model_feature_order_and_saved_threshold(monkeypatch):
    class FakeModel:
        classes_ = np.array([0, 1])

        def predict_proba(self, features):
            assert features.shape == (1, 3)
            assert features[0, 0] == 72.0
            assert features[0, 1] == 12.0
            assert features[0, 2] == 18.0
            return np.array([[0.4, 0.6]])

    monkeypatch.setattr("web_app.load_rhythm_model", lambda: {
        "model": FakeModel(),
        "features": ["Mean_HR", "SDNN", "irregular_pct"],
        "threshold": 0.55,
    })

    result = classify_rhythm({"Mean_HR": 72, "SDNN": 12, "irregular_pct": 18})

    assert result["class_id"] == 1
    assert result["label"] == "Irregular rhythm"
    assert result["probability"] == 0.6
    assert result["class_probabilities"] == [
        {"label": "Regular", "probability": 0.4},
        {"label": "Irregular rhythm", "probability": 0.6},
    ]


def test_class_zero_is_labeled_regular(monkeypatch):
    class FakeModel:
        classes_ = np.array([0, 1])

        def predict_proba(self, features):
            return np.array([[0.8, 0.2]])

    monkeypatch.setattr("web_app.load_rhythm_model", lambda: {
        "model": FakeModel(),
        "features": ["Mean_HR"],
        "threshold": 0.5,
    })

    result = classify_rhythm({"Mean_HR": 72})

    assert result["class_id"] == 0
    assert result["label"] == "Regular"


def test_classification_fails_if_model_feature_is_missing(monkeypatch):
    monkeypatch.setattr("web_app.load_rhythm_model", lambda: {
        "model": object(),
        "features": ["Mean_HR", "irregular_pct"],
        "threshold": 0.5,
    })

    with pytest.raises(RhythmModelError, match="irregular_pct"):
        classify_rhythm({"Mean_HR": 72})
