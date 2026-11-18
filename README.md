# PulseSense PPG Analysis

PulseSense extracts a fingertip PPG waveform from an uploaded video, estimates
heart rate in BPM, and displays the script's 16 HRV measures with signal-quality
notes. The bundled `rhythm_rf.joblib` model classifies the HRV feature vector as
Regular (class 0) or Irregular rhythm (class 1), and displays the model's
probability for each class. The original video is saved in the `uploads/`
folder.

Use a steady recording with the fingertip covering the camera and flash. The
results are informational and are not a medical diagnosis.

## Run locally

```powershell
python -m pip install -r requirements.txt
python web_app.py
```

Open http://127.0.0.1:5000 in your browser and upload an MP4, MOV, AVI, MKV,
M4V, or WebM video.

To record from a webcam, connect the Android phone to Windows using a supported
phone-as-webcam option, allow camera access in the browser, choose the phone
camera, and select **Start recording**. Select **Stop & analyze** to upload the
recorded clip to the same analysis pipeline. The camera feature requires a
secure browser context; `http://127.0.0.1:5000` counts as secure for local use.

## Test

```powershell
pytest -q
```
