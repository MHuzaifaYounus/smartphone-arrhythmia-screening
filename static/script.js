const fileInput = document.getElementById('fileInput');
const uploadFileBtn = document.getElementById('uploadFileBtn');
const uploadStatus = document.getElementById('uploadStatus');
const cameraSelect = document.getElementById('cameraSelect');
const startCameraBtn = document.getElementById('startCameraBtn');
const stopCameraBtn = document.getElementById('stopCameraBtn');
const cameraPreview = document.getElementById('cameraPreview');
const cameraPlaceholder = document.getElementById('cameraPlaceholder');
const cameraStatus = document.getElementById('cameraStatus');
const startRecordingBtn = document.getElementById('startRecordingBtn');
const stopRecordingBtn = document.getElementById('stopRecordingBtn');
const recordingTimer = document.getElementById('recordingTimer');
const recordingIndicator = document.getElementById('recordingIndicator');
const cameraPanel = document.querySelector('.camera-panel');
const resultsPanel = document.getElementById('resultsPanel');
const metricsBody = document.getElementById('metricsBody');
const warningsPanel = document.getElementById('warningsPanel');
let latestCharts = null;
let cameraStream = null;
let cameraRecorder = null;
let recordingChunks = [];
let recordingStartedAt = 0;
let recordingInterval = null;
let isRecording = false;
let isAnalyzing = false;
let isLeavingPage = false;

function displayNumber(value, decimals = 1) {
  return Number.isFinite(value) ? value.toFixed(decimals) : '—';
}

function drawSignal(canvas, times, values, color, peaks = null) {
  const bounds = canvas.getBoundingClientRect();
  if (!bounds.width || !times.length || !values.length) return;

  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.round(bounds.width * scale);
  canvas.height = Math.round(250 * scale);
  const context = canvas.getContext('2d');
  context.setTransform(scale, 0, 0, scale, 0, 0);

  const width = bounds.width;
  const height = 250;
  const pad = { left: 52, right: 16, top: 16, bottom: 34 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const finiteValues = values.filter(Number.isFinite);
  if (!finiteValues.length) return;
  let low = Math.min(...finiteValues);
  let high = Math.max(...finiteValues);
  if (high === low) {
    low -= 1;
    high += 1;
  }
  const margin = (high - low) * 0.08;
  low -= margin;
  high += margin;
  const start = times[0];
  const end = times[times.length - 1] || start + 1;
  const x = time => pad.left + ((time - start) / (end - start)) * plotWidth;
  const y = value => pad.top + ((high - value) / (high - low)) * plotHeight;

  context.clearRect(0, 0, width, height);
  context.font = '11px Inter, sans-serif';
  context.textAlign = 'right';
  context.textBaseline = 'middle';
  context.strokeStyle = 'rgba(148, 163, 184, 0.16)';
  context.fillStyle = '#b6c5d9';
  for (let tick = 0; tick <= 4; tick += 1) {
    const value = low + ((high - low) * tick) / 4;
    const py = y(value);
    context.beginPath();
    context.moveTo(pad.left, py);
    context.lineTo(width - pad.right, py);
    context.stroke();
    context.fillText(value.toFixed(2), pad.left - 8, py);
  }

  context.beginPath();
  values.forEach((value, index) => {
    if (!Number.isFinite(value)) return;
    const px = x(times[index]);
    const py = y(value);
    if (index === 0) context.moveTo(px, py);
    else context.lineTo(px, py);
  });
  context.strokeStyle = color;
  context.lineWidth = 1.6;
  context.stroke();

  if (peaks) {
    context.fillStyle = '#3dd7c3';
    peaks.time.forEach((time, index) => {
      if (time < start || time > end) return;
      context.beginPath();
      context.arc(x(time), y(peaks.value[index]), 3, 0, Math.PI * 2);
      context.fill();
    });
  }

  context.textAlign = 'center';
  context.textBaseline = 'top';
  context.fillStyle = '#b6c5d9';
  context.fillText(`${start.toFixed(1)} s`, pad.left, height - pad.bottom + 9);
  context.fillText(`${end.toFixed(1)} s`, width - pad.right, height - pad.bottom + 9);
}

function renderCharts(charts) {
  latestCharts = charts;
  drawSignal(document.getElementById('rawChart'), charts.time, charts.raw, '#5ab6ff');
  drawSignal(
    document.getElementById('filteredChart'),
    charts.time,
    charts.filtered,
    '#ff6b81',
    { time: charts.peak_time, value: charts.peak_value },
  );
}

function renderResults(payload) {
  document.getElementById('resultTitle').textContent = payload.filename;
  document.getElementById('recordingDetails').textContent =
    `${displayNumber(payload.duration_seconds)} seconds · ${displayNumber(payload.sampling_rate_hz, 0)} samples/second`;
  document.getElementById('meanBpm').textContent = displayNumber(payload.mean_bpm, 0);
  document.getElementById('spectralBpm').textContent = displayNumber(payload.spectral_bpm, 0);

  const classificationCard = document.getElementById('classificationCard');
  const classification = payload.classification;
  classificationCard.classList.toggle('class-zero', classification.class_id === 0);
  classificationCard.classList.toggle('class-one', classification.class_id === 1);
  document.getElementById('rhythmClass').textContent = classification.label;
  const probabilityPercent = Math.round(classification.probability * 1000) / 10;
  document.getElementById('rhythmProbability').textContent = `${probabilityPercent.toFixed(1)}%`;
  const probabilityBar = document.getElementById('rhythmProbabilityBar');
  probabilityBar.setAttribute('aria-valuenow', String(probabilityPercent));
  document.getElementById('rhythmProbabilityFill').style.width = `${probabilityPercent}%`;
  document.getElementById('classProbabilities').textContent =
    classification.class_probabilities
      .map(item => `${item.label}: ${(item.probability * 100).toFixed(1)}%`)
      .join(' · ');

  metricsBody.replaceChildren();
  payload.metrics.forEach(metric => {
    const row = document.createElement('tr');
    const name = document.createElement('th');
    const value = document.createElement('td');
    const unit = document.createElement('td');
    name.scope = 'row';
    name.textContent = metric.label;
    value.textContent = metric.value === null ? '—' : displayNumber(metric.value, 2);
    unit.textContent = metric.unit;
    row.append(name, value, unit);
    metricsBody.append(row);
  });

  const warningsList = document.getElementById('warningsList');
  warningsList.replaceChildren();
  payload.warnings.forEach(warning => {
    const item = document.createElement('li');
    item.textContent = warning;
    warningsList.append(item);
  });
  warningsPanel.hidden = payload.warnings.length === 0;
  resultsPanel.hidden = false;
  renderCharts(payload.charts);
}

function updateControls() {
  uploadFileBtn.disabled = isAnalyzing;
  fileInput.disabled = isAnalyzing;
  startCameraBtn.disabled = isAnalyzing || isRecording;
  cameraSelect.disabled = isAnalyzing || isRecording || !cameraStream;
  stopCameraBtn.disabled = isAnalyzing || isRecording || !cameraStream;
  startRecordingBtn.disabled = isAnalyzing || isRecording || !cameraStream;
  stopRecordingBtn.disabled = isAnalyzing || !isRecording;
}

async function analyzeVideo(file, statusElement) {
  const formData = new FormData();
  formData.append('file', file);
  isAnalyzing = true;
  updateControls();
  statusElement.textContent = 'Uploading and analyzing video...';
  resultsPanel.hidden = true;

  try {
    const response = await fetch('/api/upload', { method: 'POST', body: formData });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || 'The file could not be uploaded.');
    }

    renderResults(payload);
    statusElement.textContent = 'Analysis complete.';
  } catch (error) {
    statusElement.textContent = error.message || 'The video could not be analyzed.';
  } finally {
    isAnalyzing = false;
    updateControls();
  }
}

async function uploadFile() {
  const file = fileInput.files[0];
  if (!file) {
    uploadStatus.textContent = 'Choose a video before analyzing.';
    return;
  }
  await analyzeVideo(file, uploadStatus);
}

async function refreshCameraDevices(activeDeviceId) {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cameras = devices.filter(device => device.kind === 'videoinput');
  cameraSelect.replaceChildren();

  cameras.forEach((camera, index) => {
    const option = document.createElement('option');
    option.value = camera.deviceId;
    option.textContent = camera.label || `Camera ${index + 1}`;
    cameraSelect.append(option);
  });

  if (activeDeviceId && cameras.some(camera => camera.deviceId === activeDeviceId)) {
    cameraSelect.value = activeDeviceId;
  }
  cameraSelect.disabled = !cameraStream || isRecording || isAnalyzing;
}

function releaseCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(track => track.stop());
    cameraStream = null;
  }
  cameraPreview.srcObject = null;
  cameraPlaceholder.hidden = false;
}

async function startCamera(deviceId = '') {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    cameraStatus.textContent = 'Camera access requires a supported browser and a secure page (HTTPS or localhost).';
    return;
  }

  startCameraBtn.disabled = true;
  cameraSelect.disabled = true;
  cameraStatus.textContent = 'Requesting camera access...';
  releaseCamera();

  try {
    const videoConstraints = {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      frameRate: { ideal: 30 },
    };
    if (deviceId) videoConstraints.deviceId = { exact: deviceId };

    cameraStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: videoConstraints,
    });
    cameraPreview.srcObject = cameraStream;
    await cameraPreview.play();
    cameraPlaceholder.hidden = true;

    const activeDeviceId = cameraStream.getVideoTracks()[0].getSettings().deviceId;
    await refreshCameraDevices(activeDeviceId);
    cameraStatus.textContent = 'Camera ready. Cover the camera lens with your fingertip, then start recording.';
    updateControls();
  } catch (error) {
    releaseCamera();
    cameraStatus.textContent = error.name === 'NotAllowedError'
      ? 'Camera permission was denied. Allow camera access in your browser and try again.'
      : error.name === 'NotFoundError'
        ? 'No camera was found. Connect your Android phone as a Windows webcam and try again.'
        : error.message || 'Could not open the selected camera.';
    updateControls();
  } finally {
    startCameraBtn.disabled = isAnalyzing || isRecording;
  }
}

function stopCamera() {
  if (isRecording) {
    cameraStatus.textContent = 'Stop the recording before turning off the camera.';
    return;
  }
  releaseCamera();
  cameraSelect.replaceChildren(new Option('Connect a camera to see devices', ''));
  cameraStatus.textContent = 'Camera is off.';
  updateControls();
}

function chooseRecordingMimeType() {
  const candidates = [
    'video/webm;codecs=vp9',
    'video/webm;codecs=vp8',
    'video/webm',
    'video/mp4',
  ];
  return candidates.find(type => MediaRecorder.isTypeSupported(type));
}

function updateRecordingTimer() {
  const elapsedSeconds = Math.floor((Date.now() - recordingStartedAt) / 1000);
  const minutes = String(Math.floor(elapsedSeconds / 60)).padStart(2, '0');
  const seconds = String(elapsedSeconds % 60).padStart(2, '0');
  recordingTimer.textContent = `${minutes}:${seconds}`;
}

function finishRecording(recorder) {
  clearInterval(recordingInterval);
  recordingInterval = null;
  isRecording = false;
  cameraPanel.classList.remove('is-recording');
  recordingIndicator.hidden = true;
  updateControls();

  const mimeType = recorder.mimeType || 'video/webm';
  const recording = new Blob(recordingChunks, { type: mimeType });
  recordingChunks = [];
  if (!recording.size) {
    cameraStatus.textContent = 'The camera did not produce a video. Check the camera and try again.';
    return;
  }

  const extension = mimeType.includes('mp4') ? 'mp4' : 'webm';
  const file = new File(
    [recording],
    `fingertip-recording-${Date.now()}.${extension}`,
    { type: mimeType },
  );
  cameraStatus.textContent = 'Recording finished. Sending it for analysis...';
  analyzeVideo(file, cameraStatus);
}

function startRecording() {
  if (!cameraStream) {
    cameraStatus.textContent = 'Enable a camera before recording.';
    return;
  }
  if (typeof MediaRecorder === 'undefined') {
    cameraStatus.textContent = 'Video recording is not supported by this browser.';
    return;
  }

  const mimeType = chooseRecordingMimeType();
  if (!mimeType) {
    cameraStatus.textContent = 'This browser does not support a video format for recording.';
    return;
  }

  try {
    cameraRecorder = new MediaRecorder(cameraStream, { mimeType });
    recordingChunks = [];
    cameraRecorder.addEventListener('dataavailable', event => {
      if (event.data.size) recordingChunks.push(event.data);
    });
    cameraRecorder.addEventListener('error', event => {
      clearInterval(recordingInterval);
      recordingInterval = null;
      isRecording = false;
      cameraPanel.classList.remove('is-recording');
      recordingIndicator.hidden = true;
      cameraStatus.textContent = event.error.message || 'Recording failed.';
      updateControls();
    }, { once: true });
    cameraRecorder.addEventListener('stop', () => {
      if (!isLeavingPage) finishRecording(cameraRecorder);
    }, { once: true });
    cameraRecorder.start(1000);
    isRecording = true;
    recordingStartedAt = Date.now();
    recordingTimer.textContent = '00:00';
    recordingIndicator.hidden = false;
    cameraPanel.classList.add('is-recording');
    cameraStatus.textContent = 'Recording. Keep your fingertip steady over the camera and flash.';
    recordingInterval = setInterval(updateRecordingTimer, 250);
    updateControls();
  } catch (error) {
    cameraStatus.textContent = error.message || 'Could not start recording.';
  }
}

function stopRecording() {
  if (cameraRecorder && cameraRecorder.state === 'recording') {
    cameraStatus.textContent = 'Finishing recording...';
    cameraRecorder.stop();
  }
}

uploadFileBtn.addEventListener('click', uploadFile);
startCameraBtn.addEventListener('click', () => startCamera(cameraSelect.value));
stopCameraBtn.addEventListener('click', stopCamera);
cameraSelect.addEventListener('change', () => startCamera(cameraSelect.value));
startRecordingBtn.addEventListener('click', startRecording);
stopRecordingBtn.addEventListener('click', stopRecording);
window.addEventListener('pagehide', () => {
  isLeavingPage = true;
  clearInterval(recordingInterval);
  if (cameraRecorder && cameraRecorder.state !== 'inactive') {
    cameraRecorder.stop();
  }
  releaseCamera();
});
window.addEventListener('resize', () => {
  if (latestCharts && !resultsPanel.hidden) renderCharts(latestCharts);
});
