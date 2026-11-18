#!/usr/bin/env python3
"""
Fingertip-video PPG extraction (improved version of the Zhang et al. 2021
Section II-B pipeline: spatial averaging of frames -> 1-D PPG).

Pipeline:
  video -> ROI mean of R/G/B per frame -> true timestamps -> uniform resampling
        -> trim start-up transient -> band-pass -> adaptive beat detection
        -> sub-sample peak timing -> PPI series (+ irregular-interval flags)
        -> the 16 HRV parameters of Zhang et al. (2021) + spectral HR cross-check
        + quality flags
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import signal
from scipy.integrate import trapezoid
from scipy.ndimage import median_filter


# --------------------------------------------------------------------------- #
# Data container
# --------------------------------------------------------------------------- #
@dataclass
class PPGResult:
    t: np.ndarray                 # uniform time axis (s), after trimming
    raw: np.ndarray               # inverted mean-red trace (unfiltered), trimmed
    filtered: np.ndarray          # band-passed PPG
    fs: float                     # sampling rate (Hz)
    peaks: np.ndarray             # sample indices of detected beats in `filtered`
    peak_times: np.ndarray        # sub-sample peak times (s)
    ppi_ms: np.ndarray            # pulse-to-pulse intervals (ms), len = beats - 1
    ppi_irregular: np.ndarray     # bool per PPI: out of range / deviates from neighbours
    rgb: np.ndarray               # (N, 3) resampled mean R,G,B traces, trimmed
    metrics: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 1. Video -> per-frame channel means
# --------------------------------------------------------------------------- #
def read_video_traces(video_path: str, roi_fraction: float = 0.6,
                      sat_level: int = 250):
    """Return per-frame mean (R, G, B), saturation fraction, timestamps, meta fps.

    Only a centred ROI is averaged: the frame edges often contain vignetting or
    parts not covered by the finger, which add noise.
    """
    if not 0 < roi_fraction <= 1:
        raise ValueError("roi_fraction must be in (0, 1]")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    meta_fps = cap.get(cv2.CAP_PROP_FPS)
    rgb, sat, ts = [], [], []

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            h, w = frame.shape[:2]
            dy, dx = int(h * (1 - roi_fraction) / 2), int(w * (1 - roi_fraction) / 2)
            roi = frame[dy:h - dy, dx:w - dx]            # BGR order
            b, g, r, _ = cv2.mean(roi)                   # fast spatial average, Eq. (1)
            rgb.append((r, g, b))
            sat.append(float(np.mean(roi[:, :, 2] >= sat_level)))
            ts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
    finally:
        cap.release()

    if len(rgb) < 10:
        raise ValueError("Video has too few readable frames.")

    return np.asarray(rgb), np.asarray(sat), np.asarray(ts), meta_fps


# --------------------------------------------------------------------------- #
# 2. Reliable time axis + uniform resampling
# --------------------------------------------------------------------------- #
def build_time_axis(ts, n, meta_fps, effective_fps):
    """Pick the most trustworthy sampling clock. Returns (t, fs, source)."""
    if effective_fps:
        return np.arange(n) / effective_fps, float(effective_fps), "user override"

    if n > 2 and np.all(np.diff(ts) > 0):
        fs = (n - 1) / (ts[-1] - ts[0])
        return ts - ts[0], fs, "frame timestamps"

    if meta_fps and np.isfinite(meta_fps) and meta_fps > 1:
        return np.arange(n) / meta_fps, float(meta_fps), "container metadata"

    raise ValueError("Cannot determine frame rate; pass --fps explicitly.")


def resample_uniform(t, x, fs):
    """Linear-interpolate (possibly jittery) samples onto a uniform grid."""
    t_u = np.arange(0, t[-1], 1.0 / fs)
    if x.ndim == 1:
        return t_u, np.interp(t_u, t, x)
    return t_u, np.column_stack([np.interp(t_u, t, x[:, k]) for k in range(x.shape[1])])


# --------------------------------------------------------------------------- #
# 3. Filtering
# --------------------------------------------------------------------------- #
def design_bandpass(fs, low=0.7, high=4.0, ftype="cheby2",
                    gpass=1.0, gstop=40.0):
    """Return second-order sections for a band-pass filter.

    cheby2 (default, as in Zhang et al. 2021): flat pass band, >= `gstop` dB
    rejection outside the stop-band edges. The order is chosen automatically
    with cheb2ord from explicit pass-band and stop-band edges, so the pass band
    really is [low, high] (cheby2's own `Wn` is the *stop-band* edge, which is
    easy to get wrong). SOS form avoids the numerical instability of (b, a)
    at very low normalised cutoffs such as 0.7 Hz / 50 Hz.
    """
    nyq = fs / 2
    high = min(high, 0.9 * nyq)
    if low >= high:
        raise ValueError(f"Sampling rate {fs:.1f} Hz too low for band {low}-{high} Hz")

    if ftype == "butter":
        return signal.butter(3, [low, high], btype="band", fs=fs, output="sos")
    if ftype == "cheby2":
        stop = [low * 0.5, min(high * 1.5, 0.98 * nyq)]
        order, wn = signal.cheb2ord([low, high], stop, gpass, gstop, fs=fs)
        return signal.cheby2(order, gstop, wn, btype="band", fs=fs, output="sos")
    raise ValueError("ftype must be 'cheby2' or 'butter'")


def bandpass(x, fs, low=0.7, high=4.0, ftype="cheby2"):
    """Zero-phase band-pass: removes drift (< low) and camera/motion wiggles (> high).

    The mean is removed first (raw red intensity sits at a large DC offset that
    would otherwise excite the filter's start-up transient), then the filter is
    run forward and backward (sosfiltfilt) so peak timing is not shifted.
    """
    sos = design_bandpass(fs, low, high, ftype)
    x = np.asarray(x, dtype=np.float64) - np.mean(x)
    if len(x) <= 3 * (2 * len(sos) + 1):
        raise ValueError("Signal too short to filter.")
    return signal.sosfiltfilt(sos, x)


# --------------------------------------------------------------------------- #
# 4. Beat detection, PPI extraction, HRV
# --------------------------------------------------------------------------- #
def detect_beats(x, fs, prom_frac=0.15, min_gap_s=0.33, edge_margin_s=1.0):
    """Sample indices of pulse peaks in the filtered PPG (Section II-C).

    - Prominence is *relative* to the signal's own amplitude (5th-95th percentile
      range), so it works whatever the camera gain / finger pressure; a fixed
      absolute value (e.g. 0.5) only suits one recording.
    - The typical beat period is taken from the spectrum and peaks closer than
      half of it are forbidden, which rejects the dicrotic-notch bump that is
      often mistaken for a second beat (at 30 fps a bump ~0.30 s after the
      main peak slips past a plain 0.33 s gap). The 0.33 s floor (~180 bpm)
      still applies; premature beats at > ~50 % of the normal interval survive.
    - Peaks within `edge_margin_s` of either end are dropped: the zero-phase
      filter leaves edge artefacts there that look like real beats.
    """
    x = np.asarray(x, dtype=np.float64)
    amp = np.percentile(x, 95) - np.percentile(x, 5)
    prom = prom_frac * amp
    floor = max(int(min_gap_s * fs), 1)

    # Typical beat period from the spectrum (robust even when the time-domain
    # peaks are contaminated by dicrotic bumps), then no peaks closer than half.
    dist = floor
    try:
        hr_bpm, _, _ = spectral_hr(x, fs)
        dist = max(floor, int(0.5 * (60.0 / hr_bpm) * fs))
    except (ValueError, ZeroDivisionError):
        pass
    peaks, _ = signal.find_peaks(x, distance=dist, prominence=prom)

    m = max(int(edge_margin_s * fs), 1)
    return peaks[(peaks >= m) & (peaks < len(x) - m)]


def refine_peak_times(t, x, peaks):
    """Sub-sample peak times by fitting a parabola through each peak and its
    two neighbours. At 30 fps the raw sample grid limits timing to +-17 ms
    (33 ms steps), which shows up as fake HRV; this removes most of it."""
    peaks = np.asarray(peaks, dtype=int)
    if len(peaks) == 0:
        return np.array([])
    dt = t[1] - t[0]
    y0, y1, y2 = x[peaks - 1], x[peaks], x[peaks + 1]
    denom = y0 - 2 * y1 + y2
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(denom != 0, 0.5 * (y0 - y2) / denom, 0.0)
    return t[peaks] + np.clip(delta, -0.5, 0.5) * dt


def compute_ppi(peak_times, lo_ms=300.0, hi_ms=2000.0, tol=0.20):
    """Pulse-to-pulse intervals (ms) plus quality flags.

    Returns (ppi, irregular, normal). An interval is `irregular` if it is
    outside [lo_ms, hi_ms] or differs from the median of its neighbours
    (window of 5) by more than `tol` (the usual 20 % rule). Irregular
    intervals are FLAGGED, not deleted: they may be a missed/extra detection
    OR a genuine ectopic beat, and for arrhythmia work you want to keep them.
    HRV statistics use only the `normal` ones.
    """
    if len(peak_times) < 2:
        e = np.array([])
        return e, e.astype(bool), e.astype(bool)
    ppi = np.diff(peak_times) * 1000.0
    ref = median_filter(ppi, size=5, mode="nearest")
    out_of_range = (ppi < lo_ms) | (ppi > hi_ms)
    deviates = np.abs(ppi - ref) > tol * ref
    irregular = out_of_range | deviates
    return ppi, irregular, ~irregular


# The 16 HRV parameters of Zhang et al. (2021), IEEE Access, Table 1.
HRV16_ORDER = ["Mean_HR", "Max_HR", "Min_HR", "SDNN", "RMSSD", "NN50", "pNN50",
               "VLF", "LF", "HF", "TP", "LFnu", "HFnu", "LF_HF", "SD1", "SD2"]
HRV16_UNITS = {"Mean_HR": "bpm", "Max_HR": "bpm", "Min_HR": "bpm", "SDNN": "ms",
               "RMSSD": "ms", "NN50": "count", "pNN50": "%", "VLF": "ms^2",
               "LF": "ms^2", "HF": "ms^2", "TP": "ms^2", "LFnu": "n.u.",
               "HFnu": "n.u.", "LF_HF": "", "SD1": "ms", "SD2": "ms"}
VLF_BAND, LF_BAND, HF_BAND = (0.0, 0.04), (0.04, 0.15), (0.15, 0.5)   # Hz


def hrv_time_domain(ppi, normal):
    """Time-domain HRV (Task Force of ESC/NASPE, 1996) on NN intervals in ms.

    NN = the `normal` pulse-to-pulse intervals; N = their count.
      Mean HR = 60000 / mean(NN)                      [bpm]
      Max/Min HR = max/min of instantaneous HR_i = 60000 / NN_i
      SDNN  = sqrt( sum_i (NN_i - mean NN)^2 / (N - 1) )
      RMSSD = sqrt( sum_i (NN_{i+1} - NN_i)^2 / M )   M = # successive pairs
      NN50  = #{ i : |NN_{i+1} - NN_i| > 50 ms }
      pNN50 = 100 * NN50 / M
    Successive differences use only pairs where BOTH intervals are normal, so a
    flagged (ectopic/missed-beat) interval never contaminates RMSSD/NN50.
    """
    keys = ["Mean_HR", "Max_HR", "Min_HR", "SDNN", "RMSSD", "NN50", "pNN50"]
    out = dict.fromkeys(keys, np.nan)
    nn = ppi[normal]
    n = len(nn)
    if n < 3:
        return out
    hr = 60000.0 / nn
    out["Mean_HR"] = 60000.0 / nn.mean()
    out["Max_HR"], out["Min_HR"] = hr.max(), hr.min()
    out["SDNN"] = np.sqrt(np.sum((nn - nn.mean()) ** 2) / (n - 1))
    d = np.diff(ppi)[normal[:-1] & normal[1:]]
    if len(d) >= 2:
        nn50 = int(np.sum(np.abs(d) > 50.0))
        out["RMSSD"] = np.sqrt(np.sum(d ** 2) / len(d))
        out["NN50"] = float(nn50)
        out["pNN50"] = 100.0 * nn50 / len(d)
    return out


def band_power(f, p, lo, hi):
    """Integral of the PSD over [lo, hi] Hz (trapezoid rule, exact band edges
    obtained by linear interpolation of the PSD)."""
    hi = min(hi, f[-1])
    inside = (f > lo) & (f < hi)
    fx = np.concatenate(([lo], f[inside], [hi]))
    px = np.concatenate(([np.interp(lo, f, p)], p[inside], [np.interp(hi, f, p)]))
    return float(trapezoid(px, fx))


def hrv_frequency_domain(t_nn, nn, fs_interp=4.0):
    """Frequency-domain HRV from the NN series (ms) at times t_nn (s).

    1. The NN series is unevenly sampled (one value per beat), so it is cubic-
       spline interpolated onto a uniform 4 Hz grid. Using only normal intervals
       means any flagged interval is bridged by the spline.
    2. PSD by Welch's method: Hann window, linear detrend, segments of up to
       120 s with 50 % overlap, zero-padded FFT; scaling = density [ms^2/Hz].
    3. Band powers P = integral of PSD over the band [ms^2]:
         VLF 0-0.04 Hz, LF 0.04-0.15 Hz, HF 0.15-0.5 Hz, TP = 0-0.5 Hz
         LFnu = 100 * LF / (TP - VLF),  HFnu = 100 * HF / (TP - VLF)
         LF/HF = LF / HF
    """
    from scipy.interpolate import CubicSpline

    keys = ["VLF", "LF", "HF", "TP", "LFnu", "HFnu", "LF_HF"]
    out = dict.fromkeys(keys, np.nan)
    if len(nn) < 10 or (t_nn[-1] - t_nn[0]) < 20.0:
        return out

    t_u = np.arange(t_nn[0], t_nn[-1], 1.0 / fs_interp)
    x = CubicSpline(t_nn, nn)(t_u)
    nperseg = min(len(x), int(120 * fs_interp))
    nfft = max(4096, 1 << (nperseg - 1).bit_length())
    f, p = signal.welch(x, fs=fs_interp, window="hann", nperseg=nperseg,
                        noverlap=nperseg // 2, nfft=nfft, detrend="linear",
                        scaling="density")

    vlf, lf, hf = (band_power(f, p, *VLF_BAND), band_power(f, p, *LF_BAND),
                   band_power(f, p, *HF_BAND))
    tp = band_power(f, p, 0.0, 0.5)
    out.update(VLF=vlf, LF=lf, HF=hf, TP=tp)
    if tp - vlf > 0:
        out["LFnu"] = 100.0 * lf / (tp - vlf)
        out["HFnu"] = 100.0 * hf / (tp - vlf)
    if hf > 0:
        out["LF_HF"] = lf / hf
    return out


def hrv_poincare(ppi, normal):
    """Poincare-plot descriptors from successive normal pairs (x, y) = (NN_i, NN_{i+1}).

    The cloud is rotated 45 degrees; SD1/SD2 are the standard deviations across
    and along the line of identity:
         SD1 = std( (y - x) / sqrt(2) )   = sqrt( 0.5 * Var(NN_{i+1} - NN_i) )
         SD2 = std( (y + x) / sqrt(2) )   = sqrt( 2*SDNN^2 - 0.5*Var(dNN) )
    """
    out = {"SD1": np.nan, "SD2": np.nan}
    ok = normal[:-1] & normal[1:]
    if ok.sum() < 3:
        return out
    x, y = ppi[:-1][ok], ppi[1:][ok]
    out["SD1"] = float(np.std((y - x) / np.sqrt(2.0), ddof=1))
    out["SD2"] = float(np.std((y + x) / np.sqrt(2.0), ddof=1))
    return out


def hrv_metrics(peak_times, ppi, normal):
    """All 16 HRV parameters (always present; NaN if not computable) plus
    bookkeeping. Returns (metrics, notes). A PPI that ends at beat k+1 is
    stamped with peak_times[k+1] for the spectral analysis."""
    metrics = dict.fromkeys(HRV16_ORDER, np.nan)
    notes = []
    metrics.update(hrv_time_domain(ppi, normal))
    metrics.update(hrv_poincare(ppi, normal))

    if len(ppi) and normal.sum() >= 10:
        t_end = np.asarray(peak_times)[1:][normal]
        metrics.update(hrv_frequency_domain(t_end, ppi[normal]))
        span = t_end[-1] - t_end[0]
        if span < 100:
            notes.append(f"Only {span:.0f} s of usable data: LF/HF resolution is poor. "
                         "Record >= ~2 min (a few seconds at each end are trimmed).")
        elif span < 300:
            notes.append(f"{span:.0f} s of usable data: fine for LF/HF, but VLF power is "
                         "unreliable below ~5 min; treat it with caution.")
    else:
        notes.append("Too few normal intervals for frequency-domain HRV.")

    metrics["n_beats"] = int(len(peak_times))
    metrics["n_NN"] = int(normal.sum())
    metrics["n_irregular_PPI"] = int((~normal).sum())
    if len(ppi):
        metrics["irregular_pct"] = float(100.0 * np.mean(~normal))
    return metrics, notes


def hrv_feature_vector(metrics, log_power=True):
    """The 16 parameters as a fixed-order vector for the classifier. As in the
    paper, the (skewed) spectral powers can be natural-log transformed."""
    v = np.array([metrics[k] for k in HRV16_ORDER], dtype=float)
    if log_power:
        for k in ("VLF", "LF", "HF", "TP"):
            i = HRV16_ORDER.index(k)
            v[i] = np.log(v[i]) if v[i] > 0 else np.nan
    return v


def print_hrv(m):
    groups = [("Time domain", HRV16_ORDER[:7]), ("Frequency domain", HRV16_ORDER[7:14]),
              ("Poincare", HRV16_ORDER[14:])]
    for title, keys in groups:
        print(f"\n{title}")
        for k in keys:
            v = m.get(k, np.nan)
            txt = f"{v:10.2f}  {HRV16_UNITS[k]}" if np.isfinite(v) else f"{'n/a':>10s}"
            print(f"  {k:8s} {txt}")
    print("\nOther")
    for k in ("n_beats", "n_NN", "n_irregular_PPI", "irregular_pct", "HR_spectral_bpm"):
        if k in m:
            v = m[k]
            print(f"  {k:16s} {v:.1f}" if isinstance(v, float) else f"  {k:16s} {v}")


def spectral_hr(x, fs, low=0.7, high=4.0):
    nperseg = min(len(x), int(fs * 15))
    nfft = max(int(fs * 120), nperseg)           # ~0.5 bpm frequency grid
    f, p = signal.welch(x, fs, nperseg=nperseg, nfft=nfft)
    band = (f >= low) & (f <= min(high, fs / 2))
    return 60.0 * f[band][np.argmax(p[band])], f, p


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def extract_ppg(video_path: str, effective_fps: float | None = None,
                trim_start: float = 3.0, roi_fraction: float = 0.6,
                band=(0.7, 4.0), filter_type: str = "cheby2") -> PPGResult:
    warnings: list[str] = []

    rgb, sat, ts, meta_fps = read_video_traces(video_path, roi_fraction)
    t, fs, src = build_time_axis(ts, len(rgb), meta_fps, effective_fps)
    t, rgb = resample_uniform(t, rgb, fs)
    print(f"{len(rgb)} samples, fs = {fs:.2f} Hz ({src}), duration = {t[-1]:.1f} s")

    # --- quality checks on the raw traces ---------------------------------- #
    if np.mean(sat) > 0.10:
        warnings.append(f"Red channel saturated in {100*np.mean(sat):.0f}% of ROI pixels "
                        "-> clipped pulse. Reduce exposure / press more lightly.")
    if np.mean(rgb[:, 0]) < 100 or np.mean(rgb[:, 0]) < 1.5 * np.mean(rgb[:, 1]):
        warnings.append("Frame does not look dominated by red -> finger may not fully "
                        "cover lens and flash.")
    if fs < 20:
        warnings.append(f"Low sampling rate ({fs:.1f} Hz): HRV timing will be coarse.")

    # --- trim camera auto-exposure / finger-settling transient ------------- #
    keep = t >= trim_start
    if keep.sum() < fs * 8:
        warnings.append("Video too short to trim start-up transient; using all of it.")
        keep = np.ones_like(t, dtype=bool)
    t, rgb = t[keep] - t[keep][0], rgb[keep]

    # Absorption increases with blood volume -> intensity dips; invert so pulses go up.
    raw = -rgb[:, 0]
    filt = bandpass(raw, fs, *band, ftype=filter_type)

    peaks = detect_beats(filt, fs)
    peak_times = refine_peak_times(t, filt, peaks)
    ppi, irregular, normal = compute_ppi(peak_times)
    metrics, hrv_notes = hrv_metrics(peak_times, ppi, normal)
    warnings.extend(hrv_notes)

    if len(peaks) < 5:
        warnings.append(f"Only {len(peaks)} beats detected -> metrics unreliable.")
    elif metrics.get("irregular_pct", 0) > 30:
        warnings.append(f"{metrics['irregular_pct']:.0f}% of intervals are irregular -> "
                        "noisy signal OR arrhythmia; inspect the tachogram.")

    hr_spec, _, _ = spectral_hr(filt, fs, *band)
    metrics["HR_spectral_bpm"] = hr_spec
    if np.isfinite(metrics["Mean_HR"]) and abs(metrics["Mean_HR"] - hr_spec) > 8:
        warnings.append("Peak-based and spectral HR disagree -> noisy signal or "
                        "irregular rhythm; inspect the plot.")

    return PPGResult(t, raw, filt, fs, peaks, peak_times, ppi, irregular,
                     rgb, metrics, warnings)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_result(res: PPGResult, window=(0, 15), save=None, show=True):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0, t1 = window
    m = (res.t >= t0) & (res.t <= t1)
    pt = res.peak_times
    pw = (pt >= t0) & (pt <= t1)
    _, f, p = spectral_hr(res.filtered, res.fs)

    fig, ax = plt.subplots(4, 1, figsize=(11, 12))
    ax[0].plot(res.t[m], (res.raw - res.raw.mean())[m], color="tab:blue", alpha=0.5,
               lw=1, label="Raw (inverted mean red, mean removed)")
    ax[0].plot(res.t[m], res.filtered[m], color="crimson", lw=1.4, label="Filtered")
    ax[0].set(title="Before vs after filtering (drift and wiggles removed)",
              ylabel="Amplitude (a.u.)")
    ax[0].legend(loc="upper right")

    ax[1].plot(res.t[m], res.filtered[m], color="crimson", lw=1.4)
    ax[1].plot(pt[pw], np.interp(pt[pw], res.t, res.filtered), "go", ms=6,
               label=f"detected peaks ({len(pt)} total)")
    ax[1].set(title="Peak detection (sub-sample timing)", xlabel="Time (s)",
              ylabel="Amplitude")
    ax[1].legend(loc="upper right")

    if len(res.ppi_ms):
        tt = pt[1:]                                   # each PPI ends at its 2nd beat
        ax[2].plot(tt, res.ppi_ms, "-", color="gray", lw=0.8)
        ax[2].plot(tt[~res.ppi_irregular], res.ppi_ms[~res.ppi_irregular], "o",
                   color="tab:green", ms=4, label="normal")
        if res.ppi_irregular.any():
            ax[2].plot(tt[res.ppi_irregular], res.ppi_ms[res.ppi_irregular], "o",
                       color="tab:orange", ms=6, label="irregular (flagged)")
        ax[2].legend(loc="upper right")
    ax[2].set(title="PPI tachogram (whole recording)", xlabel="Time (s)",
              ylabel="PPI (ms)")

    ax[3].plot(f * 60, p, color="navy")
    ax[3].set(xlim=(30, 240), title="Power spectrum", xlabel="Rate (bpm)", ylabel="PSD")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    if show:
        plt.show()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="Extract PPG from a fingertip video.")
    ap.add_argument("video", nargs="?", default="fingertip_test.mp4")
    ap.add_argument("--fps", type=float, default=None,
                    help="Effective fps override (e.g. 100 for 4x slow-mo of 25 fps)")
    ap.add_argument("--trim", type=float, default=3.0, help="Seconds to discard at start")
    ap.add_argument("--roi", type=float, default=0.6, help="Centre ROI fraction (0-1]")
    ap.add_argument("--filter", dest="ftype", choices=["cheby2", "butter"],
                    default="cheby2", help="Band-pass design (default: Chebyshev II)")
    ap.add_argument("--band", nargs=2, type=float, default=[0.7, 4.0],
                    metavar=("LOW", "HIGH"), help="Pass band in Hz (default 0.7 4.0)")
    ap.add_argument("--save", default=None, help="Save figure to this path")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args(argv)

    res = extract_ppg(args.video, args.fps, args.trim, args.roi,
                      tuple(args.band), args.ftype)

    print_hrv(res.metrics)
    for w in res.warnings:
        print(f"WARNING: {w}", file=sys.stderr)

    plot_result(res, window=(0, min(15, res.t[-1])), save=args.save, show=not args.no_show)
    return res


if __name__ == "__main__":
    main()