"""Signal processing: denoising, R-peak detection, beat segmentation."""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

FS = 360          # MIT-BIH sampling rate
WIN = 700         # paper: 700 samples = previous + current + next cardiac cycle
CLASSES = ["Normal", "PVC", "Paced", "RBBB", "LBBB", "Atrial premature",
           "Fusion (V+N)", "Fusion (Paced+N)", "Others"]
SYM2CLASS = {"N": 0, "V": 1, "/": 2, "R": 3, "L": 4, "A": 5, "F": 6, "f": 7}
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ")      # annotation symbols that are heartbeats

def denoise(x, fs=FS):
    """Band-pass 0.5-40 Hz: removes baseline wander (<0.5 Hz) and muscle/mains noise (>40 Hz)."""
    b, a = butter(3, [0.5, 40], btype="band", fs=fs)
    return filtfilt(b, a, x)             # filtfilt = zero phase, so peaks do not shift

def r_peaks(x, fs=FS):
    """Classical fallback: derivative -> square -> moving average, then refine on the filtered signal."""
    s = np.diff(x, prepend=x[0]) ** 2
    w = int(0.12 * fs)
    m = np.convolve(s, np.ones(w) / w, mode="same")
    cand, _ = find_peaks(m, height=0.25 * np.percentile(m, 99), distance=int(0.25 * fs))
    h = int(0.08 * fs); out = []
    for c in cand:
        lo, hi = max(0, c - h), min(len(x), c + h)
        out.append(lo + int(np.argmax(np.abs(x[lo:hi] - np.median(x)))))
    return np.unique(out)

def segment(x, peaks, win=WIN):
    """Cut a window centred on every R peak; z-score each window. Returns (N,win,1) and kept peaks."""
    half = win // 2
    xp = np.pad(x, half, mode="edge")
    segs, kept = [], []
    for p in peaks:
        s = xp[p:p + win]
        segs.append((s - s.mean()) / (s.std() + 1e-8)); kept.append(p)
    return np.asarray(segs, dtype="float32")[..., None], np.asarray(kept)
