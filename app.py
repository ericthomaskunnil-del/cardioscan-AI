"""FastAPI backend + static hospital-style UI. Run: uvicorn app:app --reload"""
import io, os, numpy as np, wfdb
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from scipy.signal import resample
from ecg.dsp import *
from ecg.rpeak import detect, FS_T
from ecg.explain import features, check, agent_summary, grad_cam

app = FastAPI(title="CardioScan AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
_model = None
_rm = None

def model():
    global _model
    if _model is None and os.path.exists("ecg_model.keras"):
        import tensorflow as tf
        try: _model = tf.keras.models.load_model("ecg_model.keras")
        except Exception: _model = None
    return _model

def peaks_for(x):
    """Use the paper's R-peak CNN if trained, else fall back to the classical detector."""
    global _rm
    if _rm is None and os.path.exists("rpeak_model.keras"):
        import tensorflow as tf; _rm = tf.keras.models.load_model("rpeak_model.keras")
    if _rm is None: return r_peaks(x)
    p = detect(_rm, resample(x, int(len(x) * FS_T / FS)))
    return np.round(p * FS / FS_T).astype(int)

def analyze(raw, fs):
    if fs != FS: raw = resample(raw, int(len(raw) * FS / fs))
    x = denoise(raw); peaks = peaks_for(x)
    if len(peaks) < 3: raise HTTPException(422, "Too few heartbeats detected.")
    segs, kept = segment(x, peaks)
    m = model()
    if m is not None:
        probs = m.predict(segs, verbose=0); cls = probs.argmax(1)
    else:
        probs, cls = None, None
    beats = []
    for i, p in enumerate(kept):
        f = features(x, kept, i)
        if m is not None:
            c = cls[i]
            name = CLASSES[c]
            conf = float(probs[i, c])
            sal = grad_cam(m, segs[i], c)
        else:
            if f["premature"] and f["wide_qrs"]:
                name, conf = "PVC", 0.88
            elif f["wide_qrs"]:
                name, conf = "RBBB", 0.82
            elif f["premature"]:
                name, conf = "Atrial premature", 0.85
            else:
                name, conf = "Normal", 0.95
            sal = None
        sup, con = check(name, f)
        beats.append(dict(index=int(p), label=name, confidence=conf, features=f,
                          supports=sup, conflicts=con, saliency=sal))
    abn = [b for b in beats if b["label"] != "Normal"]
    focus = max(abn, key=lambda b: b["confidence"]) if abn else beats[len(beats)//2]
    hr = 60 * FS / np.median(np.diff(kept))
    sum_text = agent_summary(focus["label"], focus["confidence"], focus["features"],
                             focus["supports"], focus["conflicts"])
    if m is None:
        sum_text += " (Phase 2 R-peak model & clinical rules active; Phase 3 CNN classifier pending training)."
    return dict(fs=FS, raw=raw.tolist(), clean=x.tolist(), beats=beats, heart_rate=round(float(hr)),
                abnormal=len(abn), summary=sum_text)

def parse_ecg_csv(content: bytes) -> np.ndarray:
    text = content.decode("utf-8", errors="ignore")
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    if not lines:
        raise ValueError("The uploaded CSV file is empty.")
    
    start_row = 0
    first_row_parts = [p.strip().strip('"').strip("'") for p in lines[0].replace(";", ",").replace("\t", ",").split(",") if p.strip()]
    try:
        float(first_row_parts[0])
    except (ValueError, IndexError):
        start_row = 1  # Header detected, skip first row
        
    data_rows = []
    for line in lines[start_row:]:
        parts = [p.strip().strip('"').strip("'") for p in line.replace(";", ",").replace("\t", ",").split(",") if p.strip()]
        vals = []
        for p in parts:
            try:
                vals.append(float(p))
            except ValueError:
                pass
        if vals:
            data_rows.append(vals)
            
    if not data_rows:
        raise ValueError("No numeric ECG voltage samples found in file.")
        
    arr = np.array(data_rows, dtype=float)
    if arr.ndim == 2:
        if arr.shape[1] >= 2:
            col0_diff = np.diff(arr[:, 0])
            # If col 0 is sequential timestamp/index (0, 1, 2... or 0.00, 0.01...), take col 1
            if len(col0_diff) > 0 and np.all(col0_diff > 0):
                return arr[:, 1]
            return arr[:, -1]
        return arr[:, 0]
    return arr

@app.post("/api/analyze")
async def upload(file: UploadFile = File(...), fs: int = Form(360)):
    try:
        content = await file.read()
        raw = parse_ecg_csv(content)
    except Exception as e:
        raise HTTPException(422, f"CSV parsing error: {str(e)}")
    if len(raw) < 100:
        raise HTTPException(422, "ECG signal is too short. Please provide at least 100 samples.")
    return analyze(raw.astype(float), fs)

@app.get("/api/samples")
def list_samples():
    return [
        {"id": "100", "filename": "record_100_normal.csv", "name": "Record 100 (Normal Sinus)", "label": "Normal Rhythm", "fs": 360},
        {"id": "119", "filename": "record_119_pvc_arrhythmia.csv", "name": "Record 119 (PVC Arrhythmia)", "label": "Premature Ventricular Contractions", "fs": 360},
        {"id": "214", "filename": "record_214_rbbb.csv", "name": "Record 214 (RBBB)", "label": "Right Bundle Branch Block", "fs": 360},
        {"id": "106", "filename": "record_106_bigeminy.csv", "name": "Record 106 (Bigeminy)", "label": "Ventricular Bigeminy", "fs": 360},
    ]

@app.get("/api/demo/{record}")
def demo(record: int = 100):
    rec_path = os.path.join("data", "mitdb", str(record))
    if os.path.exists(rec_path + ".hea"):
        rec = wfdb.rdrecord(rec_path, sampto=3600 * 2, channels=[0])
    else:
        rec = wfdb.rdrecord(str(record), pn_dir="mitdb", sampto=3600 * 2, channels=[0])
    return analyze(rec.p_signal[:, 0], FS)

if os.path.exists("sample_ecgs"):
    app.mount("/sample_ecgs", StaticFiles(directory="sample_ecgs"), name="sample_ecgs")

app.mount("/", StaticFiles(directory="static", html=True), name="static")
