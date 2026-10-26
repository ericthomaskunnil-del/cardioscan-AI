"""FastAPI backend + static hospital-style UI. Run: uvicorn app:app --reload"""
import io, os, numpy as np, wfdb
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from scipy.signal import resample
from ecg.dsp import *
from ecg.rpeak import detect, FS_T
from ecg.explain import features, check, agent_summary, grad_cam

app = FastAPI(title="CardioScan AI")
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

@app.post("/api/analyze")
async def upload(file: UploadFile = File(...), fs: int = Form(360)):
    try: raw = np.loadtxt(io.BytesIO(await file.read()), delimiter=",", usecols=0, skiprows=0, ndmin=1)
    except Exception: raise HTTPException(422, "Upload a CSV with one ECG sample per row.")
    return analyze(raw.astype(float), fs)

@app.get("/api/demo/{record}")
def demo(record: int = 100):
    rec_path = os.path.join("data", "mitdb", str(record))
    if os.path.exists(rec_path + ".hea"):
        rec = wfdb.rdrecord(rec_path, sampto=3600 * 2, channels=[0])
    else:
        rec = wfdb.rdrecord(str(record), pn_dir="mitdb", sampto=3600 * 2, channels=[0])
    return analyze(rec.p_signal[:, 0], FS)

app.mount("/", StaticFiles(directory="static", html=True), name="static")
