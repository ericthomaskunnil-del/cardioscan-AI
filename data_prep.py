"""Loads MIT-BIH, QT, CPSC-2020, UoG -> list of (dataset, record, signal@400Hz, R-peak indices@400Hz)."""
import os, glob, pickle, urllib.request, numpy as np, wfdb
from concurrent.futures import ThreadPoolExecutor
from scipy.signal import resample
from ecg.dsp import denoise, BEAT_SYMBOLS
from ecg.rpeak import FS_T

CACHE = "data/rpeak_cache.pkl"
CACHE_DIR = "data/cache"
S3_BASE = "https://physionet-open.s3.amazonaws.com"

def _prep(ds, rec, sig, fs, peaks):
    n = int(len(sig) * FS_T / fs)
    x = denoise(resample(sig, n), FS_T)
    p = np.unique(np.round(np.asarray(peaks) * FS_T / fs).astype(int))
    p = p[(p >= 0) & (p < n)]
    return (ds, str(rec), x.astype("float32"), p)

def _download_file(url, target_path):
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        return True
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    try:
        urllib.request.urlretrieve(url, target_path)
        return True
    except Exception:
        if os.path.exists(target_path):
            os.remove(target_path)
        return False

def _fetch_mit_record_files(r):
    local_dir = "data/mitdb"
    for ext in ("hea", "dat", "atr"):
        target = f"{local_dir}/{r}.{ext}"
        url = f"{S3_BASE}/mitdb/1.0.0/{r}.{ext}"
        if not _download_file(url, target):
            return False
    return True

def mitdb():
    cache_path = os.path.join(CACHE_DIR, "mit.pkl")
    if os.path.exists(cache_path):
        print("Loaded MIT-BIH from cache.")
        return pickle.load(open(cache_path, "rb"))

    recs = [r for r in wfdb.get_record_list("mitdb") if r not in ("108", "207")]
    print(f"Downloading and preparing MIT-BIH ({len(recs)} records) via high-speed S3 mirror...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(_fetch_mit_record_files, recs))

    out = []
    for r in recs:
        try:
            rec = wfdb.rdrecord(f"data/mitdb/{r}", channels=[0])
            a = wfdb.rdann(f"data/mitdb/{r}", "atr")
            peaks = [s for s, y in zip(a.sample, a.symbol) if y in BEAT_SYMBOLS]
            out.append(_prep("mit", r, rec.p_signal[:, 0], 360, peaks))
        except Exception as e:
            print(f"  [MIT-BIH] skipping record {r}: {e}")

    os.makedirs(CACHE_DIR, exist_ok=True)
    pickle.dump(out, open(cache_path, "wb"))
    print(f"MIT-BIH prepared: {len(out)} records.")
    return out

def _fetch_qt_record_files(r):
    local_dir = "data/qtdb"
    for ext in ("hea", "dat"):
        _download_file(f"{S3_BASE}/qtdb/1.0.0/{r}.{ext}", f"{local_dir}/{r}.{ext}")
    _download_file(f"{S3_BASE}/qtdb/1.0.0/{r}.q1c", f"{local_dir}/{r}.q1c")
    _download_file(f"{S3_BASE}/qtdb/1.0.0/{r}.pu0", f"{local_dir}/{r}.pu0")
    return True

def qtdb():
    cache_path = os.path.join(CACHE_DIR, "qt.pkl")
    if os.path.exists(cache_path):
        print("Loaded QT from cache.")
        return pickle.load(open(cache_path, "rb"))

    recs = wfdb.get_record_list("qtdb")
    print(f"Downloading and preparing QT database ({len(recs)} records) via high-speed S3 mirror...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(_fetch_qt_record_files, recs))

    out = []
    for r in recs:
        try:
            rec = wfdb.rdrecord(f"data/qtdb/{r}", channels=[0])
            try:
                a = wfdb.rdann(f"data/qtdb/{r}", "q1c")
            except Exception:
                a = wfdb.rdann(f"data/qtdb/{r}", "pu0")
            peaks = [s for s, y in zip(a.sample, a.symbol) if y == "N"]
            out.append(_prep("qt", r, rec.p_signal[:, 0], 250, peaks))
        except Exception as e:
            pass

    os.makedirs(CACHE_DIR, exist_ok=True)
    pickle.dump(out, open(cache_path, "wb"))
    print(f"QT database prepared: {len(out)} records.")
    return out

def cpsc():
    """MANUAL download: get CPSC2020 training set from http://2020.icbeb.org/CSPC2020 and unzip so that
    data/cpsc2020/TrainingSet/data/A01.mat and data/cpsc2020/TrainingSet/ref/R01.mat exist. 400 Hz."""
    import scipy.io as sio
    out = []
    for f in sorted(glob.glob("data/cpsc2020/TrainingSet/data/A*.mat")):
        n = os.path.basename(f)[1:3]
        ref = f"data/cpsc2020/TrainingSet/ref/R{n}.mat"
        if not os.path.exists(ref):
            continue
        try:
            sig = np.squeeze(sio.loadmat(f)["ecg"]).astype(float)
            peaks = np.squeeze(sio.loadmat(ref)["ref"]["R_peak"][0][0])
            out.append(_prep("cpsc", n, sig, 400, peaks))
        except Exception as e:
            print("skip cpsc", f, e)
    if not out:
        print("CPSC-2020 not found - To include it, download CPSC2020 training set from http://2020.icbeb.org/CSPC2020 and place it in data/cpsc2020/TrainingSet/ (data/ holds A*.mat, ref/ holds R*.mat). Continuing with other datasets.")
    return out

def uog():
    cache_path = os.path.join(CACHE_DIR, "uog.pkl")
    if os.path.exists(cache_path):
        print("Loaded UoG from cache.")
        return pickle.load(open(cache_path, "rb"))

    from ecg_gudb_database import GUDb
    out = []
    print("Loading UoG (Glasgow University Database)...")
    for u in range(25):
        for act in ("sitting", "maths", "walking"):
            try:
                e = GUDb(u, act)
                anno = getattr(e, "anno_cs", getattr(e, "cs_anno", None))
                sig = getattr(e, "einthoven_II", getattr(e, "einthoven_ii", None))
                if getattr(e, "anno_cs_exists", True) and anno is not None and sig is not None:
                    out.append(_prep("uog", f"{u}_{act}", np.asarray(sig), 250, anno))
            except Exception as ex:
                pass
    print(f"  [UoG] loaded {len(out)} records")
    os.makedirs(CACHE_DIR, exist_ok=True)
    pickle.dump(out, open(cache_path, "wb"))
    return out

def load_all():
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    os.makedirs("data", exist_ok=True)
    d = {"mit": mitdb(), "qt": qtdb(), "cpsc": cpsc(), "uog": uog()}
    pickle.dump(d, open(CACHE, "wb"))
    return d

if __name__ == "__main__":
    data = load_all()
    print("\n--- Dataset Summary ---")
    for k, v in data.items():
        n_peaks = sum(len(rec[3]) for rec in v)
        print(f"{k.upper():6s}: {len(v):3d} records, {n_peaks:6d} R-peaks")
