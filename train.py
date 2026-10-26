"""Trains the 9-class CNN on MIT-BIH, evaluated on unseen patients (inter-patient DS1/DS2 split)."""
import os, json, numpy as np, wfdb, tensorflow as tf
from tensorflow.keras import layers as L
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from ecg.dsp import *
tf.keras.utils.set_random_seed(42)   # reproducible runs

DS1 = [101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,207,208,209,215,220,223,230]
DS2 = [100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,221,222,228,231,232,233,234]

def load(records):
    X, y = [], []
    for r in records:
        path = f"data/mitdb/{r}"
        if not os.path.exists(path + ".hea"):
            print("skipping missing record", r)
            continue
        rec = wfdb.rdrecord(path, channels=[0])
        ann = wfdb.rdann(path, "atr")
        sig = denoise(rec.p_signal[:, 0])
        idx = [(s, a) for s, a in zip(ann.sample, ann.symbol) if a in BEAT_SYMBOLS]
        segs, _ = segment(sig, np.array([s for s, _ in idx]))
        X.append(segs); y += [SYM2CLASS.get(a, 8) for _, a in idx]
        print("loaded", r, len(idx), "beats")
    return np.concatenate(X), np.array(y)

def build(n=len(CLASSES)):
    """Table A2: Conv+BatchNorm blocks, MaxPool, Dense(1024)+BN+Dropout(0.68), Dense(128), softmax."""
    m = tf.keras.Sequential([L.Input((WIN, 1))])
    for f, k, s, pool in [(64,5,3,0),(64,5,2,1),(128,3,1,0),(128,3,2,0),(256,3,1,1)]:
        m.add(L.Conv1D(f, k, s, activation="relu")); m.add(L.BatchNormalization())
        if pool: m.add(L.MaxPool1D(2))
    m.add(L.Flatten()); m.add(L.Dense(1024, activation="relu")); m.add(L.BatchNormalization())
    m.add(L.Dropout(0.68)); m.add(L.Dense(128, activation="relu")); m.add(L.Dense(n, activation="softmax"))
    m.compile(tf.keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    return m

if __name__ == "__main__":
    Xtr, ytr = load(DS1); Xte, yte = load(DS2)
    # Augment rare classes (Fusion V+N, Fusion Paced+N, Others - idx 6, 7, 8)
    aug_X, aug_y = [], []
    for i, c in enumerate(ytr):
        if c in [6, 7, 8]:
            aug_X.append(Xtr[i] + np.random.normal(0, 0.05, Xtr[i].shape))  # noise
            aug_y.append(c)
            aug_X.append(np.roll(Xtr[i], 5, axis=0))                        # time shift
            aug_y.append(c)
    if aug_X:
        Xtr = np.concatenate([Xtr, np.array(aug_X)], axis=0)
        ytr = np.concatenate([ytr, np.array(aug_y)], axis=0)
    
    cw = {int(k): float(v) for k, v in zip(np.unique(ytr), compute_class_weight("balanced", classes=np.unique(ytr), y=ytr))}
    m = build()
    cb = [tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.1),
          tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)]
    m.fit(Xtr, ytr, validation_split=0.1, epochs=18, batch_size=128, class_weight=cw, callbacks=cb)
    m.save("ecg_model.keras")
    pred = m.predict(Xte).argmax(1)
    rep = classification_report(yte, pred, labels=range(9), target_names=CLASSES, output_dict=True, zero_division=0)
    json.dump({"report": rep, "confusion": confusion_matrix(yte, pred, labels=range(9)).tolist()},
              open("metrics.json", "w"), indent=1)
    print(classification_report(yte, pred, labels=range(9), target_names=CLASSES, zero_division=0))
