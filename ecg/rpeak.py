"""Paper Task 1: knowledge integration (array K) -> encoder-decoder CNN -> post-processing."""
import numpy as np
FS_T, FRAG = 400, 8000            # all datasets resampled to 400 Hz, 8000-sample fragments

def knowledge(x):
    """Step 1: in each 260-sample window find the max deviation; if it is a true peak (not on the window edge)
    mark +-20 samples with 1, then skip 100 samples (R peaks recur at regular intervals)."""
    k = np.zeros(len(x), "float32"); i = 0
    while i + 260 <= len(x):
        m = int(np.argmax(x[i:i + 260]))
        if 0 < m < 259:
            k[max(0, i + m - 20): i + m + 21] = 1; i += m + 100
        else:
            i += 130
    return k

def postprocess(P, thr=0.1, w=70):
    """Step 3: threshold 0.1, take argmax inside a 70-sample (175 ms) window, then skip the window."""
    out, i = [], 0
    while i < len(P):
        if P[i] > thr:
            out.append(i + int(np.argmax(P[i:i + w]))); i += w
        else:
            i += 1
    return np.array(out, dtype=int)

def make_input(x):
    x = (x - x.mean()) / (x.std() + 1e-8)
    return np.stack([x, knowledge(x)], -1).astype("float32")      # channels: S and K

def build():
    """Table A1: 4x (Conv-ReLU-MaxPool) encoder, 4x (Upsample-Conv) decoder, sigmoid output per sample."""
    import tensorflow as tf
    L = tf.keras.layers; i = L.Input((FRAG, 2)); x = i
    for f in (32, 64, 128, 256): x = L.MaxPool1D(2)(L.Conv1D(f, 3, padding="same", activation="relu")(x))
    for f in (128, 64, 32): x = L.Conv1D(f, 3, padding="same", activation="relu")(L.UpSampling1D(2)(x))
    x = L.Conv1D(1, 3, padding="same", activation="sigmoid")(L.UpSampling1D(2)(x))
    m = tf.keras.Model(i, L.Reshape((FRAG,))(x))
    m.compile(tf.keras.optimizers.Adam(1e-3), "binary_crossentropy"); return m

def detect(model, x400):
    n = len(x400); pad = (-n) % FRAG; x = np.pad(x400, (0, pad), mode="edge")
    X = np.stack([make_input(x[s:s + FRAG]) for s in range(0, len(x), FRAG)])
    P = model.predict(X, verbose=0).reshape(-1)[:n]
    return postprocess(P)
