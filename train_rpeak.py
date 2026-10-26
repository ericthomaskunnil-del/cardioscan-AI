"""Trains the R-peak CNN on MIT-BIH + QT + CPSC (80/20 split BY RECORD), tests on UoG (independent)."""
import json, numpy as np, tensorflow as tf
from data_prep import load_all
from ecg.rpeak import *
SEED = 42; np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)

def fragments(recs):
    X, Y = [], []
    for _, _, x, p in recs:
        for s in range(0, len(x) - FRAG + 1, FRAG):
            q = p[(p >= s) & (p < s + FRAG)] - s
            if len(q) < 3: continue
            y = np.zeros(FRAG, "float32")
            for d in range(-3, 4): y[np.clip(q + d, 0, FRAG - 1)] = 1          # target: R peak +-3 samples
            X.append(make_input(x[s:s + FRAG])); Y.append(y)
    return np.array(X), np.array(Y)

def score(model, recs, tol=int(0.05 * FS_T)):           # a hit = within +-50 ms of the annotation
    tp = fp = fn = 0
    for _, _, x, p in recs:
        d = detect(model, x); hit = 0
        for r in p: hit += np.any(np.abs(d - r) <= tol) if len(d) else 0
        tp += hit; fn += len(p) - hit; fp += max(0, len(d) - hit)
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return dict(precision=float(P), recall=float(R), f1=float(2 * P * R / max(P + R, 1e-9)))

if __name__ == "__main__":
    D = load_all(); rng = np.random.default_rng(SEED); tr, te = [], []
    for k in ("mit", "qt", "cpsc"):
        recs = list(D[k]); rng.shuffle(recs); c = int(0.8 * len(recs)); tr += recs[:c]; te += recs[c:]
    X, Y = fragments(tr); print("train fragments", X.shape)
    idx = np.arange(len(X)); rng.shuffle(idx); X, Y = X[idx], Y[idx]
    m = build()
    print("Training R-peak CNN (Table A1 encoder-decoder)...")
    m.fit(X, Y, validation_split=0.1, epochs=45, batch_size=32, callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.1, verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True, verbose=1)])
    m.save("rpeak_model.keras")
    print("Evaluating model on held-out test records...")
    ho_score = score(m, te)
    print(f"Held-out Test (MIT + QT): {ho_score}")
    print("Evaluating model on independent UoG dataset...")
    uog_score = score(m, D["uog"]) if D["uog"] else "not available"
    print(f"UoG Independent Test: {uog_score}")
    res = {"held_out_mit_qt_cpsc": ho_score, "uog_independent": uog_score}
    json.dump(res, open("rpeak_metrics.json", "w"), indent=1)
    print("Saved rpeak_model.keras and rpeak_metrics.json successfully.")

