"""Task 3: clinical features + rule checks per class + optional LLM 'AI agent' summary."""
import os, numpy as np
from .dsp import FS

def features(x, peaks, i):
    r = peaks[i]; med = np.median(np.diff(peaks)) / FS if len(peaks) > 2 else None
    rr_p = (r - peaks[i-1]) / FS if i > 0 else None
    rr_f = (peaks[i+1] - r) / FS if i < len(peaks) - 1 else None
    seg = x[max(0, r-45):r+45]; base = np.median(x[max(0, r-200):r+200])
    amp = abs(x[r] - base); w = np.abs(seg - base) > 0.5 * amp
    qrs_ms = 1000 * w.sum() / FS                                      # width at half height
    comp = (rr_p + rr_f) / med if None not in (rr_p, rr_f, med) else None
    return dict(rr_prev_s=rr_p, rr_next_s=rr_f, median_rr_s=med, qrs_width_ms=round(qrs_ms),
                premature=bool(rr_p and med and rr_p < 0.8 * med),
                compensatory_pause=bool(comp and abs(comp - 2) < 0.3),
                wide_qrs=bool(qrs_ms > 120))

# what a doctor expects for each class (section 3.3.3 of the paper)
RULES = {"Normal": {"wide_qrs": False, "premature": False},
         "PVC": {"wide_qrs": True, "premature": True, "compensatory_pause": True},
         "LBBB": {"wide_qrs": True}, "RBBB": {"wide_qrs": True},
         "Atrial premature": {"premature": True, "wide_qrs": False},
         "Fusion (V+N)": {"wide_qrs": True, "compensatory_pause": False}}
LABEL = {"wide_qrs": "wide QRS (>120 ms)", "premature": "premature beat (short RR)",
         "compensatory_pause": "compensatory pause"}

def check(cls, f):
    sup, con = [], []
    for k, want in RULES.get(cls, {}).items():
        (sup if f[k] == want else con).append(("" if want else "no ") + LABEL[k])
    return sup, con

def agent_summary(cls, prob, f, sup, con):
    """LLM agent: explains model output + measured features. Falls back to a template."""
    text = (f"Beat classified as {cls} ({prob:.0%}). Supporting findings: {', '.join(sup) or 'none'}. "
            f"Conflicting findings: {', '.join(con) or 'none'}. Needs clinician confirmation.")
    if not os.getenv("ANTHROPIC_API_KEY"): return text
    try:
        import anthropic
        r = anthropic.Anthropic().messages.create(model="claude-3-5-sonnet-20241022", max_tokens=200, messages=[{
            "role": "user", "content": f"You assist cardiologists. In 3 sentences explain this ECG beat result, "
            f"say if measured features agree with the CNN, and add that a clinician must confirm. Do NOT make a diagnosis.\n"
            f"Class={cls}, confidence={prob:.2f}, features={f}, supports={sup}, conflicts={con}"}])
        return r.content[0].text
    except Exception:
        return text


def grad_cam(model, seg, cls_idx):
    """Generate 1D Grad-CAM for the given segment and target class."""
    import tensorflow as tf
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv1D):
            last_conv_layer = layer
            break
    if not last_conv_layer: return [0.0] * len(seg)
    
    with tf.GradientTape() as tape:
        x = tf.cast(seg[None, ...], tf.float32)
        conv_outputs = None
        for layer in model.layers:
            x = layer(x)
            if layer is last_conv_layer:
                conv_outputs = x
                tape.watch(conv_outputs)
        loss = x[:, cls_idx]
        
    grads = tape.gradient(loss, conv_outputs)
    if grads is None or conv_outputs is None:
        return [0.0] * len(seg)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    
    heatmap = heatmap.numpy()
    from scipy.signal import resample
    heatmap = resample(heatmap, len(seg))
    heatmap = np.clip(heatmap, 0, 1)
    return [float(v) for v in heatmap.tolist()]
