# Viva Notes: CardioScan AI

## 1. Denoising
**Q: Why was a 0.5-40 Hz bandpass filter used?**
A: Baseline wander (caused by respiration or electrode movement) is typically below 0.5 Hz. High-frequency noise (like muscle artifacts and 50/60 Hz mains interference) is above 40 Hz. A zero-phase Butterworth filter removes these artifacts without shifting the phase (and thus the timing) of the ECG peaks.

## 2. K-array Knowledge Integration
**Q: What is the K-array and why is it used?**
A: The Knowledge (K) array is a parallel input channel to the R-peak CNN. It acts as a heuristic prior, marking likely peak locations using a simple max-deviation algorithm combined with physiological refractory periods (skipping 100 samples after a peak). This helps the CNN by providing domain knowledge alongside raw signal data.

## 3. Encoder-Decoder R-Peak Architecture
**Q: Why use an encoder-decoder for R-peak detection?**
A: R-peak detection can be framed as a dense prediction (segmentation) task where every sample needs a probability of being a peak. The encoder compresses the signal to find high-level morphological features, while the decoder upsamples it back to the original resolution, outputting a probability for each sample.

## 4. Post-processing
**Q: How do you go from probabilities to discrete R-peaks?**
A: We apply a threshold (0.1) and take the argmax within a small window (70 samples, ~175 ms). Once a peak is found, we skip the rest of the window to avoid multiple detections for a single physiological beat.

## 5. Triad Segmentation
**Q: Why use a 700-sample window for the classifier?**
A: A 700-sample window (at 360 Hz) captures nearly two seconds of signal. This is wide enough to capture the "triad": the beat of interest in the center, plus the previous and next beats. Context from neighboring beats is crucial for identifying premature or compensatory patterns.

## 6. CNN Layers and Class Imbalance
**Q: How do you handle rare classes like Fusion and Others?**
A: We use class weighting during training (via scikit-learn's `compute_class_weight`), which penalizes errors on rare classes more heavily. We also apply light data augmentation (small random noise and temporal shifts) to artificially increase the representation of these rare classes.

## 7. Inter-patient Evaluation
**Q: Why is the DS1/DS2 split important?**
A: A random split of beats would leak data, placing beats from the same patient in both the training and test sets (resulting in falsely high accuracy since the model learns patient-specific morphology). Splitting by record (patient ID) ensures the model generalizes to completely unseen patients.

## 8. Clinical Interpretation (Grad-CAM & Rules)
**Q: How is the model made explainable?**
A: We use a multi-pronged approach:
1. **Grad-CAM**: Highlights which parts of the ECG trace most strongly influenced the CNN's decision.
2. **Rule-Based Features**: We extract known clinical markers (QRS width, RR interval timing, compensatory pauses).
3. **LLM Agent**: Synthesizes the model's prediction with the extracted features to generate a cohesive explanation for a clinician, explicitly stating supporting and conflicting evidence.
