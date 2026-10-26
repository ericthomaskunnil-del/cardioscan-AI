# CardioScan AI

An explainable ECG arrhythmia detection system based on the pipeline from "Towards Transparent AI in Medicine" (Kovalchuk et al., Technologies 2025).

## Pipeline Stages

1. **Denoising**: Band-pass filtering (0.5-40 Hz) using a zero-phase Butterworth filter to remove baseline wander and high-frequency noise.
2. **R-Peak Detection (Task 1)**: An encoder-decoder 1D CNN with a 'knowledge' (K) array input to identify R-peaks across multiple datasets.
3. **Heartbeat Segmentation**: Triad extraction to capture morphological structure around each R-peak.
4. **CNN Classification (Task 2)**: A 9-class deep convolutional network for classifying heartbeat segments (Normal, PVC, LBBB, RBBB, etc.).
5. **Clinical Interpretation (Task 3)**: Extraction of interpretable rules (QRS width, RR intervals), generation of a Grad-CAM saliency map, and a final LLM-based agent summary.

## Dataset Sources

- **MIT-BIH Arrhythmia Database**: Downloaded via PhysioNet (`wfdb`). Used for training the classifier and evaluating R-peaks.
- **QT Database**: Downloaded via PhysioNet. Used for R-peak model training.
- **CPSC-2020**: **MANUAL DOWNLOAD REQUIRED**. Download the TrainingSet from [http://2020.icbeb.org/CSPC2020](http://2020.icbeb.org/CSPC2020) and place the `.mat` files in `data/cpsc2020/TrainingSet/data/` and `data/cpsc2020/TrainingSet/ref/`. Used for R-peak model training.
- **UoG (Glasgow University Database)**: Downloaded via the `ecg_gudb_database` package. Used as an independent test set for R-peak evaluation.

## Run Order

1. \`pip install -r requirements.txt\`
2. \`python data_prep.py\` (Requires the CPSC dataset to be downloaded manually, or it will skip it)
3. \`python train_rpeak.py\` (Trains the encoder-decoder R-peak detector)
4. \`python train.py\` (Trains the 9-class heartbeat classifier on an inter-patient split)
5. \`uvicorn app:app --reload\` (Runs the FastAPI backend and static frontend on \`localhost:8000\`)

## Metrics

**(Note: Values will be populated after a successful run of the scripts)**
- **R-Peak Detection**: See `rpeak_metrics.json`.
- **Classification**: See `metrics.json`.
