
# Resting-State EEG, Parkinson's Disease & Cognition

Machine learning analysis of resting-state EEG to investigate cognitive impairment in Parkinson's disease (PD), using three complementary analytical approaches: unsupervised clustering, supervised classification, and deep learning on spectrograms.

The corresponding Jupyter Notebook (`.ipynb`) files are provided for each analysis. The project's final report and full description are submitted as a separate PDF. Additional explanations and comments can also be found within the notebooks themselves.

**Research Question:**
Do resting-state EEG-derived spectral features contain meaningful information related to cognitive impairment and Parkinson's disease, and how are these relationships structured?

This project evaluates both:
- Whether EEG spectral features can *classify* diagnosis and cognitive status
- Whether EEG feature space exhibits *intrinsic latent structure* that aligns with clinical phenotypes

---

## Analytical Approaches

### Route A: Supervised Learning: Broad Feature Exploration (`supervised_.py`)

**EEG spectral features → PD diagnosis / cognitive status**

- Multiple feature representations (absolute power, relative power, log power, aperiodic-corrected)
- Random Forest and SVM classifiers
- Tasks: PD vs. Control, binary cognitive impairment, ternary cognitive staging
- Feature importance analysis (SHAP)
- Shuffling-based validity controls

### Route B: Supervised Learning: LEAPD Feature Set (`supB.py`)

**EEG LEAPD features → PD diagnosis / cognitive status**

- Uses the same biologically motivated low-frequency features (delta, theta, low-alpha + TAR ratios) as the unsupervised pipeline.
- Random Forest and SVM classifiers
- Tasks: PD vs. Control, binary cognitive impairment
- Permutation testing for validity

### Route C: Unsupervised Learning: Latent Structure Discovery (`unsupervised.py`)

**EEG feature space → Latent geometry → Clinical alignment**

- Feature engineering: full-spectrum PSD, LEAPD bands (0.5–12 Hz), sham control bands
- Dimensionality reduction: PCA, t-SNE, UMAP
- Clustering: Hierarchical, KMeans, GMM, DBSCAN
- Cluster validation: Silhouette score, Hopkins statistic
- Clinical alignment: NMI, ARI, permutation testing

### Route D: Deep Learning — CNN on EEG Spectrograms (`cnn.py`)

**Raw EEG → Spectrograms → Cognitive status classification**

- Band-pass filtering (0.5–12 Hz) preserving cognitive-relevant frequencies
- 4-second windowing (~8,000 samples from ~149 subjects)
- 3-layer CNN on spectrograms (8 electrodes × time × frequency)
- Batch normalization, Dropout, Adaptive Max Pooling
- Gradient-based interpretability (Captum)

---

## Dataset

The dataset is the OpenNeuro `ds004584` release (Anjum et al., 2024): resting-state EEG from Parkinson's disease patients and healthy controls, with NIH Toolbox and MoCA cognitive scores.

- ~149 subjects after filtering for complete cognitive scores and valid EEG files
- ~2 minutes of resting-state EEG per subject
- Cognitive measures: MoCA, NIH Toolbox (FICAT, DCCST, PCPST, PSMT, PVT)
- Clinical and cognitive metadata: derived from dataset variables and included in this repo as `participants_clin_cog.csv`

**Download the dataset:**
- OpenNeuro: https://openneuro.org/datasets/ds004584/versions/1.0.0
- Direct ZIP (recommended): https://nemar.org/dataexplorer/detail?dataset_id=ds004584

---

## Project File Structure
Each analysis exists as both a `.py` script (for running via `main.py`) and a `.ipynb` notebook (the original development environment in Google Colab, kept for transparency and ease of exploration).
```
.
├── main.py                    # Runs all four pipelines sequentially
├── utils.py                   # Shared data loading, feature construction, preprocessing
├── requirements.txt           # Python dependencies with version pins
├── supervised_.py             # Route A: supervised ML with broad feature exploration
├── supB.py                    # Route B: supervised ML using LEAPD features (matches unsupervised)
├── unsupervised.py            # Route C: clustering and latent structure analysis
├── cnn.py                     # Route D: CNN on EEG spectrograms
├── participants_clin_cog.csv  # Clinical and cognitive scores derived from dataset metadata
├── notebooks/                 # Jupyter notebooks (same analyses)
│   ├── supervised_.ipynb
│   ├── supB.ipynb
│   ├── unsupervised.ipynb
│   └── cnn.ipynb
└── README.md
```

### Expected data structure after first run
```
data
├── sub-001/                    # Individual subject folders
│   └── eeg/
│       └── sub-001_task-Rest_eeg.set
│       └── sub-001_task-Rest_eeg.fdt
│       └── sub-001_task-Rest_events.tsv
│       └── sub-001_task-Rest_channels.tsv
│       └── sub-001_task-Rest_coordsystem.json
│       └── sub-001_task-Rest_eeg.json
│       └── sub-001_task-Rest_electrodes.tsv
├── sub-002/
│   └── eeg/
│       
└── ...                         # Remaining subject folders
```

---

## Setup & Running

**Requirements:** 
```bash
git clone https://github.com/AnnaGanChen/ML-final-project---Group-5-Anna-Inbal-Luna-Petra
cd ML-final-project---Group-5-Anna-Inbal-Luna-Petra

pip install -r requirements.txt

python main.py
```

The dataset will be downloaded automatically on first run. Subsequent runs use the cached local copy.
The notebooks can also be run independently in Jupyter or Google Colab. When run in Colab, the notebooks expect the dataset to be available in Google Drive.

---

## Authors

Anna Gan Chen, Inbal Israeli Gaffa, Luna Gutierrez, Petra Reuwsaat Paul — NeuroData MSc, Group 5

---

## Citations

**Dataset:**
Anjum, M. F., Espinoza, A. I., Cole, R. C., Singh, A., May, P., Uc. E. Y., Dasgupta, S., & Narayanan, N. S. (2024). Resting-state EEG measures cognitive impairment in Parkinson's disease. *npj Parkinson's Disease*, 10(1). https://doi.org/10.1038/s41531-023-00602-0

**LEAPD feature methodology:**
Anjum MF, Dasgupta S, Mudumbai R, et al. Linear predictive coding distinguishes spectral EEG features of Parkinson's disease. Parkinsonism Relat Disord. 2020 Oct;79:79-85. doi: 10.1016/j.parkreldis.2020.08.001. Epub 2020 Aug 23. PMID: 32891924; PMCID: PMC7900258.
Anjum MF, Espinoza AI, Cole RC, et al. Resting-state EEG measures cognitive impairment in Parkinson's disease. NPJ Parkinsons Dis. 2024 Jan 3;10(1):6. doi: 10.1038/s41531-023-00602-0. PMID: 38172519; PMCID: PMC10764756.
Spoa M, Monti S, Bjekić J, Guerra A, et al. Resting-state EEG changes associated with cognitive decline in Parkinson's disease: a systematic review. J Neural Transm (Vienna). 2025 Nov 18. doi: 10.1007/s00702-025-03065-0. Epub ahead of print. PMID: 41251764.
