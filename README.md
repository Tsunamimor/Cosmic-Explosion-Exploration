# 🌌 Cosmic Explosion Exploration

> *Listening to the universe's most violent events — one gravitational wave at a time.*

This is an exploratory, modular data science pipeline for analysing gravitational wave signals from the [LIGO Gravitational Wave Observatory](https://www.ligo.caltech.edu/), built for educational purposes. The project is written in Python using open public data from the Gravitational Wave Open Science Center (GWOSC), sourced via the `gwpy` software package. It walks through every stage of a real scientific analysis — from raw strain data to matched filtering and parameter estimation — and explains the physics at every step.

**Primary dataset:** GW150914 — the first gravitational wave ever detected, recorded on 14 September 2015.

📖 **See also:** [`COMPANION_GUIDE.md`](COMPANION_GUIDE.md) — an in-depth walk-through of the science behind every stage.

---

## What Is a Gravitational Wave?

In 1916, Albert Einstein predicted that accelerating masses should disturb the fabric of spacetime itself, sending ripples outward at the speed of light — **gravitational waves**. They are produced by the most violent events in the universe: the merging of black holes, colliding of neutron stars, and some supernovae.

For nearly a century, no instrument was sensitive enough to detect them. Then, on 14 September 2015, two black holes that had been spiralling toward each other for over a billion years finally collided — and the resulting ripple was detected 1.3 billion years later. It stretched and squeezed LIGO's 4km interferometer detector arms by *less than one-thousandth the width of a proton*.

LIGO heard it anyway.

That moment — designated **GW150914** — confirmed the last great unverified prediction of general relativity, it also proved that stellar-mass binary black holes exist and merge, and it launched an completely new field of astronomy. Today, LIGO and its partner observatories have detected over 200 such cosmic events. We are no longer limited to observing the universe through light. We can *listen* to the vibrations that remain from these massive collisions.

### Why Is This a Useful Subject for a Data Science Demonstration?

Gravitational wave astronomy is one of the most data-intensive fields in modern science. The LIGO detectors continuously produce 4,096 samples per second per channel, across hundreds of channels. The signal — when it arrives — is buried under noise that is *many orders of magnitude larger* than it. Extracting the actual signal from this noise requires a lot of processing such as:

- Advanced **signal processing** (whitening, bandpass filtering, notch filtering)
- **Time-frequency analysis** (Q-transforms, spectrograms, the Hilbert transform)
- **Matched filtering** — the optimal linear detector for a known signal in coloured Gaussian noise
- **Statistical inference** — distinguishing genuine detections from noise fluctuations
- **Feature engineering** — extracting physically meaningful quantities from raw waveforms

This is not a toy dataset. These are the actual techniques used by the LIGO Scientific Collaboration in their published papers. The data is real collected from the LIGO detectors. The physics is real based on predictions from General Relativity. The pipeline mirrors — in a simplified but faithful manner — what it takes to detect a collision between two black holes.

---

## Project Overview

```
Cosmic Explosion Explorer
│
├── A six-notebook sequential pipeline
├── Three shared support modules (config.py, utils.py, pipeline_banner.py)
├── Primary event: GW150914 (binary black hole merger)
├── Detectors: LIGO Hanford (H1) + LIGO Livingston (L1)
└── Data source: GWOSC (Gravitational Wave Open Science Center)
```

This pipeline is designed to be **read as much as run**. Every cell explains not just *what* the code does but *why* — with the signal-processing theory, the background physics, and the practical reasoning behind each choice. It is intended to be both a working analysis and a reference learning resource.

A consistent **deep-space visual theme** (navy/teal) runs across all six notebooks. The theme, colour palette, and matplotlib styling are centralised in `config.THEME` and applied with a single `apply_theme()` call, so the whole pipeline can be restyled from one place.

---

## Pipeline Structure

```
01_data_acquisition
        │  Fetch raw strain data from GWOSC via gwpy
        │  Explore the TimeSeries data model
        │  Visualise the noise landscape (ASD)
        ▼
02_preprocessing
        │  Whiten the strain (flatten the noise floor)
        │  Bandpass filter (35–350 Hz signal band)
        │  Notch filter (60/120/180 Hz mains harmonics)
        ▼
03_visualisation
        │  Time-series: raw vs processed comparison
        │  STFT spectrogram (fixed-window frequency analysis)
        │  Q-transform (the gold standard for GW chirp visualisation)
        │  Detector correlation (H1 vs L1 consistency check)
        ▼
04_feature_extraction
        │  Time-domain features (peak amplitude, kurtosis, skewness, SNR)
        │  Frequency-domain features (spectral centroid, bandwidth, entropy)
        │  Hilbert transform → instantaneous frequency & amplitude envelope
        │  Chirp mass estimation from the frequency sweep rate
        │  Q-transform energy features + radar chart
        ▼
05_modelling
        │  Noise PSD estimation (Welch's method, off-source window)
        │  Post-Newtonian waveform template (leading-order inspiral)
        │  Matched filter (frequency-domain cross-correlation)
        │  SNR time-series and merger time recovery
        │  Chirp mass scan (SNR vs template mass — basic parameter estimation)
        ▼
06_results
        │  Pipeline vs published comparison (Abbott et al. 2016)
        │  Four-panel summary figure
        │  Full provenance record (JSON)
        └  Retrospective: what the pipeline recovered and what it didn't
```

---

## Skills Demonstrated

| Domain | Techniques |
|---|---|
| **Signal processing** | Whitening, bandpass/notch filtering, Welch PSD, Butterworth filters |
| **Time-frequency analysis** | STFT spectrogram, Q-transform, Hilbert transform, instantaneous frequency |
| **Statistical methods** | Matched filtering, SNR estimation, kurtosis, skewness, spectral entropy |
| **Feature engineering** | Physics-motivated feature extraction, chirp mass estimation |
| **Scientific computing** | NumPy, SciPy, gwpy, h5py, Pandas, Matplotlib |
| **Software engineering** | Modular pipeline design, centralised config & theme, reusable utility functions |
| **Physics** | General relativity, gravitational wave generation, post-Newtonian approximation |

---

## Repository Structure

```
cosmic-explosion-explorer/
│
├── README.md
├── COMPANION_GUIDE.md           ← In-depth science companion + glossary + references
├── requirements.txt
├── environment.yml              ← Anaconda environment
│
├── config.py                   ← All constants: paths, event/signal/viz params, theme
├── utils.py                    ← Reusable functions: I/O, plotting, theme, signal utilities
├── pipeline_banner.py          ← SVG pipeline-stage banner embedded in each notebook
│
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_visualisation.ipynb
│   ├── 04_feature_extraction.ipynb
│   ├── 05_modelling.ipynb
│   └── 06_results.ipynb
│
├── figures/                    ← Key output plots (committed as static PNGs)
│   ├── GW150914_qtransform_H1.png
│   └── GW150914_overview_raw_vs_processed.png
│
└── data/                       ← Not committed — regenerated by the notebooks
    ├── raw/                    ← HDF5 from GWOSC (gitignored)
    ├── processed/              ← Cleaned strain (gitignored)
    ├── visualisations/         ← Generated plots (gitignored)
    ├── features/               ← Feature matrix CSVs (gitignored)
    ├── models/                 ← Matched-filter results JSON (gitignored)
    └── results/                ← Final summary + provenance JSON (gitignored)
```

> **Note on data files:** Raw HDF5 files are not saved to this repository — they are large and freely regenerable. Running `01_data_acquisition.ipynb` fetches everything automatically from the GWOSC portal.

---

## Getting Started

### Prerequisites

- Python 3.10+
- Anaconda or Miniconda (recommended)
- ~500 MB disk space for data files

### Installation

**Option 1 — Conda environment (recommended):**
```bash
git clone https://github.com/Tsunamimor/cosmic-explosion-explorer.git
cd cosmic-explosion-explorer
conda env create -f environment.yml
conda activate cosmic-gw
jupyter notebook
```

**Option 2 — pip:**
```bash
git clone https://github.com/Tsunamimor/cosmic-explosion-explorer.git
cd cosmic-explosion-explorer
pip install -r requirements.txt
jupyter notebook
```

### Running the Pipeline

Open the notebooks in order and run all cells. Each notebook saves its outputs as HDF5, CSV, or JSON files that the next notebook reads. The full pipeline from raw data to results takes roughly 15–20 minutes on a standard laptop, with the Q-transform steps being the most time-consuming (~20–30 seconds per detector).

```
01 → 02 → 03 → 04 → 05 → 06
```

> **Tip:** Because each notebook reads the previous notebook's saved outputs, re-running an earlier notebook means you should re-run the later ones too, so downstream results reflect your latest changes.

---

## Key Results

### The Chirp — GW150914 in the Q-transform

The Q-transform is the standard tool for visualising gravitational wave signals. It reveals the characteristic **chirp** of a binary black hole inspiral — a bright arc sweeping from ~40 Hz to ~150 Hz in the final 0.4 seconds before the two black holes merged, 1.3 billion years ago.

> **Note:** For other event types the `config.py` parameters would differ. For a binary neutron star event such as GW170817, the segment duration would be much longer (the inspiral lasts tens of seconds in band) and the bandpass would extend to higher frequencies.

> *[Q-transform figure — generated by notebook 03 and saved to `figures/`]*

### What the Pipeline Recovers

| Quantity | This pipeline | Published (Abbott et al. 2016) |
|---|---|---|
| Merger time | Within ~0.05 s of t = 0 | GPS 1126259462.4 |
| Peak GW frequency | ~140–160 Hz | ~150 Hz |
| Chirp mass (matched-filter scan) | ~25–37 M☉ | 28.3 ± 1.5 M☉ |
| H1 matched-filter SNR | ~24 (0PN template) | 18.2 (full NR template) |
| L1 matched-filter SNR | ~20 (0PN template) | 13.8 (full NR template) |
| H1 / L1 consistency | Confirmed (~10 ms) | 6.9 ms arrival delay |

> **On the SNR values:** The matched-filter SNR captured in this pipeline uses a leading-order (0PN) post-Newtonian template and a noise PSD estimated from a short off-source window. This is in contrast to the actual published values which use full numerical-relativity waveforms (covering merger and ringdown) against a carefully calibrated PSD. The difference between our results and LIGO's illustrates how much engineering actually goes into the production analysis and not a flaw in the physics extracted. See `COMPANION_GUIDE.md` for more background details.

> **On chirp mass:** The Hilbert-based instantaneous-frequency estimate in notebook 04 is unreliable for chirp mass because the pre-merger signal is buried in noise; that estimate is honestly reported as `nan`. The matched-filter scan in notebook 05 is the trustworthy route and lands in the right ballpark.

---

## Technical Notes

**Why `unit=""` not `unit="dimensionless"`**
gwpy's `TimeSeries` constructor requires `unit=""` for dimensionless strain data. Passing `unit="dimensionless"` is an invalid astropy unit string that causes silent unrecognised-unit failures in downstream signal-processing methods. Gravitational wave strain *h(t)* is genuinely dimensionless — it is a fractional change in length, ΔL/L — so `unit=""` is both technically correct and the format that gwpy (built on astropy) accepts cleanly.

**Why `method="median"` for Welch PSD**
LIGO detectors are not sitting in a quiet laboratory. They are in constant contact with the outside world — seismic disturbances, road traffic, electronics noise, laser fluctuations, and hundreds of other transient noise sources (glitches). Median averaging is robust to occasional loud glitches that would inflate a mean-based estimate. gwpy applies a 0.9635 bias correction to keep the median-based PSD unbiased.

**Why whitening over alternatives**
Whitening flattens the noise floor — frequencies where the detector is noisy get suppressed, frequencies where it is quiet get amplified — so no single band dominates. This is the standard approach in LIGO analysis, preferred over fixed-reference normalisation, bandpass-only filtering, or adaptive methods for its robustness and interpretability.

**Why `DejaVu Sans` for plots**
The pipeline sets `font.family = "DejaVu Sans"` in the shared theme. DejaVu Sans includes the ☉ (solar mass) and √ glyphs that Arial lacks, avoiding "missing glyph" warnings on Windows (which was experienced early on in the Notebook creation!).

**Centralised theme and configuration**
All styling lives in `config.THEME` (deep-space colour tokens + Wes Anderson palette + matplotlib rcParams), applied via `utils.apply_theme()`. All paths, event parameters, signal parameters, physical constants, and per-event phase markers live in `config.py`. No magic numbers are embedded in the notebooks.

---

## The Event: GW150914

| Parameter | Value |
|---|---|
| Detection date | 14 September 2015, 09:50:45 UTC |
| Primary black hole | ~36 M☉ |
| Secondary black hole | ~29 M☉ |
| Final black hole | ~62 M☉ |
| Energy radiated | ~3 M☉c² as gravitational waves |
| Luminosity distance | ~410 Mpc (~1.3 billion light-years) |
| Network SNR | 24.4 |
| False alarm rate | < 1 per 203,000 years |

In the final 0.2 seconds before merger, the two black holes were orbiting each other at roughly **half the speed of light**, completing dozens of orbits per second. The power radiated as gravitational waves at peak emission briefly exceeded the combined light output of every star in the observable universe.

---

## Data Source

All strain data is sourced from the **Gravitational Wave Open Science Center (GWOSC)**:
- Portal: [gwosc.org](https://gwosc.org)
- Python access via `gwpy`: `TimeSeries.fetch_open_data('H1', ...)`
- Licence: Creative Commons Attribution 4.0

**Primary references:**
- Abbott et al. (2016), *Observation of Gravitational Waves from a Binary Black Hole Merger*, Phys. Rev. Lett. **116**, 061102. [DOI](https://doi.org/10.1103/PhysRevLett.116.061102) · [arXiv:1602.03837](https://arxiv.org/abs/1602.03837)
- Abbott et al. (2016), *Properties of the Binary Black Hole Merger GW150914*, Phys. Rev. Lett. **116**, 241102. [DOI](https://doi.org/10.1103/PhysRevLett.116.241102) · [arXiv:1602.03840](https://arxiv.org/abs/1602.03840)

---

## Potential Extensions

- [ ] **GW170817** — the binary neutron star merger
- [ ] **Audio chirp** — pitch-shifted strain as a `.wav` file
- [ ] **Template bank** — Implement a grid or MCMC search for joint (m₁, m₂) estimation
- [ ] **Higher-order PN** — add higher-order PN corrections to the waveform template (1.5PN, 2PN, 2.5PN) to further improve accuracy
- [ ] **LALSuite templates** — Use PyCBC or LALSuite to create a more accurate matched filter to capture merger and ringdown
- [ ] **Bilby** — Use Bilby for proper Bayesian parameter estimation — what LIGO actually does

---

## About This Project

Built as a portfolio project exploring the intersection of astrophysics and data science. The aim is to work through a complete, real-world scientific data analysis pipeline — not with simplified synthetic data, but with the actual signals that changed our understanding of the observable universe.

Developed using Python, Jupyter Notebooks, and the gwpy / scientific Python ecosystem.

---

*"The universe has been orchestrating this symphony for billions of years and we have only just learned to listen."*
— Janna Levin, *Black Hole Blues*, 2017
