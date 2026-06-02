# 🌌 Cosmic Explosion Exploration

> *Listening to the universe's most violent events — one gravitational wave at a time.*

This is an exploratory, modular, data science pipeline, for analysing gravitational wave signals from the [LIGO Gravitational Wave Observatory](https://www.ligo.caltech.edu/), which I put together for for educational purposes. The project is built in Python utilising open public data from Gravitational Wave Open Science Center (GWOSC) sourced using the gwpy software package. This project walks through every stage of a real scientific analysis — from raw strain data to matched filtering and parameter estimation — and tries to provide explanations of the physics at every step.



**Primary dataset:** GW150914 — this was the first gravitational wave ever detected, recorded on 14 September 2015.

---

## What Is a Gravitational Wave?

In 1916, Albert Einstein predicted that accelerating masses should disturb the fabric of spacetime itself, sending ripples outward at the speed of light — **gravitational waves**. They are the result of the most violent collisions in the universe produced when massive objects accelerate, especially in extreme events like merging black holes, colliding neutron stars, or some supernovae giving **cosmic-explosions**!.

For nearly a century, no instrument was sensitive enough to detect them. Then, on 14 September 2015, two black holes that had been spiralling toward each other for over a billion years finally collided — and the resulting ripple, 1.3 billion years later, stretched and squeezed LIGO's 4 km detector arms by *less than one-thousandth the width of a proton*.

LIGO heard it anyway.

That moment — designated **GW150914** — confirmed the last great unverified prediction of general relativity, proved that stellar-mass binary black holes exist and merge, and launched an entirely new field of astronomy. Today, LIGO and its partner observatories have detected over 200 such events. We are no longer limited to observing the universe through light. We can *listen* to it.

### Why Is This A Useful Subject For Data Science Demonstration?

Gravitational wave astronomy is one of the most data-intensive fields in modern science. The LIGO detectors produce 4,096 samples per second per channel, across hundreds of channels, continuously. The signal — when it arrives — is buried under noise that is *ten million times larger* than it. Finding the actual signal requires:

- Advanced **signal processing** (whitening, bandpass filtering, notch filtering)
- **Time-frequency analysis** (Q-transforms, spectrograms)
- **Matched filtering** — the optimal linear detector for a known signal in coloured noise
- **Statistical inference** — distinguishing genuine detections from noise fluctuations
- **Feature engineering** — extracting physically meaningful quantities from raw waveforms

This is not a sample dataset. These are the actual techniques used by the LIGO Scientific Collaboration in their published papers. The data is real. The physics is real. The pipeline mirrors — in simplified but faithful fashion — what it takes to detect a collision between two black holes.

---

## Planned Project Overview

```
Cosmic Explosion Explorer
│
├── A six-notebook sequential pipeline
├── Two shared support python modules (config.py, utils.py)
├── Primary event: GW150914 (Binary Black Hole merger)
├── Detectors: LIGO Hanford (H1) + LIGO Livingston (L1)
└── Data source: GWOSC (Gravitational Wave Open Science Center)
```

The pipeline is designed to be **read as much as run**. Every cell explains not just *what* the code does but *why* — with some summary insight into the signal processing theory, some background physics, and details of the practical choices made at each step. It is hopefully both a working analysis and a reference learning resource.

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
        │  Time-domain features (peak amplitude, kurtosis, SNR)
        │  Frequency-domain features (spectral centroid, entropy)
        │  Hilbert transform → instantaneous frequency & amplitude envelope
        │  Chirp mass estimation from the frequency sweep rate
        │  Q-transform energy features
        ▼
05_modelling
        │  Noise PSD estimation (Welch's method, off-source window)
        │  Post-Newtonian waveform template (leading-order inspiral)
        │  Matched filter (frequency-domain cross-correlation)
        │  SNR time-series and merger time recovery
        │  Chirp mass scan (SNR vs template mass — baby parameter estimation)
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
| **Statistical methods** | Matched filtering, SNR estimation, kurtosis, spectral entropy |
| **Feature engineering** | Physics-motivated feature extraction, chirp mass estimation |
| **Scientific computing** | NumPy, SciPy, gwpy, h5py, Pandas, Matplotlib |
| **Software engineering** | Modular pipeline design, centralised config, reusable utility functions |
| **Physics** | General relativity, gravitational wave generation, post-Newtonian approximation |

---

## Repository Structure

```
cosmic-explosion-explorer/
│
├── README.md
├── requirements.txt
├── ligo.yml              ← Anaconda environment
│
├── config.py                   ← All constants: paths, event params, signal params
├── utils.py                    ← Reusable functions: I/O, plotting, signal utilities
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
└── data/                       ← Not committed — regenerated by notebook 01
    ├── raw/                    ← HDF5 from GWOSC (gitignored)
    ├── processed/              ← Cleaned strain (gitignored)
    ├── features/               ← Feature matrix CSVs (gitignored)
    ├── models/                 ← Model results (gitignored)
    └── visualisations/         ← Generated plots (gitignored)
```

> **Note on data files:** Raw HDF5 files are not saved to this repository — they are large and freely regenerable. Running `01_data_acquisition.ipynb` will get everything automatically from the GWOSC portal.

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

Open notebooks in order and run all cells. Each notebook saves its outputs as HDF5 or CSV files that the next notebook reads. The full pipeline from raw data to results takes approximately 15–20 minutes on a standard laptop to execute, with the Q-transform computation steps being the most time consuming (~20–30 seconds per selected detector).

```
01 → 02 → 03 → 04 → 05 → 06
```

---

## Key Results

### The Chirp — GW150914 in the Q-transform

The Q-transform is the standard visualisation tool for viewing gravitational wave signals. It reveals the characteristic **chirp** of the gravitational wave signal (in this case setup binary black hole inspiral) — a bright arc sweeping from ~40 Hz to ~150 Hz in the final 0.4 seconds before the two black holes merged, 1.3 billion years ago.
NOTE: If we were investigating other event types then we would set our config file differently (for example if we were looking at a Binary Neutron Star (BSN) event the segment duration would be much longer and the bandpass filters would start lower and end on a higher frequency)

> *[Q-transform figure — generated by notebook 03 and saved to `figures/`]*

### What the Pipeline Recovers

| Quantity | This pipeline | Published (Abbott et al. 2016) |
|---|---|---|
| Merger time | Within ~0.01 s of t=0 | GPS 1126259462.4 |
| Peak GW frequency | ~140–160 Hz | ~150 Hz |
| Chirp mass estimate | ~25–32 M☉ | 28.3 ± 1.5 M☉ |
| H1 detection significance | Above noise threshold | SNR = 18.2 |
| H1 / L1 consistency | Confirmed | 7 ms arrival delay |

---

## Technical Notes

**Why `unit=""` not `unit="dimensionless"`**
gwpy's `TimeSeries` constructor requires `unit=""` for dimensionless strain data. Passing `unit="dimensionless"` is an invalid astropy unit string that causes silent unrecognised unit failures in signal processing methods.
Gravitational wave strain *h(t)* is genuinely dimensionless as it's a fracitonal change in lenght ΔL/L. So  unit="" is both technically correct and the format that gwpy (based on astropy) accepts cleanly.

**Why `method="median"` for Welch PSD**
LIGO detectors are not sitting in a quiet laboratory. They are in constant contact with the outside physical world — seismic disturbances, nearby road traffic, electronics noise, laser fluctuations, and hundreds of other sources of transient noise called glitches are measured. Median averaging is robust to occasional loud noise transients (glitches) that would have otherwise been inflated if a mean-based estimate were utilised. 
gwpy applies a 0.9635 bias correction to maintain an unbiased PSD estimate.

**Why whitening over alternatives**
Whitening flattens the noise floor — frequencies where the detector is noisy get suppressed, frequencies where it's quiet get amplified — so that no single band dominates. This is the standard approach in LIGO analysis, preferred over fixed-reference normalisation, bandpass-only filtering, or adaptive methods for its robustness and interpretability.

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

In the final 0.2 seconds before merger, the two black holes were orbiting each other at roughly **half the speed of light**, completing 30 orbits per second. The power radiated as gravitational waves at peak emission exceeded the combined light output of every star in the observable universe.

---

## Data Source

All strain data is sourced from the **Gravitational Wave Open Science Center (GWOSC)**:
- Portal: [gwosc.org](https://gwosc.org)
- Python access via `gwpy`: `TimeSeries.fetch_open_data('H1', ...)`
- License: Creative Commons Attribution 4.0

**Primary reference:**
Abbott et al. (2016), *Observation of Gravitational Waves from a Binary Black Hole Merger*, Physical Review Letters 116, 061102. [DOI: 10.1103/PhysRevLett.116.061102](https://doi.org/10.1103/PhysRevLett.116.061102)

---

## Planned Extensions which may be explored in future!

- [ ] **GW170817** — the binary neutron star merger; multi-messenger event with optical counterpart
- [ ] **Audio chirp** — pitch-shifted strain as a `.wav` file
- [ ] **Template bank** — extend the matched filter across a grid of (m₁, m₂) pairs
- [ ] **GraceDB alert monitor** — subscribe to real-time candidate alerts for O5

---

## About This Project

Built as a portfolio project exploring the intersection of astrophysics and data science. The aim is to work through a complete, real-world scientific analysis pipeline — not with simplified synthetic data, but with the actual signals first detected that changed our understanding of the observable universe.

Developed using Python, Jupyter Notebooks, and the gwpy / scientific Python ecosystem.

---

*"The universe has been orchestrating this symphony for billions of years and we have only just learned to listen."*
— Janna Levin — Black Hole Blues, 2017
