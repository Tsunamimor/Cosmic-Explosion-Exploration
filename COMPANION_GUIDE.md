# 🌌 Cosmic Explosion Explorer — Companion Guide

> *The science behind the pipeline — the "why" behind every "what".*

This companion guide captures the physics and signal-processing reasoning behind the **Cosmic Explosion Explorer** pipeline in more depth than the notebooks themselves. Where the notebooks show *what* each step does, this guide explains *why* — the concepts, the trade-offs, and reflects some honest limitations of the used approach.

It is organised to follow the six-notebook pipeline, followed by a **glossary** and **references for further reading**.

**Contents**

1. [The big picture: data-driven vs physics-driven modelling](#1-the-big-picture)
2. [Data acquisition — where the data comes from](#2-data-acquisition)
3. [Preprocessing — making the signal visible](#3-preprocessing)
4. [Visualisation — seeing the chirp](#4-visualisation)
5. [Feature extraction — turning waveforms into numbers](#5-feature-extraction)
6. [Modelling — matched filtering and waveform templates](#6-modelling)
7. [Results — comparing against the ground truth](#7-results)
8. [Glossary](#8-glossary)
9. [References & further reading](#9-references--further-reading)

---

## 1. The Big Picture

### Two philosophies of modelling

Most data science is **data-driven**: you collect many examples, find patterns, and let the data define the model. You don't need to understand the mechanism — only the correlation.

This pipeline is **physics-driven**, which is closer to how physics has always worked. You start from a theory (General Relativity) that makes precise mathematical predictions, then test whether the data conforms to those predictions. The model exists independently of the data — it came from Einstein's field equations, not from observations.

| | Data-driven (typical ML) | Physics-driven (this pipeline) |
|---|---|---|
| Where the model comes from | The data | The theory (GR) |
| Training data required | Yes — many examples | No — one event suffices |
| Interpretability | Often low | Very high — every parameter is physical |
| Generalisation | Within the training distribution | Extrapolation from first principles |

This matters because before GW150914 there were **zero** confirmed gravitational wave detections to train on. You cannot build a data-driven classifier for a signal you have never seen. The theoretical waveform was the only prior knowledge available — and it was enough.

### Why feature extraction at all, in the age of deep learning?

A reasonable question: don't powerful computers and modern AI make hand-crafted features obsolete? Sometimes — in data-rich, compute-rich, interpretability-flexible settings, deep learning does learn its own features. But gravitational wave detection is **data-scarce** (≈90 confirmed events across three observing runs), the signal sits far below the noise, and *interpretability is the goal* — the point is to measure physical quantities (masses, distance) and trust them. Even LIGO's modern deep-learning pipelines are trained on simulated injections generated from the same post-Newtonian physics this project uses. The theory is always upstream.

---

## 2. Data Acquisition

### Where the 131,072 samples come from

The raw strain segment is `4,096 Hz × 32 s = 131,072` samples (= 2¹⁷, convenient for FFTs).

- **4,096 Hz sampling** — by the Nyquist theorem, this faithfully captures signals up to 2,048 Hz, comfortably above the ~350 Hz top of the binary-black-hole signal band. (LIGO's internal pipelines use 16,384 Hz; the public GWOSC release at 4,096 Hz captures everything relevant for compact binary coalescences.)
- **32-second segment** — long enough to estimate the noise floor for whitening and to capture the ringdown, short enough to keep memory and computation manageable. The actual chirp occupies only ~0.4 s (~1,638 samples) — about 1.25% of the array. The rest is noise used for calibration.

### Strain is genuinely dimensionless

Gravitational wave strain *h(t)* is a **fractional change in length**, ΔL/L — the wave stretches and squeezes the 4 km arms by less than one-thousandth the width of a proton. Because it is a ratio of two lengths, it has no units. This is why gwpy requires `unit=""` (not `unit="dimensionless"`, which is an invalid astropy unit string).

---

## 3. Preprocessing

### Is all LIGO noise Gaussian?

No — and this is one of the most practically important complications in the field. The matched filter (notebook 05) assumes stationary Gaussian noise, but real LIGO data is significantly non-Gaussian:

- **Glitches** — short, high-amplitude transients from scattered light, seismic coupling, electronics, and more. They occur far more often than Gaussian statistics predict and have their own taxonomy (blips, koi fish, scattered-light arches, tomtes).
- **Spectral lines** — deterministic, narrow features: 60 Hz from electrical mains and harmonics, violin modes (mirror-suspension resonances), calibration lines.
- **Non-stationarity** — the noise floor drifts as environmental conditions change, so a PSD estimated from one segment may not describe the next.

LIGO handles this with chi-squared tests (checking that signal power is distributed across frequency the way a real template predicts), data-quality vetoes (hundreds of auxiliary sensors), and time-slide background estimation. Your kurtosis feature in notebook 04 is, in effect, an informal test of non-Gaussianity.

### Why whitening, and why median Welch PSD

**Whitening** flattens the noise spectrum so no one single band dominates — frequencies where the detector is noisy get suppressed, quiet frequencies get amplified. It is preferred over fixed-reference normalisation, bandpass-only filtering, wavelet de-noising, adaptive filtering, and ICA for its robustness and standard use in LIGO analysis.

**`method="median"`** for the Welch PSD is chosen because median averaging is robust to occasional glitch-contaminated segments (a mean would be inflated by a single loud glitch). gwpy applies a 0.9635 bias correction so the median-based estimate stays unbiased.

---

## 4. Visualisation

### Why the Q-transform beats a spectrogram

A standard STFT spectrogram uses a **fixed window length** at all frequencies, forcing a single trade-off: a short window gives good time resolution but poor frequency resolution, and vice versa. For a chirp sweeping from 35 Hz to 150 Hz in ~0.2 s, neither extreme is ideal.

The **Q-transform** uses a **frequency-dependent window** whose width scales as Q/f — wide at low frequency (good frequency resolution where the chirp evolves slowly), narrow at high frequency (good time resolution where it sweeps fast). This is exactly matched to the physics of an inspiral, which is why it is the gold standard for visualising the chirp arc.

### The two-detector consistency check

A real gravitational wave passes through both detectors at the speed of light, so it must arrive within the ~10 ms light-travel time between Hanford and Livingston. GW150914 arrived at H1 ~7 ms before L1. A glitch is local and would not appear coherently at both sites at the right delay. The detector-correlation plot (notebook 03) applies the per-event time shift and sign flip (detectors have opposing arm orientations) and overlays the two waveforms — if they align, the signal is almost certainly astrophysical.

---

## 5. Feature Extraction

The notebooks extract four groups of features. Each captures a different, physically meaningful aspect of the signal.

### Time-domain features

- **RMS** — the natural measure of signal *power*. Strain oscillates around zero, so a plain mean would cancel to nothing; squaring (then rooting) makes excursions count. The simple SNR is a ratio of signal-window RMS to noise-window RMS. (Same idea as a household mains-voltage rating: the quoted 230 V / 120 V is an RMS.)
- **Kurtosis** — the "tailedness" of the amplitude distribution. Gaussian noise has Fisher kurtosis ≈ 0; a real transient adds heavy tails → kurtosis > 0. One of the earliest signal flags in GW searches.
- **Skewness** — asymmetry of the distribution. The weakest of the features, but a cheap glitch discriminator: a real oscillatory chirp is fairly symmetric, while one-sided glitches are strongly skewed.

### Frequency-domain features

Three numbers sketch the spectral shape efficiently:

- **Spectral centroid** — the power-weighted "centre of mass" of the spectrum. Encodes (indirectly) the total mass, since heavier systems merge at lower frequencies. (Turning up the bass on a speaker lowers the centroid — same concept.)
- **Spectral bandwidth** — the power-weighted spread around the centroid. Narrow = tone-like; wide = broadband or rapidly sweeping.
- **Spectral entropy** — how flat-vs-peaked the spectrum is. White noise = maximum entropy; a pure tone = near-zero. Sensitive to the *organisation* of the spectrum, which neither centroid nor bandwidth captures.

Kurtosis and spectral bandwidth are only loosely correlated (~0.3–0.5), which is exactly what you want — complementary features carry independent information.

### The Hilbert-based instantaneous frequency

The **Hilbert transform** builds the *analytic signal* `z(t) = A(t)·e^(iφ(t))` from the real strain. From it:
- the **amplitude envelope** `A(t) = |z(t)|`, and
- the **instantaneous frequency** `f(t) = (1/2π) dφ/dt`, computed by unwrapping the phase and differentiating numerically.

This directly visualises the chirp sweeping from ~40 Hz toward ~150 Hz.

**Its honest limitation:** in the low-SNR inspiral window *before* merger, the phase is dominated by noise rather than signal, so the instantaneous-frequency track wanders unphysically (medians around 200+ Hz instead of the physical 40–80 Hz). This makes the Hilbert-based **chirp mass estimate unreliable** — in this project it is *'honestly reported'* as `nan`, with the matched filter (notebook 05) providing the trustworthy route. This failure is itself instructive: it is exactly *why* LIGO uses matched filtering rather than direct instantaneous-frequency estimation for parameter extraction.

### Q-transform energy features

Four numbers summarise the time-frequency map: peak energy (how loud and focused), time of peak (should sit near t = 0), frequency of peak (~150 Hz for GW150914, set by total mass), and energy concentration (fraction of energy in the brightest bins — high for a coherent chirp, lower for diffuse noise).

### Why a radar chart (and its limits)

The radar chart compares the normalised features of H1 vs L1 at a glance — for a real event the two profiles should overlay closely. It works for two or three detectors but becomes unreadable with more; for many events a **heatmap** or **parallel-coordinates plot** is the better choice. Radar charts also mislead on area (which depends on axis ordering), so they are best treated as a quick visual sanity check, not a rigorous comparison.

---

## 6. Modelling

### What is a chirp, physically?

As two compact objects spiral inward, they orbit faster, and a faster orbit radiates higher-frequency gravitational waves. The frequency sweeps upward — the signal "chirps." In the final 0.2 s before merger, GW150914's black holes orbited at roughly half the speed of light, dozens of times per second.

### The post-Newtonian (PN) approximation

Solving Einstein's equations exactly for two orbiting masses is intractable. PN theory bridges the gap: it starts from Newtonian gravity and adds relativistic corrections as a series in powers of (v/c). Orders are numbered 0PN, 1PN, 1.5PN, 2PN, 2.5PN, …; half-integer orders appear because effects like radiation reaction and spin-orbit coupling first enter at odd powers of v/c.

The **leading-order (0PN)** frequency and phase evolution used in this project are:

```
f(t) = (1 / 8πℳ) · (5ℳ / (t_merge − t))^(3/8)

Φ(t) = −2 · ((t_merge − t) / 5ℳ)^(5/8) + Φ₀
```

where ℳ = Gℳ_chirp/c³ is the chirp mass in geometric (time) units. These are consistent: differentiating Φ(t) and dividing by 2π recovers f(t). The amplitude grows as (t_merge − t)^(−1/4) — the last cycles are the loudest.

PN theory works well in the early inspiral but **breaks down near merger**, where velocities approach c and the series stops converging. Capturing the merger and ringdown requires higher-order PN, or full **numerical relativity** (supercomputer solutions of Einstein's equations, first achieved in 2005), or fast NR-calibrated models (EOB, IMRPhenom, surrogate models) accessed through **LALSuite**.

Note: PN theory is entirely **classical** — no quantum effects. The black holes are ~39 orders of magnitude too massive for quantum gravity to matter in their orbital dynamics. (For neutron stars, quantum physics enters indirectly through the nuclear equation of state, via tidal deformability — but that is a different story.)

### Why chirp mass specifically?

The chirp mass, ℳ_chirp = (m₁m₂)^(3/5) / (m₁+m₂)^(1/5), is the single combination of the two masses that controls the leading-order frequency evolution. You can measure it precisely from the sweep rate alone (~1% precision for GW150914), whereas the individual masses are far less well constrained from the inspiral. It immediately classifies the system (BNS ≈ 1.2 M☉; stellar BBH ≈ 10–35 M☉; etc.) and, historically, GW150914's chirp mass proved that black holes this massive form and merge.

### Matched filtering — the optimal linear detector

Matched filtering is *provably* the optimal **linear** strategy for detecting a **known** signal in **Gaussian** noise. Every word matters:

- **Known signal** — not "previously observed", but *predicted from theory*. The template comes from GR via the PN equations, given the masses. The signal was known mathematically decades before it was ever measured.
- **Gaussian noise** — the optimality proof assumes Gaussian, stationary noise characterised by its PSD. Real noise isn't perfectly Gaussian (see §3), which is why LIGO adds chi-squared tests and vetoes on top.
- **Linear** — only linear operations on the data, which is what makes the optimality provable (via Cauchy–Schwarz).

The core operation, in the frequency domain:

```
ρ(t) = 4 · Re ∫ [ s̃(f) · h̃*(f) / Sₙ(f) ] · e^(2πift) df
```

| Term | Meaning |
|---|---|
| ρ(t) | Output SNR as a function of time; its peak locates the signal |
| s̃(f) | Data in the frequency domain |
| h̃*(f) | Template, conjugated → frequency-domain correlation |
| Sₙ(f) | Noise PSD; dividing by it down-weights noisy frequencies |
| e^(2πift) | Time-shift factor; with the integral, an inverse FFT over all arrival times |
| 4 Re | Real part + normalisation for integrating positive frequencies only |

The noise weighting (1/Sₙ) is the heart of the method — it listens harder where the detector is quiet. This is the same principle behind LIGO's production pipelines (PyCBC, GstLAL), which scan banks of thousands of templates.

### The template bank — resolving the "known signal" paradox

We did not know GW150914's masses in advance. The resolution is a **template bank**: build thousands of templates spanning all plausible masses/spins, run all of them against the data, and the best-matching template's parameters become the estimate. Notebook 05's chirp-mass scan is a 1-D simplified version of this.

### Debugging the matched filter (a worked lesson)

Getting the matched filter right involved three instructive fixes:
1. **Don't double-whiten.** Feeding already-whitened (processed) data *and* dividing by the PSD double-counts. Fix: run the filter on **raw** data and let the filter's 1/Sₙ weighting do the whitening. (Symptom: SNR ≈ 10²².)
2. **Avoid FFT wraparound.** The inverse FFT treats data as periodic, producing spurious peaks at the segment edges. Fix: restrict the peak search to ±2 s around the merger. (Symptom: peak at t ≈ +15.8 s.)
3. **Correct the template reference.** The PN template's merger sits at its *end*, so the raw peak time is offset by the template duration (~0.19 s). Subtracting it brings the peak to within ~0.05 s of t = 0.

Final result: H1 SNR ≈ 24, L1 SNR ≈ 20, both within ~10 ms of each other — a clean detection. The values exceed LIGO's published 18.2 / 13.8 partly because the absolute normalisation isn't calibrated to LIGO's convention; the physically meaningful facts (a strong, coincident peak at the merger) are what matter.

---

## 7. Results

### What "LIGO's Bayesian posterior" means

LIGO doesn't report a single number for each parameter — it reports a **probability distribution** (the *posterior*) over all source parameters jointly (~15 dimensions: both masses, spins, distance, sky position, inclination, etc.). The published "28.3 ± 1.5 M☉" is a compressed summary (median + 90% credible interval) of that distribution.

The posterior is computed via Bayes' theorem — **prior** (what was believed beforehand) updated by the **likelihood** (how well each parameter set fits the data) — using stochastic samplers (MCMC, nested sampling) in tools like **LALInference** and **Bilby**. Because it is *joint*, it captures correlations between parameters (e.g. the distance–inclination degeneracy) that a one-at-a-time estimate would miss.

In plain terms: this pipeline's chirp-mass scan gives a single dart on the board ("about 28"). LIGO's posterior gives the whole heat-map of where the dart probably is — the best guess *and* how tightly to trust it.

### Why our numbers differ from LIGO's — and why that's the point

| Source of difference | This pipeline | LIGO production |
|---|---|---|
| Waveform template | 0PN inspiral only | Full NR / EOB / Phenom (inspiral-merger-ringdown) |
| Noise PSD | Short off-source window | Long, carefully calibrated baseline |
| Parameter estimation | 1-D grid scan | Joint Bayesian inference (~15-D) |
| Detector network | H1 + L1 | H1 + L1 + Virgo + KAGRA |
| Background / significance | None | Time slides, chi-squared, data-quality vetoes |

The gap between our result and LIGO's is a *measure of how much engineering goes into the production analysis* — not a flaw in the physics extracted. What the pipeline genuinely recovers — the merger time, the chirp sweep, a chirp mass in the right ballpark, detection significance is above threshold in both detectors, H1/L1 consistency — is real.

---

## 8. Glossary

**Amplitude Spectral Density (ASD)** — the square root of the PSD; describes how noise amplitude varies with frequency, in units of strain/√Hz.

**Analytic signal** — a complex-valued signal built from a real signal and its Hilbert transform; its magnitude is the amplitude envelope and its phase derivative is the instantaneous frequency.

**Bandpass filter** — a filter that passes frequencies within a band (here 35–350 Hz) and attenuates those outside it.

**Bayesian posterior** — the probability distribution over parameters *after* combining prior knowledge with the data likelihood; LIGO's full parameter-estimation output.

**Chirp** — the upward frequency sweep of a compact-binary inspiral as the orbit speeds up toward merger.

**Chirp mass (ℳ_chirp)** — the mass combination (m₁m₂)^(3/5)/(m₁+m₂)^(1/5) that controls the leading-order frequency evolution; the most precisely measurable mass parameter.

**Coalescence / merger** — the moment the two compact objects collide and form a single object.

**Credible interval** — the Bayesian analogue of a confidence interval; e.g. the 90% range that contains 90% of the posterior probability.

**Geometric units** — units where mass is expressed in seconds via Gℳ/c³; natural for PN waveform formulae.

**Glitch** — a short, non-astrophysical noise transient in detector data.

**Hilbert transform** — an operation producing a 90°-phase-shifted copy of a signal, used to form the analytic signal.

**Inspiral** — the long phase where two compact objects gradually spiral inward, radiating gravitational waves.

**Kurtosis (Fisher)** — a measure of distribution tail-weight; 0 for Gaussian, positive ("leptokurtic") for heavy-tailed transient signals.

**Matched filter** — the optimal linear detector for a known signal in Gaussian noise; cross-correlates data with a template weighted by the inverse noise PSD.

**Numerical relativity (NR)** — direct supercomputer solution of Einstein's equations, capturing the full merger and ringdown.

**Post-Newtonian (PN) approximation** — an expansion of GR in powers of (v/c), valid in the slow-motion inspiral regime.

**Power Spectral Density (PSD)** — noise power as a function of frequency; central to whitening and matched filtering.

**Q-transform** — a time-frequency transform with frequency-dependent window width; the standard tool for visualising GW chirps.

**Ringdown** — the final phase where the newly formed black hole oscillates and settles, radiating gravitational waves at its characteristic frequency.

**Skewness** — a measure of distribution asymmetry; ≈ 0 for symmetric noise.

**Signal-to-noise ratio (SNR)** — a measure of signal strength relative to noise; the matched-filter SNR peaks at the signal arrival time.

**Spectral centroid / bandwidth / entropy** — power-weighted mean frequency, power-weighted spread, and flatness of the spectrum.

**Strain (h)** — the dimensionless fractional change in detector arm length, ΔL/L, caused by a passing gravitational wave.

**Template** — a theoretical waveform (here from PN theory) cross-correlated against the data in matched filtering.

**Template bank** — a grid of templates spanning plausible source parameters, run together to detect signals of unknown parameters.

**Whitening** — rescaling the data so the noise has roughly equal power at all frequencies, so no band dominates.

---

## 9. References & Further Reading

### The discovery and its parameters

- Abbott, B. P. et al. (LIGO Scientific Collaboration & Virgo Collaboration), 2016. *Observation of Gravitational Waves from a Binary Black Hole Merger.* Phys. Rev. Lett. **116**, 061102. [DOI](https://doi.org/10.1103/PhysRevLett.116.061102) · [arXiv:1602.03837](https://arxiv.org/abs/1602.03837) — the discovery paper.
- Abbott, B. P. et al., 2016. *Properties of the Binary Black Hole Merger GW150914.* Phys. Rev. Lett. **116**, 241102. [DOI](https://doi.org/10.1103/PhysRevLett.116.241102) · [arXiv:1602.03840](https://arxiv.org/abs/1602.03840) — the source-parameter (mass, spin, distance) paper.
- Abbott, B. P. et al., 2016. *Tests of General Relativity with GW150914.* Phys. Rev. Lett. **116**, 221101. [arXiv:1602.03841](https://arxiv.org/abs/1602.03841).

### Open data and software

- **GWOSC — Gravitational Wave Open Science Center:** [gwosc.org](https://gwosc.org) — the source of all strain data, plus tutorials and event catalogues.
- **gwpy:** [gwpy.github.io](https://gwpy.github.io) — the Python package used throughout this project.
- **PyCBC:** [pycbc.org](https://pycbc.org) — a production matched-filtering pipeline; excellent tutorials.
- **LALSuite / Bilby:** [lscsoft.docs.ligo.org/lalsuite](https://lscsoft.docs.ligo.org/lalsuite/) · [bilby-dev.github.io/bilby](https://bilby-dev.github.io/bilby/) — production waveform and parameter-estimation tools.

### Tutorials and background

- LIGO/GWOSC, *Signal Processing with GW150914 Open Data* — the main tutorial this pipeline echoes (available via the GWOSC site).
- Abbott, B. P. et al., 2016. *GW150914: The Advanced LIGO Detectors in the Era of First Discoveries.* Phys. Rev. Lett. **116**, 131103 — on the instruments themselves.
- Chatterji, S. et al., 2004. *Multiresolution techniques for the detection of gravitational-wave bursts* — the Q-transform as adapted for GW astronomy.

### Popular science

- Levin, J., 2016. *Black Hole Blues and Other Songs from Outer Space* — a vivid history of LIGO and the people behind it.
- Bartusiak, M., 2017. *Einstein's Unfinished Symphony* — gravitational wave astronomy for a general audience.

### Going further

- **GWTC catalogues** — the Gravitational-Wave Transient Catalogues list every confirmed detection (via GWOSC).
- **Gravity Spy** — a citizen-science project for classifying detector glitches: [gravityspy.org](https://www.zooniverse.org/projects/zooniverse/gravity-spy).
- **GraceDB** — the Gravitational-Wave Candidate Event Database, with near-real-time alerts.

---

*This guide accompanies the Cosmic Explosion Explorer pipeline. For the runnable analysis, see the notebooks and `README.md`.*
