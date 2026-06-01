"""
config.py — LIGO Gravitational Wave Analysis Pipeline
======================================================
Central configuration file for all pipeline notebooks.

Keeping constants in one place means that if you want to analyse a different
event, change a filter parameter, or point the pipeline at a different data
directory, you only ever edit this file. Every notebook imports from here.

Usage (in any notebook):
    import sys
    sys.path.insert(0, ".")          # Ensure project root is on the path
    from config import PATHS, EVENT, SIGNAL, VIZ
"""

from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# 1. FILE PATHS
# ══════════════════════════════════════════════════════════════════════════════

class PATHS:
    """
    All filesystem paths used by the pipeline.

    Directory layout produced by notebook 01:
        data/
        ├── raw/          ← HDF5 files from GWOSC
        └── processed/    ← cleaned strain from notebook 02

    Output images are written alongside the data they describe.
    """

    # Root of the project (the folder containing the notebooks)
    ROOT = Path(".")

    # Raw strain data fetched from GWOSC
    RAW_DIR       = ROOT / "data" / "raw"

    # Cleaned/processed strain from notebook 02
    PROC_DIR      = ROOT / "data" / "processed"

    # Visualisation outputs (spectrograms, Q-transforms, etc.)
    VIZ_DIR       = ROOT / "data" / "visualisations"

    # ── Convenience constructors ───────────────────────────────────────────────

    @staticmethod
    def raw_hdf5(event_name: str) -> Path:
        """Path to the raw HDF5 file for a given event."""
        return PATHS.RAW_DIR / f"{event_name}_raw.hdf5"

    @staticmethod
    def processed_hdf5(event_name: str) -> Path:
        """Path to the processed HDF5 file for a given event."""
        return PATHS.PROC_DIR / f"{event_name}_processed.hdf5"

    @staticmethod
    def viz_output(event_name: str, suffix: str) -> Path:
        """
        Path for a visualisation output file.

        Examples:
            PATHS.viz_output("GW150914", "qtransform_H1")
            → data/visualisations/GW150914_qtransform_H1.png
        """
        return PATHS.VIZ_DIR / f"{event_name}_{suffix}.png"

    @classmethod
    def make_dirs(cls):
        """Create all data directories (safe to call multiple times)."""
        for d in [cls.RAW_DIR, cls.PROC_DIR, cls.VIZ_DIR]:
            d.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 2. EVENT PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

class EVENT:
    """
    Parameters for the primary event under analysis: GW150914.

    GPS time note: LIGO uses GPS time — seconds elapsed since 6 January 1980.
    GW150914 occurred at GPS 1126259462. The 32-second segment centred on this
    time gives enough data for preprocessing and visualisation without being
    unwieldy.

    To switch to a different event, update NAME, GPS_MERGE_TIME, and
    DETECTORS, and re-run all notebooks from the top.
    """

    NAME             = "GW150914"
    GPS_MERGE_TIME   = 1126259462   # GPS time of binary black hole merger
    SEGMENT_DURATION = 32           # Total seconds of data fetched around merger
    DETECTORS        = ["H1", "L1"] # Hanford (H1) and Livingston (L1)

    # Detector full names for plot labels
    DETECTOR_LABELS  = {
        "H1": "LIGO Hanford (H1)",
        "L1": "LIGO Livingston (L1)",
        "V1": "Virgo (V1)",
    }

    # Colour palette — consistent across all notebooks
    DETECTOR_COLOURS = {
        "H1": "#1f77b4",   # Blue
        "L1": "#ff7f0e",   # Orange
        "V1": "#2ca02c",   # Green
    }

    # Nominal sample rate of the fetched data
    SAMPLE_RATE_HZ = 4096  # Hz


# ══════════════════════════════════════════════════════════════════════════════
# 3. SIGNAL CONDITIONING PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

class SIGNAL:
    """
    Parameters for whitening, bandpass, and notch filtering.

    These were chosen based on known LIGO noise characteristics for GW150914:

    Bandpass (35–350 Hz):
        - Below 35 Hz, seismic noise dominates — the signal is undetectable.
        - Above 350 Hz, the binary black hole chirp has negligible power.
        - This band is standard in LIGO BBH analyses.

    Whitening FFT length (4 s):
        - Controls frequency resolution of the noise model used to whiten.
        - Longer segments → finer resolution but more memory.
        - 4 s is a well-established default for this event.

    Notch frequencies (60, 120, 180 Hz):
        - US mains power frequency (60 Hz) and its second and third harmonics.
        - These appear as sharp spectral lines in the PSD and contaminate
          the whitened data if not suppressed.
        - Violin modes (mechanical resonances of mirror suspension fibres)
          would be added here for more thorough cleaning.

    Notch Q factor (30):
        - Controls how narrow the notch is: Q = f / Δf.
        - Q = 30 at 60 Hz means a ~2 Hz notch width — narrow enough to
          avoid cutting signal power in the adjacent band.
    """

    WHITEN_FFTLENGTH  = 4       # seconds

    BANDPASS_LOW_HZ   = 35.0    # Hz — low-frequency seismic wall cutoff
    BANDPASS_HIGH_HZ  = 350.0   # Hz — high-frequency BBH signal rolloff
    BANDPASS_ORDER    = 4       # Butterworth filter order

    NOTCH_FREQS_HZ    = [60.0, 120.0, 180.0]   # Mains harmonics
    NOTCH_Q           = 30.0   # Quality factor

    # Time windows relative to GPS_MERGE_TIME (seconds)
    # Used for SNR estimation and cropped visualisations
    NOISE_WINDOW      = (-15.0, -5.0)   # Quiet region before merger
    SIGNAL_WINDOW     = (-0.5,   0.5)   # Expected signal region


# ══════════════════════════════════════════════════════════════════════════════
# 4. VISUALISATION PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

class VIZ:
    """
    Parameters controlling plot appearance and Q-transform settings.

    Q-transform background:
        The Q-transform is a time-frequency analysis technique adapted for
        gravitational wave astronomy by Shourov Chatterji et al. (2004).
        It is better suited than a standard spectrogram (STFT) for signals
        that sweep rapidly through frequency, because it uses a
        frequency-dependent time window — finer time resolution at high
        frequency, finer frequency resolution at low frequency.

        Key parameters:
            qrange  — range of Q values to search over. Higher Q = more
                      time-frequency localisation. (7, 110) is standard.
            outseg  — time segment to display around the merger.
            frange  — frequency axis range for the plot.
            tres    — time resolution of the output image in seconds.
    """

    # ── Q-transform settings ──────────────────────────────────────────────────
    QTRANSFORM_QRANGE  = (7, 110)        # Search range for Q
    QTRANSFORM_FRANGE  = (20, 500)       # Hz — frequency axis extents
    QTRANSFORM_OUTSEG  = (-2.0, 1.0)     # Seconds relative to merger
    QTRANSFORM_TRES    = 0.001           # Time resolution (seconds)

    # ── Spectrogram settings (STFT-based) ────────────────────────────────────
    SPECGRAM_NFFT      = 1024            # FFT points per segment
    SPECGRAM_OVERLAP   = 0.9             # Fractional overlap between segments
    SPECGRAM_FRANGE    = (20, 500)       # Hz — frequency axis extents

    # ── Time-series crop window ───────────────────────────────────────────────
    TIMESERIES_WINDOW  = (-0.5, 0.5)     # Seconds around merger for zoomed plots

    # ── Plot appearance ───────────────────────────────────────────────────────
    FIGURE_DPI         = 150
    COLOURMAP_QTRANS   = "viridis"       # Perceptually uniform — recommended
    COLOURMAP_SPECGRAM = "plasma"

    # Common axis label strings
    XLABEL_TIME        = "Time relative to merger (s)"
    XLABEL_FREQ        = "Frequency (Hz)"
    YLABEL_STRAIN      = "Strain h(t)"
    YLABEL_NORM_STRAIN = "Normalised strain"
