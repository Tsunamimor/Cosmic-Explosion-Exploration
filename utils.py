"""
utils.py — LIGO Gravitational Wave Analysis Pipeline
=====================================================
Reusable functions shared across pipeline notebooks.

Rather than copy-pasting the same load/plot logic into every notebook, we
centralise it here. Each function is documented with:
    - What it does
    - Why it is implemented this way
    - What to watch out for

Usage (in any notebook):
    import sys
    sys.path.insert(0, ".")
    from utils import load_raw_hdf5, load_processed_hdf5, plot_timeseries
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py

from gwpy.timeseries import TimeSeries
from gwpy.spectrogram import Spectrogram

from config import PATHS, EVENT, SIGNAL, VIZ


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA I/O
# ══════════════════════════════════════════════════════════════════════════════

def load_raw_hdf5(
    event_name: str = EVENT.NAME,
    detectors: List[str] = None,
) -> Dict[str, TimeSeries]:
    """
    Load raw strain data from the HDF5 file produced by notebook 01.

    Reconstructs gwpy.TimeSeries objects from the saved numpy arrays.
    The TimeSeries wrapper is important because it carries:
        - `.times`        → GPS timestamps as a Quantity array
        - `.sample_rate`  → sampling frequency as a Quantity
        - Signal processing methods (`.whiten()`, `.bandpass()`, etc.)

    Parameters
    ----------
    event_name : str
        Event identifier, e.g. "GW150914". Used to find the HDF5 file.
    detectors : list of str, optional
        Which detectors to load. Defaults to EVENT.DETECTORS.

    Returns
    -------
    dict
        {detector_name: gwpy.TimeSeries}, e.g. {"H1": <TimeSeries>, ...}
    """
    detectors = detectors or EVENT.DETECTORS
    filepath   = PATHS.raw_hdf5(event_name)

    if not filepath.exists():
        raise FileNotFoundError(
            f"Raw HDF5 not found: {filepath}\n"
            "Run 01_data_acquisition.ipynb first."
        )

    raw = {}
    with h5py.File(filepath, "r") as f:
        for det in detectors:
            strain      = f[det]["strain"][:]
            times       = f[det]["times"][:]
            sample_rate = f[det].attrs["sample_rate"]

            # Reconstruct TimeSeries from saved arrays.
            # unit="" is the correct string for dimensionless strain —
            # unit="dimensionless" is invalid in astropy and causes downstream
            # AttributeError failures (learned the hard way in notebook 02).
            raw[det] = TimeSeries(
                strain,
                t0=times[0],
                dt=1.0 / sample_rate,
                unit="",
                name=f"{det}:STRAIN",
            )

    print(f"Loaded raw data ({event_name}): "
          + ", ".join(f"{d} [{len(raw[d])} samples]" for d in raw))
    return raw


def load_processed_hdf5(
    event_name: str = EVENT.NAME,
    detectors: List[str] = None,
) -> Tuple[Dict[str, TimeSeries], dict]:
    """
    Load processed strain data from the HDF5 file produced by notebook 02.

    Also returns the preprocessing parameters stored in the file, so
    downstream notebooks can verify what conditioning was applied.

    Parameters
    ----------
    event_name : str
        Event identifier, e.g. "GW150914".
    detectors : list of str, optional
        Which detectors to load. Defaults to EVENT.DETECTORS.

    Returns
    -------
    processed : dict
        {detector_name: gwpy.TimeSeries} of whitened + filtered strain.
    params : dict
        Preprocessing parameters stored in the HDF5 metadata group.
    """
    detectors = detectors or EVENT.DETECTORS
    filepath   = PATHS.processed_hdf5(event_name)

    if not filepath.exists():
        raise FileNotFoundError(
            f"Processed HDF5 not found: {filepath}\n"
            "Run 02_preprocessing.ipynb first."
        )

    processed = {}
    params    = {}

    with h5py.File(filepath, "r") as f:
        # Read preprocessing parameters for provenance / logging
        if "preprocessing_params" in f:
            params = dict(f["preprocessing_params"].attrs)

        for det in detectors:
            strain      = f[det]["strain"][:]
            times       = f[det]["times"][:]
            sample_rate = f[det].attrs["sample_rate"]

            processed[det] = TimeSeries(
                strain,
                t0=times[0],
                dt=1.0 / sample_rate,
                unit="",
                name=f"{det}:PROCESSED",
            )

    print(f"Loaded processed data ({event_name}): "
          + ", ".join(f"{d} [{len(processed[d])} samples]" for d in processed))
    print(f"Preprocessing applied: {params.get('steps_applied', 'unknown')}")
    return processed, params


# ══════════════════════════════════════════════════════════════════════════════
# 2. SIGNAL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def time_relative_to_merger(
    ts: TimeSeries,
    gps_merge_time: float = EVENT.GPS_MERGE_TIME,
) -> np.ndarray:
    """
    Return the time axis of a TimeSeries as seconds relative to the merger.

    LIGO GPS times are large numbers (~1.1 billion seconds). Plotting them
    directly produces an illegible x-axis. Subtracting the merger GPS time
    gives a human-readable axis centred on zero.

    Parameters
    ----------
    ts : gwpy.TimeSeries
    gps_merge_time : float
        GPS time of the event merger. Defaults to GW150914.

    Returns
    -------
    np.ndarray
        Seconds relative to merger (negative = before, positive = after).
    """
    return ts.times.value - gps_merge_time


def crop_to_window(
    ts: TimeSeries,
    window: Tuple[float, float],
    gps_merge_time: float = EVENT.GPS_MERGE_TIME,
) -> TimeSeries:
    """
    Crop a TimeSeries to a time window specified relative to the merger.

    Parameters
    ----------
    ts : gwpy.TimeSeries
    window : (float, float)
        (start, end) in seconds relative to merger. E.g. (-0.5, 0.5).
    gps_merge_time : float

    Returns
    -------
    gwpy.TimeSeries
        Cropped to [gps_merge_time + window[0], gps_merge_time + window[1]].
    """
    t_start = gps_merge_time + window[0]
    t_end   = gps_merge_time + window[1]
    return ts.crop(t_start, t_end)


def estimate_snr(
    ts: TimeSeries,
    gps_merge_time: float  = EVENT.GPS_MERGE_TIME,
    noise_window: Tuple    = SIGNAL.NOISE_WINDOW,
    signal_window: Tuple   = SIGNAL.SIGNAL_WINDOW,
) -> dict:
    """
    Estimate a simple RMS-based signal-to-noise ratio.

    This is a rough sanity check, not a proper matched-filter SNR. It
    compares RMS power in a quiet noise window (well before the merger)
    to RMS power in the signal window (around the merger).

    Proper matched-filter SNR, which accounts for the frequency-dependent
    noise floor, is left to notebook 05 (modelling).

    Parameters
    ----------
    ts : gwpy.TimeSeries
        Processed (whitened + filtered) strain.
    gps_merge_time : float
    noise_window : (float, float)
        Seconds relative to merger defining the quiet reference region.
    signal_window : (float, float)
        Seconds relative to merger defining the expected signal region.

    Returns
    -------
    dict with keys: noise_rms, signal_rms, snr
    """
    t = time_relative_to_merger(ts, gps_merge_time)
    v = ts.value

    noise_mask  = (t >= noise_window[0])  & (t <= noise_window[1])
    signal_mask = (t >= signal_window[0]) & (t <= signal_window[1])

    noise_rms  = np.sqrt(np.mean(v[noise_mask]  ** 2))
    signal_rms = np.sqrt(np.mean(v[signal_mask] ** 2))
    snr        = signal_rms / noise_rms if noise_rms > 0 else np.nan

    return {"noise_rms": noise_rms, "signal_rms": signal_rms, "snr": snr}


# ══════════════════════════════════════════════════════════════════════════════
# 3. PLOTTING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _merger_line(ax, label: bool = True):
    """Draw a red dashed vertical line at t=0 (the merger time)."""
    ax.axvline(0, color="red", lw=1.2, linestyle="--",
               label="Merger" if label else None)


def _signal_window_band(ax, window: Tuple = SIGNAL.SIGNAL_WINDOW):
    """Shade the expected signal window in translucent red."""
    ax.axvspan(window[0], window[1], alpha=0.08, color="red", zorder=0)


def plot_timeseries(
    data: Dict[str, TimeSeries],
    gps_merge_time: float = EVENT.GPS_MERGE_TIME,
    window: Optional[Tuple[float, float]] = None,
    title: str = "Strain vs Time",
    ylabel: str = VIZ.YLABEL_NORM_STRAIN,
    show_merger: bool = True,
    show_signal_band: bool = True,
    save_path: Optional[Path] = None,
    figsize: Tuple = (12, 4),
) -> plt.Figure:
    """
    Plot one or more TimeSeries on a shared figure, one axis per detector.

    Parameters
    ----------
    data : dict
        {detector: TimeSeries} to plot.
    gps_merge_time : float
    window : (float, float), optional
        If provided, crop each series to this window before plotting.
    title : str
    ylabel : str
    show_merger : bool
        Whether to draw a red dashed line at t=0.
    show_signal_band : bool
        Whether to shade the signal window.
    save_path : Path, optional
        If provided, save the figure here.
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(data)
    fig, axes = plt.subplots(n, 1, figsize=(figsize[0], figsize[1] * n),
                             sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (det, ts) in zip(axes, data.items()):
        if window:
            ts = crop_to_window(ts, window, gps_merge_time)
        t = time_relative_to_merger(ts, gps_merge_time)
        colour = EVENT.DETECTOR_COLOURS.get(det, "steelblue")

        ax.plot(t, ts.value, color=colour, lw=0.5, alpha=0.85)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"{EVENT.DETECTOR_LABELS.get(det, det)}", fontsize=11)
        ax.grid(True, alpha=0.25)

        if show_merger:
            _merger_line(ax)
        if show_signal_band:
            _signal_window_band(ax)

        ax.legend(fontsize=8)

    axes[-1].set_xlabel(VIZ.XLABEL_TIME, fontsize=10)
    fig.suptitle(title, fontsize=13, y=1.01)
    fig.set_constrained_layout(True)

    if save_path:
        fig.savefig(save_path, dpi=VIZ.FIGURE_DPI, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_asd(
    data: Dict[str, TimeSeries],
    fftlength: int = SIGNAL.WHITEN_FFTLENGTH,
    frange: Tuple = (10, 2000),
    bandpass: Optional[Tuple] = (SIGNAL.BANDPASS_LOW_HZ, SIGNAL.BANDPASS_HIGH_HZ),
    title: str = "Amplitude Spectral Density",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot the Amplitude Spectral Density (ASD) for each detector.

    ASD = sqrt(PSD), with units of strain/√Hz. Plotting the ASD rather than
    the PSD is conventional in LIGO work because it has the same units as
    strain, making it easier to compare the noise floor to signal amplitude.

    Parameters
    ----------
    data : dict
        {detector: TimeSeries}. Can be raw or processed.
    fftlength : int
        FFT segment length in seconds (controls frequency resolution).
    frange : (float, float)
        Frequency axis display range in Hz.
    bandpass : (float, float), optional
        If provided, shade the bandpass region on the plot.
    title : str
    save_path : Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for det, ts in data.items():
        colour = EVENT.DETECTOR_COLOURS.get(det, "steelblue")
        asd = ts.asd(fftlength=fftlength, method="median")
        ax.plot(asd.frequencies.value, asd.value,
                color=colour, lw=1.0, label=EVENT.DETECTOR_LABELS.get(det, det),
                alpha=0.85)

    if bandpass:
        ax.axvspan(bandpass[0], bandpass[1], alpha=0.12, color="green",
                   label=f"Bandpass {bandpass[0]}–{bandpass[1]} Hz")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(frange)
    ax.set_xlabel(VIZ.XLABEL_FREQ, fontsize=10)
    ax.set_ylabel("ASD (strain / √Hz)", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    fig.set_constrained_layout(True)


    if save_path:
        fig.savefig(save_path, dpi=VIZ.FIGURE_DPI, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_qtransform(
    ts: TimeSeries,
    detector: str,
    gps_merge_time: float = EVENT.GPS_MERGE_TIME,
    outseg: Tuple  = VIZ.QTRANSFORM_OUTSEG,
    frange: Tuple  = VIZ.QTRANSFORM_FRANGE,
    qrange: Tuple  = VIZ.QTRANSFORM_QRANGE,
    tres: float    = VIZ.QTRANSFORM_TRES,
    colormap: str  = VIZ.COLOURMAP_QTRANS,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Compute and plot the Q-transform (time-frequency representation).

    The Q-transform is the standard tool for visualising gravitational wave
    signals. Unlike a fixed-window STFT spectrogram, it uses a
    frequency-dependent window width, giving better time resolution at high
    frequencies and better frequency resolution at low frequencies. This is
    well-matched to chirp signals that sweep upward rapidly.

    What you should see for GW150914:
        A bright arc starting below 100 Hz around t = -0.2 s and sweeping
        upward to ~150 Hz at t = 0 (the merger). This is the binary black
        hole chirp — the inspiral phase where the two black holes are
        spiralling together faster and faster.

    Parameters
    ----------
    ts : gwpy.TimeSeries
        Processed strain. Raw strain is too noisy for a useful Q-transform.
    detector : str
        e.g. "H1" — used for the plot title and colour.
    gps_merge_time : float
    outseg : (float, float)
        Time window (seconds relative to merger) to display.
    frange : (float, float)
        Frequency display range in Hz.
    qrange : (float, float)
        Range of Q values to search. Higher Q = finer frequency localisation.
    tres : float
        Output time resolution in seconds.
    colormap : str
    save_path : Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Convert relative-time outseg to absolute GPS times for gwpy
    abs_outseg = (gps_merge_time + outseg[0], gps_merge_time + outseg[1])

    # gwpy's q_transform handles window selection and optimisation internally.
    # logf=True gives a log-spaced frequency axis, which is standard for LIGO.
    qt = ts.q_transform(
        qrange=qrange,
        frange=frange,
        outseg=abs_outseg,
        tres=tres,
        logf=True,
    )

    # The Q-transform returns a Spectrogram. Its time axis is in GPS time —
    # we subtract the merger time to get a relative axis.
    t_axis = qt.times.value - gps_merge_time
    f_axis = qt.frequencies.value

    fig, ax = plt.subplots(figsize=(10, 5))

    # pcolormesh gives a smooth rendering (vs imshow which can alias badly)
    im = ax.pcolormesh(t_axis, f_axis, qt.value.T,
                       cmap=colormap, shading="auto", vmin=0)

    ax.set_yscale("log")
    ax.set_ylabel(VIZ.XLABEL_FREQ, fontsize=10)
    ax.set_xlabel(VIZ.XLABEL_TIME, fontsize=10)
    ax.set_title(
        f"{EVENT.DETECTOR_LABELS.get(detector, detector)} — Q-transform\n"
        f"Q range {qrange}, f range {frange[0]}–{frange[1]} Hz",
        fontsize=11,
    )

    # Mark the merger
    ax.axvline(0, color="white", lw=1.0, linestyle="--", alpha=0.7)

    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Normalised energy", fontsize=9)

    fig.set_constrained_layout(True)


    if save_path:
        fig.savefig(save_path, dpi=VIZ.FIGURE_DPI, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 4. PIPELINE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def print_hdf5_tree(filepath: Path):
    """
    Print a human-readable tree of an HDF5 file's contents.

    Useful for verifying that a file was written correctly without having
    to open it in an external tool.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return

    def _tree(name, obj):
        indent = "  " * name.count("/")
        stem   = name.split("/")[-1]
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}📊 {stem}: shape={obj.shape}, dtype={obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"{indent}📁 {stem}/")

    print(f"\n{filepath}")
    with h5py.File(filepath, "r") as f:
        f.visititems(_tree)
        if "metadata" in f:
            print("\n  Metadata attributes:")
            for k, v in f["metadata"].attrs.items():
                print(f"    {k}: {v}")


def check_pipeline_files(event_name: str = EVENT.NAME):
    """
    Check which pipeline output files exist for a given event.

    Run this at the start of a notebook to confirm the upstream notebooks
    have been executed successfully.
    """
    files = {
        "Raw HDF5 (notebook 01)":       PATHS.raw_hdf5(event_name),
        "Processed HDF5 (notebook 02)": PATHS.processed_hdf5(event_name),
    }
    all_ok = True
    print(f"Pipeline file check — {event_name}:")
    for label, path in files.items():
        status = "✓" if path.exists() else "✗ MISSING"
        print(f"  [{status}] {label}: {path}")
        if not path.exists():
            all_ok = False

    if not all_ok:
        print("\nSome files are missing. Run notebooks in order: 01 → 02 → 03.")
    else:
        print("\nAll upstream files present. Ready to proceed.")
    return all_ok
