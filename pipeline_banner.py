"""
pipeline_banner.py — LIGO Gravitational Wave Analysis Pipeline
==============================================================
Generates an SVG pipeline-flow banner for embedding in Jupyter notebooks.

The banner shows all six pipeline stages, with the current stage highlighted
in the deep-space teal accent colour and all other stages muted.

Usage (at the top of any pipeline notebook):
    from IPython.display import HTML
    import sys
    sys.path.insert(0, ".")
    from pipeline_banner import pipeline_banner

    HTML(pipeline_banner(1))   # highlight stage 01 — data acquisition
    HTML(pipeline_banner(3))   # highlight stage 03 — visualisation

Each notebook should call this once, right after the header markdown cell,
to give the reader an immediate visual fix on where they are in the pipeline.
"""

from IPython.display import HTML
from typing import Optional


# ── Pipeline stage definitions ─────────────────────────────────────────────────
# Each stage: (number_label, line1, line2, short_description)
# line2 is empty for stages that fit on one row.
# short_description appears under the active step only.

_STAGES = [
    ("01", "data",     "acquisition", "fetch · inspect · save"),
    ("02", "pre-",     "processing",  "whiten · filter · notch"),
    ("03", "visuali-", "sation",      "strain · ASD · Q-transform"),
    ("04", "feature",  "extraction",  "SNR · chirp · statistics"),
    ("05", "model-",   "ling",        "matched filter · template"),
    ("06", "results",  "",            "summary · figures · export"),
]

# ── Colour tokens ──────────────────────────────────────────────────────────────
_ACCENT      = "#4a9eca"   # teal — active step border, text, arrows
_ACCENT_GLOW = "#4a9eca"   # same hue for halo rings
_ACCENT_TEXT = "#7eb8e0"   # lighter teal for active label text
_ACCENT_SUB  = "#5a98c0"   # mid teal for active subtitle text
_ACCENT_BADGE= "#4a9eca"   # step number badge on active step

_DIM_FILL    = "#111827"   # near-black fill for inactive steps
_DIM_STROKE  = "#1e3a5f"   # dark blue border for inactive steps
_DIM_LABEL   = "#3a5a7a"   # muted teal-grey label text
_DIM_ARROW   = "#1e3a5f"   # very dim arrow colour

_ACTIVE_FILL   = "#0f2236"  # deep navy fill for active step
_ACTIVE_STROKE = _ACCENT

_BG_FILL     = "#090f16"   # outer pill background
_BG_STROKE   = "#1a2e45"   # outer pill border


def pipeline_banner(
    active: int,
    width: str = "100%",
    show_description: bool = True,
) -> HTML:
    """
    Return an IPython HTML object containing an SVG pipeline flow banner.

    Parameters
    ----------
    active : int
        Which stage to highlight, 1–6.
    width : str
        CSS width for the SVG element. Default "100%" fills the notebook cell.
    show_description : bool
        If True, show a short description line beneath the active step label.
        Set False for a more compact banner.

    Returns
    -------
    IPython.display.HTML
        Render directly by placing this call as the last expression in a cell,
        or wrap in display() for explicit rendering mid-cell.

    Examples
    --------
    >>> HTML(pipeline_banner(1))    # data acquisition highlighted
    >>> HTML(pipeline_banner(4))    # feature extraction highlighted
    >>> display(pipeline_banner(2)) # explicit display mid-cell
    """
    if not 1 <= active <= 6:
        raise ValueError(f"active must be 1–6, got {active}")

    svg = _build_svg(active, width, show_description)
    return HTML(svg)


# ── Layout constants ───────────────────────────────────────────────────────────

_VIEWBOX_W  = 680
_VIEWBOX_H  = 130
_PILL_X, _PILL_Y = 8, 18
_PILL_W, _PILL_H = 664, 94
_PILL_RX    = 12

_BOX_Y      = 31    # top of all step boxes
_BOX_H      = 68    # height of all step boxes
_BOX_RX     = 8

_ARROW_Y    = 65    # vertical centre for all arrows
_ARROW_GAP  = 2     # gap between box edge and arrow tip

# Widths: stages 1-4 are 98px wide, stages 5-6 are 80px (tighter at the end)
_WIDTHS     = [98, 98, 98, 98, 80, 80]
_GAP        = 16    # horizontal gap filled by the arrow between boxes

# Pre-compute x positions
def _box_positions():
    xs = []
    x = _PILL_X + 15   # left padding inside the pill
    for w in _WIDTHS:
        xs.append(x)
        x += w + _GAP
    return xs

_XS = _box_positions()


def _cx(i):
    """Horizontal centre of stage i (0-indexed)."""
    return _XS[i] + _WIDTHS[i] // 2


def _build_svg(active: int, width: str, show_description: bool) -> str:
    """Assemble the complete SVG string."""
    parts = [_svg_open(width), _defs(), _pill(), _glow_halos(active - 1)]

    for i, stage in enumerate(_STAGES):
        is_active = (i == active - 1)
        parts.append(_step_box(i, stage, is_active, show_description))
        if i < len(_STAGES) - 1:
            parts.append(_arrow(i, is_active))

    parts.append("</svg>")
    return "\n".join(parts)


def _svg_open(width: str) -> str:
    return (
        f'<svg width="{width}" viewBox="0 0 {_VIEWBOX_W} {_VIEWBOX_H}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="display:block;font-family:\'Courier New\',monospace;">'
    )


def _defs() -> str:
    return f"""<defs>
  <marker id="pbArrow" viewBox="0 0 10 10" refX="8" refY="5"
          markerWidth="5" markerHeight="5" orient="auto-start-reverse">
    <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke"
          stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  </marker>
</defs>"""


def _pill() -> str:
    return (
        f'<rect x="{_PILL_X}" y="{_PILL_Y}" '
        f'width="{_PILL_W}" height="{_PILL_H}" rx="{_PILL_RX}" '
        f'fill="{_BG_FILL}" stroke="{_BG_STROKE}" stroke-width="0.8"/>'
    )


def _glow_halos(idx: int) -> str:
    """Two concentric outer-glow rings around the active step."""
    x, w = _XS[idx], _WIDTHS[idx]
    lines = []
    for pad, opacity in [(0, 0.18), (-2, 0.07)]:
        lines.append(
            f'<rect x="{x - pad}" y="{_BOX_Y - pad}" '
            f'width="{w + pad*2}" height="{_BOX_H + pad*2}" '
            f'rx="{_BOX_RX + pad + 1}" '
            f'fill="none" stroke="{_ACCENT_GLOW}" stroke-width="2" '
            f'opacity="{opacity}"/>'
        )
    return "\n".join(lines)


def _step_box(idx: int, stage: tuple, is_active: bool, show_description: bool) -> str:
    """Render a single step box with appropriate styling."""
    num, line1, line2, desc = stage
    x, w = _XS[idx], _WIDTHS[idx]
    cx = x + w // 2

    if is_active:
        rect_fill   = _ACTIVE_FILL
        rect_stroke = _ACTIVE_STROKE
        sw          = "1.2"
        num_fill    = _ACCENT_BADGE
        label_fill  = _ACCENT_TEXT
        sub_fill    = _ACCENT_SUB
    else:
        rect_fill   = _DIM_FILL
        rect_stroke = _DIM_STROKE
        sw          = "0.8"
        num_fill    = _DIM_LABEL
        label_fill  = _DIM_LABEL
        sub_fill    = _DIM_LABEL

    # Vertical text layout:
    # If show_description and active:  num@44, line1@58, line2@70, desc@84
    # If no line2 (e.g. "06 results"): num@44, line1@65,           desc@82
    # Inactive:                         num@50, line1@63, line2@76

    lines = []
    lines.append(
        f'<rect x="{x}" y="{_BOX_Y}" width="{w}" height="{_BOX_H}" '
        f'rx="{_BOX_RX}" fill="{rect_fill}" stroke="{rect_stroke}" stroke-width="{sw}"/>'
    )

    has_two_lines = bool(line2)

    if is_active and show_description:
        if has_two_lines:
            y_num, y_l1, y_l2, y_desc = 44, 57, 69, 84
        else:
            y_num, y_l1, y_desc = 46, 63, 82
    else:
        if has_two_lines:
            y_num, y_l1, y_l2 = 49, 63, 76
        else:
            y_num, y_l1 = 52, 68

    # Step number badge
    lines.append(
        f'<text x="{cx}" y="{y_num}" text-anchor="middle" '
        f'font-size="9" font-weight="500" fill="{num_fill}">{num}</text>'
    )
    # Line 1
    lines.append(
        f'<text x="{cx}" y="{y_l1}" text-anchor="middle" '
        f'font-size="10" font-weight="500" fill="{label_fill}">{line1}</text>'
    )
    # Line 2 (if present)
    if has_two_lines:
        lines.append(
            f'<text x="{cx}" y="{y_l2}" text-anchor="middle" '
            f'font-size="10" font-weight="500" fill="{label_fill}">{line2}</text>'
        )
    # Description (active step only)
    if is_active and show_description:
        lines.append(
            f'<text x="{cx}" y="{y_desc}" text-anchor="middle" '
            f'font-size="8.5" font-weight="400" '
            f'font-family="\'Helvetica Neue\',Arial,sans-serif" '
            f'fill="{sub_fill}">{desc}</text>'
        )

    return "\n".join(lines)


def _arrow(from_idx: int, from_is_active: bool) -> str:
    """Render the arrow between stage from_idx and from_idx+1."""
    x1 = _XS[from_idx] + _WIDTHS[from_idx] + _ARROW_GAP
    x2 = _XS[from_idx + 1] - _ARROW_GAP

    if from_is_active:
        stroke  = _ACCENT
        sw      = "1.2"
        opacity = "0.75"
    else:
        stroke  = _DIM_ARROW
        sw      = "0.9"
        opacity = "1"

    return (
        f'<line x1="{x1}" y1="{_ARROW_Y}" x2="{x2}" y2="{_ARROW_Y}" '
        f'stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}" '
        f'fill="none" marker-end="url(#pbArrow)"/>'
    )


# ── Convenience: render all six variants side-by-side (dev/testing) ───────────

def _preview_all() -> HTML:
    """
    Return an HTML block showing all six banner variants stacked vertically.
    Useful for visual QA — call display(_preview_all()) in a notebook cell.
    """
    svgs = []
    for i in range(1, 7):
        svgs.append(f"<p style='color:#7eb8e0;font-size:11px;margin:4px 0 2px;'>Stage {i:02d} active</p>")
        svgs.append(pipeline_banner(i)._repr_html_())
    return HTML(
        "<div style='background:#090f16;padding:16px;border-radius:8px;'>"
        + "\n".join(svgs)
        + "</div>"
    )
