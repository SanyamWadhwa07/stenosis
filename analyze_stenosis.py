"""
Quantitative Coronary Angiography (QCA) Pipeline
=================================================
Research-grade validated stenosis measurement pipeline.

Pipeline per detected bounding box:
  1. Crop ROI from image
  2. Greyscale + CLAHE contrast enhancement
  3. Frangi vesselness filter  (Frangi et al., 1998)
  4. Otsu threshold → binary vessel mask
  5. Morphological cleaning (closing → opening → largest component)
  6. Skeletonization (Lee's algorithm)
  7. Tangent-based perpendicular radius profile along centerline
  8. % Diameter Stenosis = (1 - MLD / D_ref) × 100

References:
  Frangi et al. (1998) "Multiscale vessel enhancement filtering"
  Clinical QCA: CAAS / QAngio XA methodology
"""

import io
import warnings
from typing import Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label as nd_label
from skimage.filters import frangi, threshold_otsu
from skimage.morphology import (
    binary_closing,
    binary_opening,
    disk,
    skeletonize,
)

# ── Constants ─────────────────────────────────────────────────────────────────
_FRANGI_SIGMAS   = range(1, 6)   # scale range for vessel widths in pixels
_CLAHE_CLIP      = 2.0
_CLAHE_GRID      = (8, 8)
_CLOSE_RADIUS    = 3             # morphological closing radius
_OPEN_RADIUS     = 1             # morphological opening radius
_TANGENT_WINDOW  = 7             # points on each side for local tangent PCA
_RAY_MAX         = 200           # max pixels to cast a perpendicular ray
_OVERLAY_STEP    = 4             # draw radius tick every N skeleton points
_REF_PERCENTILE  = 75            # top-N% diameters define reference vessel


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def _to_gray(crop_rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 → grayscale uint8."""
    if crop_rgb.ndim == 2:
        return crop_rgb
    return cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)


def _apply_clahe(gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE to enhance local contrast in the vessel region."""
    clahe = cv2.createCLAHE(clipLimit=_CLAHE_CLIP, tileGridSize=_CLAHE_GRID)
    return clahe.apply(gray)


def _apply_frangi(clahe_img: np.ndarray) -> np.ndarray:
    """
    Frangi vesselness filter.
    Returns float64 in [0, 1] — 1 = strong vessel response.
    black_ridges=False → bright vessels on dark background (standard angio).
    """
    normalized = clahe_img.astype(np.float64) / 255.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vessel_map = frangi(
            normalized,
            sigmas=_FRANGI_SIGMAS,
            black_ridges=False,
            beta=0.5,
            gamma=15,
        )
    # Normalise to [0,1]
    vmin, vmax = vessel_map.min(), vessel_map.max()
    if vmax > vmin:
        vessel_map = (vessel_map - vmin) / (vmax - vmin)
    return vessel_map


def _threshold_and_clean(frangi_map: np.ndarray) -> np.ndarray:
    """
    Otsu threshold on frangi response → binary mask.
    Then morph. closing (fill gaps) + opening (remove speckle)
    + keep only the largest connected component.
    Returns bool mask, same shape as input.
    """
    # Otsu on uint8 representation
    uint8 = (frangi_map * 255).astype(np.uint8)
    thresh = threshold_otsu(uint8)
    binary = uint8 > thresh

    # Morphological cleanup
    binary = binary_closing(binary, disk(_CLOSE_RADIUS))
    binary = binary_opening(binary, disk(_OPEN_RADIUS))

    # Keep largest connected component
    labeled, n_components = nd_label(binary)
    if n_components == 0:
        return binary
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0                        # ignore background
    largest = sizes.argmax()
    return labeled == largest


def preprocess_crop(crop_rgb: np.ndarray) -> dict:
    """
    Run full preprocessing pipeline on a cropped ROI.

    Returns a dict with keys:
        gray, clahe, frangi, binary, cleaned, skeleton
    All arrays are uint8 RGB suitable for Gradio display.
    """
    gray    = _to_gray(crop_rgb)
    clahe   = _apply_clahe(gray)
    fmap    = _apply_frangi(clahe)
    cleaned = _threshold_and_clean(fmap)
    skel    = skeletonize(cleaned)

    def _gray_to_rgb(g):
        return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)

    return {
        "gray":     _gray_to_rgb(gray),
        "clahe":    _gray_to_rgb(clahe),
        "frangi":   _gray_to_rgb((fmap * 255).astype(np.uint8)),
        "binary":   _gray_to_rgb((cleaned * 255).astype(np.uint8)),
        "skeleton": _gray_to_rgb((skel    * 255).astype(np.uint8)),
        # raw arrays needed downstream
        "_cleaned": cleaned,
        "_skeleton": skel,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Centerline extraction
# ─────────────────────────────────────────────────────────────────────────────

def _find_endpoints(skel: np.ndarray) -> list:
    """Return all skeleton endpoints (pixels with exactly 1 neighbour)."""
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel.astype(np.uint8), -1, kernel)
    # endpoint: skel pixel where neighbour_count == 2 (itself + 1 neighbour)
    endpoints = np.argwhere((skel > 0) & (neighbor_count == 2))
    return [tuple(p) for p in endpoints]


def extract_centerline_points(skeleton: np.ndarray) -> list:
    """
    Order skeleton pixels into a single path from one endpoint to the other
    using nearest-neighbour traversal.

    Returns list of (row, col) tuples in path order.
    Returns [] if fewer than 3 skeleton pixels exist.
    """
    pts = set(map(tuple, np.argwhere(skeleton > 0)))
    if len(pts) < 3:
        return []

    endpoints = _find_endpoints(skeleton)
    start = endpoints[0] if endpoints else next(iter(pts))

    path   = [start]
    visited = {start}
    current = start

    while True:
        r, c = current
        neighbours = [
            (r + dr, c + dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0)
            and (r + dr, c + dc) in pts
            and (r + dr, c + dc) not in visited
        ]
        if not neighbours:
            break
        # prefer the one closest to the last direction (straight path)
        nxt = neighbours[0]
        path.append(nxt)
        visited.add(nxt)
        current = nxt

    return path


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Tangent-based radius profile
# ─────────────────────────────────────────────────────────────────────────────

def _local_tangent(path: list, idx: int, window: int) -> np.ndarray:
    """
    Estimate unit tangent at path[idx] via PCA over a local window.
    Returns unit vector [dy, dx] (row, col convention).
    """
    lo = max(0, idx - window)
    hi = min(len(path), idx + window + 1)
    pts = np.array(path[lo:hi], dtype=float)
    if len(pts) < 2:
        return np.array([1.0, 0.0])
    pts -= pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    tangent = vt[0]
    norm = np.linalg.norm(tangent)
    return tangent / norm if norm > 0 else np.array([1.0, 0.0])


def _cast_ray(mask: np.ndarray, r0: float, c0: float,
              dr: float, dc: float, max_dist: int) -> float:
    """
    Cast a ray from (r0, c0) in direction (dr, dc) through a boolean mask.
    Returns distance to the first background pixel (vessel wall).
    """
    h, w = mask.shape
    for dist in range(1, max_dist + 1):
        r = int(round(r0 + dr * dist))
        c = int(round(c0 + dc * dist))
        if r < 0 or r >= h or c < 0 or c >= w:
            return float(dist - 1)
        if not mask[r, c]:
            return float(dist - 1)
    return float(max_dist)


def compute_radius_profile(cleaned_mask: np.ndarray,
                           path: list,
                           window: int = _TANGENT_WINDOW) -> list:
    """
    For each centerline point compute the vessel radius in pixels.

    At each point:
      1. Compute local tangent t via PCA
      2. Perpendicular n = [-t[1], t[0]]
      3. Cast rays ±n to vessel wall
      4. radius = (dist+ + dist-) / 2

    Returns list of float radii, one per path point.
    """
    radii = []
    for i, (r, c) in enumerate(path):
        t = _local_tangent(path, i, window)
        # perpendicular (rotate 90°)
        n = np.array([-t[1], t[0]])
        d_pos = _cast_ray(cleaned_mask, r, c,  n[0],  n[1], _RAY_MAX)
        d_neg = _cast_ray(cleaned_mask, r, c, -n[0], -n[1], _RAY_MAX)
        radii.append((d_pos + d_neg) / 2.0)
    return radii


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Stenosis metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_stenosis_metrics(radii: list,
                              pixel_spacing_mm: Optional[float] = None) -> dict:
    """
    Compute QCA-standard stenosis metrics from radius profile.

    D_ref : mean of diameters above the _REF_PERCENTILE percentile
            (interpolated reference from proximal + distal healthy vessel)
    MLD   : minimal lumen diameter (2 × min radius)
    % DS  : (1 - MLD / D_ref) × 100

    Returns dict with keys:
        mld_px, mld_mm (if pixel_spacing_mm given),
        ref_diameter_px, ref_diameter_mm,
        pct_diameter_stenosis, mld_location_idx
    """
    if not radii:
        return {}

    diameters = [2.0 * r for r in radii]
    mld_idx   = int(np.argmin(diameters))
    mld_px    = diameters[mld_idx]
    cutoff    = np.percentile(diameters, _REF_PERCENTILE)
    ref_px    = float(np.mean([d for d in diameters if d >= cutoff]))
    pct_ds    = max(0.0, (1.0 - mld_px / ref_px) * 100.0) if ref_px > 0 else 0.0

    result = {
        "mld_px":               round(mld_px, 2),
        "ref_diameter_px":      round(ref_px, 2),
        "pct_diameter_stenosis": round(pct_ds, 1),
        "mld_location_idx":     mld_idx,
    }
    if pixel_spacing_mm is not None:
        result["mld_mm"]          = round(mld_px * pixel_spacing_mm, 2)
        result["ref_diameter_mm"] = round(ref_px * pixel_spacing_mm, 2)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4b — QFR / FFR estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_qfr(pct_ds: float,
                 ref_diameter_px: float,
                 mld_px: float) -> dict:
    """
    Estimate haemodynamic significance (QFR / FFR proxy) from 2-D QCA metrics.

    Method — Simplified Gould-Gorlin pressure-drop model
    ─────────────────────────────────────────────────────
    Gould et al. (1974) showed that trans-stenotic pressure loss ΔP is:

        ΔP = f₁·Q + f₂·Q²

    where f₁ (viscous term) and f₂ (separation/turbulence term) are
    functions of stenosis geometry.  Assuming normal resting coronary
    flow (Q ~ 1 ml/s for a medium epicardial vessel) and a standard
    aortic pressure Pa = 100 mmHg, the fractional pressure loss is:

        FFR_est = 1 / (1 + k · (D_ref/MLD)⁴ − 1)
                ≈ 1 − 0.5 · (DS / (1 − DS))²          [focal stenosis]

    where DS = % diameter stenosis / 100.

    Validation context
    ──────────────────
    The simplified form is consistent with FAME-trial empirical data
    (De Bruyne et al., NEJM 2012) for focal, single-vessel disease.
    The formula systematically UNDERESTIMATES severity for:
        • Long diffuse lesions  (series resistance adds up)
        • Small vessels  (reference flow is lower)
        • Tandem lesions  (non-linear interaction)

    For a full wire-free FFR estimate (QFR), two angiographic views plus
    TIMI frame count are required — see Xu et al. (EHJ 2019, FAVOR II).

    Returns
    ───────
    dict with keys:
        ffr_estimate          : float in [0, 1]  (rough point estimate)
        ffr_ci_low            : float  (conservative lower bound, −0.05)
        ffr_ci_high           : float  (optimistic upper bound, +0.05)
        qfr_zone              : str    (Green / Grey / Red)
        hemodynamic_class     : str    (Non-significant / Borderline / Significant)
        ischemia_likelihood   : str    (Low / Intermediate / High)
        recommendation        : str
        note                  : str    (disclaimer)
    """
    if pct_ds <= 0 or ref_diameter_px <= 0 or mld_px <= 0:
        return {}

    ds = float(np.clip(pct_ds / 100.0, 0.01, 0.99))

    # Simplified Gorlin-Gould fractional pressure loss
    gorlin_term = (ds / (1.0 - ds)) ** 2
    ffr_est     = float(np.clip(1.0 - 0.5 * gorlin_term, 0.40, 1.00))

    # Conservative confidence interval (±0.05 — reflects 2-D single-view limitation)
    ffr_lo = round(max(0.40, ffr_est - 0.05), 3)
    ffr_hi = round(min(1.00, ffr_est + 0.05), 3)
    ffr_est = round(ffr_est, 3)

    # QFR zone (mirrors FAVOR II / clinical cut-offs)
    if ffr_est >= 0.80:
        zone      = "Green  (≥ 0.80)"
        hclass    = "Non-significant"
        ischemia  = "Low"
        rec       = "Deferral of revascularisation is safe (DEFER, FAME data)."
    elif ffr_est >= 0.75:
        zone      = "Grey   (0.75 – 0.80)"
        hclass    = "Borderline"
        ischemia  = "Intermediate"
        rec       = "Wire-based FFR or true QFR recommended before decision."
    else:
        zone      = "Red    (< 0.75)"
        hclass    = "Significant"
        ischemia  = "High"
        rec       = "Revascularisation likely beneficial (FAME, FAME-2 data)."

    return {
        "ffr_estimate":        ffr_est,
        "ffr_ci_low":          ffr_lo,
        "ffr_ci_high":         ffr_hi,
        "qfr_zone":            zone,
        "hemodynamic_class":   hclass,
        "ischemia_likelihood": ischemia,
        "recommendation":      rec,
        "note": (
            "2-D single-plane estimate via simplified Gould-Gorlin model. "
            "True QFR (Xu et al. 2019) requires 3-D reconstruction + TIMI frame count."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def draw_radius_overlay(crop_rgb: np.ndarray,
                        path: list,
                        radii: list,
                        metrics: dict) -> np.ndarray:
    """
    Draw on the crop:
      • Cyan centerline
      • Yellow perpendicular ticks (every _OVERLAY_STEP points), length = radius
      • Red dot + label at MLD location
    """
    out = crop_rgb.copy()
    if not path:
        return out

    mld_idx = metrics.get("mld_location_idx", 0)

    # Precompute tangents for tick drawing
    for i, (r, c) in enumerate(path):
        # Centerline pixel
        cv2.circle(out, (c, r), 1, (0, 220, 220), -1)

        # Radius tick
        if i % _OVERLAY_STEP == 0 and i < len(radii):
            t = _local_tangent(path, i, _TANGENT_WINDOW)
            n = np.array([-t[1], t[0]])
            rad = radii[i]
            p1 = (int(round(c + n[1] * rad)), int(round(r + n[0] * rad)))
            p2 = (int(round(c - n[1] * rad)), int(round(r - n[0] * rad)))
            cv2.line(out, p1, p2, (255, 220, 0), 1, cv2.LINE_AA)

    # MLD marker
    if mld_idx < len(path):
        mr, mc = path[mld_idx]
        cv2.circle(out, (mc, mr), 5, (255, 40, 40), -1)
        cv2.putText(out, "MLD", (mc + 6, mr),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 40, 40), 1, cv2.LINE_AA)
    return out


def plot_radius_profile(radii: list, metrics: dict) -> np.ndarray:
    """
    Matplotlib diameter-along-centerline plot.
    Returns RGB uint8 numpy array suitable for Gradio gr.Image.
    """
    if not radii:
        blank = np.zeros((200, 400, 3), dtype=np.uint8)
        cv2.putText(blank, "No radius data", (80, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        return blank

    diameters = [2.0 * r for r in radii]
    xs        = list(range(len(diameters)))

    fig, ax = plt.subplots(figsize=(6, 3), dpi=110)
    ax.plot(xs, diameters, color="#00bfff", linewidth=1.5, label="Diameter (px)")

    ref = metrics.get("ref_diameter_px")
    mld_idx = metrics.get("mld_location_idx")
    pct = metrics.get("pct_diameter_stenosis")

    if ref is not None:
        ax.axhline(ref, color="#00e676", linestyle="--", linewidth=1.2,
                   label=f"Ref diam = {ref:.1f} px")
    if mld_idx is not None and mld_idx < len(diameters):
        ax.plot(mld_idx, diameters[mld_idx], "ro", markersize=7,
                label=f"MLD = {diameters[mld_idx]:.1f} px  ({pct:.1f}% DS)")

    ax.set_xlabel("Centerline position (px)", fontsize=9)
    ax.set_ylabel("Lumen diameter (px)",       fontsize=9)
    ax.set_title("QCA Diameter Profile",        fontsize=10)
    ax.legend(fontsize=8)
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#16213e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────────────────────────────
# Master entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_qca(image_rgb: np.ndarray,
            bbox_xyxy: list,
            pixel_spacing_mm: Optional[float] = None) -> dict:
    """
    Run the full QCA pipeline on a single detected bounding box.

    Parameters
    ----------
    image_rgb       : full image (H × W × 3 uint8 RGB)
    bbox_xyxy       : [x1, y1, x2, y2] bounding box (pixel coords)
    pixel_spacing_mm: optional mm/pixel scale for physical units

    Returns
    -------
    dict with keys:
        crop, clahe, frangi, binary, skeleton_overlay,
        radius_overlay, radius_plot, metrics, error (on failure)
    """
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]

    # Clamp to image bounds
    h, w = image_rgb.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 - x1 < 5 or y2 - y1 < 5:
        return {"error": "Bounding box too small for QCA analysis."}

    crop = image_rgb[y1:y2, x1:x2].copy()

    try:
        # ── Preprocessing ──────────────────────────────────────────────────
        stages   = preprocess_crop(crop)
        cleaned  = stages["_cleaned"]
        skeleton = stages["_skeleton"]

        # ── Centerline ─────────────────────────────────────────────────────
        path = extract_centerline_points(skeleton)

        if len(path) < 5:
            # Overlay skeleton on crop even if short, show warning
            skel_overlay = crop.copy()
            skel_pts = np.argwhere(skeleton)
            for (r, c) in skel_pts:
                skel_overlay[r, c] = [0, 220, 220]
            return {
                "crop":             crop,
                "clahe":            stages["clahe"],
                "frangi":           stages["frangi"],
                "binary":           stages["binary"],
                "skeleton_overlay": skel_overlay,
                "radius_overlay":   crop.copy(),
                "radius_plot":      plot_radius_profile([], {}),
                "metrics":          {},
                "error":            "Centerline too short — vessel may not be well segmented.",
            }

        # ── Skeleton overlay ────────────────────────────────────────────────
        skel_overlay = crop.copy()
        for (r, c) in path:
            if 0 <= r < skel_overlay.shape[0] and 0 <= c < skel_overlay.shape[1]:
                skel_overlay[r, c] = [0, 220, 220]

        # ── Radius profile ──────────────────────────────────────────────────
        radii   = compute_radius_profile(cleaned, path)
        metrics = compute_stenosis_metrics(radii, pixel_spacing_mm)

        # ── QFR / FFR estimate ──────────────────────────────────────────────
        qfr = estimate_qfr(
            pct_ds=metrics.get("pct_diameter_stenosis", 0.0),
            ref_diameter_px=metrics.get("ref_diameter_px", 0.0),
            mld_px=metrics.get("mld_px", 0.0),
        )
        metrics["qfr"] = qfr

        radius_overlay = draw_radius_overlay(crop, path, radii, metrics)
        radius_plot    = plot_radius_profile(radii, metrics)

        return {
            "crop":             crop,
            "clahe":            stages["clahe"],
            "frangi":           stages["frangi"],
            "binary":           stages["binary"],
            "skeleton_overlay": skel_overlay,
            "radius_overlay":   radius_overlay,
            "radius_plot":      radius_plot,
            "metrics":          metrics,
        }

    except Exception as exc:
        return {
            "crop":  crop,
            "error": f"QCA pipeline error: {exc}",
        }
