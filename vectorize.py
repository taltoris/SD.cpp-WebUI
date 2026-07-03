"""
vectorize.py  –  Raster → contoured, overlapping-layer SVG
----------------------------------------------------------------
Paint-over model: background layers extend *under* foreground ones.
No puzzle-piece partitioning. Each SVG layer is one connected blob
rendered as smooth cubic Bézier paths.
"""

import sys
import cv2
import numpy as np
from skimage import color as skcolor, filters, morphology
import svgwrite
from sklearn.cluster import KMeans


# ─────────────────────────────────────────────────────────────────────────────
# Bézier / contour helpers
# ─────────────────────────────────────────────────────────────────────────────

def _catmull_rom_cp(p0, p1, p2, p3, alpha=0.5):
    """
    Return cubic Bézier control points (cp1, cp2) for the segment p1→p2,
    using the centripetal Catmull-Rom formulation.

    alpha=0.5  → centripetal  (default; avoids cusps / self-intersections)
    alpha=0.0  → uniform      (classic CR)
    alpha=1.0  → chordal
    """
    def knot(ti, pa, pb):
        d = np.linalg.norm(pb - pa)
        return ti + d ** alpha if d > 1e-8 else ti + 1e-8

    t0 = 0.0
    t1 = knot(t0, p0, p1)
    t2 = knot(t1, p1, p2)
    t3 = knot(t2, p2, p3)

    dt21 = t2 - t1
    if dt21 < 1e-8:
        return p1.copy(), p2.copy()

    m1 = dt21 * (
        (p1 - p0) / (t1 - t0 + 1e-8)
        - (p2 - p0) / (t2 - t0 + 1e-8)
        + (p2 - p1) / (dt21 + 1e-8)
    )
    m2 = dt21 * (
        (p2 - p1) / (dt21 + 1e-8)
        - (p3 - p1) / (t3 - t1 + 1e-8)
        + (p3 - p2) / (t3 - t2 + 1e-8)
    )

    return p1 + m1 / 3.0, p2 - m2 / 3.0


def contour_to_bezier_d(contour, scale_x, scale_y, epsilon=1.5):
    """
    Simplify a raw OpenCV contour and convert to an SVG cubic-Bézier
    path string (no Y-flip; SVG top-left == image top-left).
    Returns None if fewer than 3 points remain after simplification.
    """
    simplified = cv2.approxPolyDP(contour, epsilon=epsilon, closed=True)
    pts = simplified[:, 0, :].astype(np.float64)

    if len(pts) < 3:
        return None

    pts[:, 0] *= scale_x
    pts[:, 1] *= scale_y

    n = len(pts)
    parts = [f"M {pts[0, 0]:.2f} {pts[0, 1]:.2f}"]
    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i % n]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]
        cp1, cp2 = _catmull_rom_cp(p0, p1, p2, p3)
        parts.append(
            f"C {cp1[0]:.2f} {cp1[1]:.2f}, "
            f"{cp2[0]:.2f} {cp2[1]:.2f}, "
            f"{p2[0]:.2f} {p2[1]:.2f}"
        )
    parts.append("Z")
    return " ".join(parts)


def mask_to_compound_path(mask, scale_x, scale_y, epsilon=1.5):
    """
    Convert a binary mask to a compound SVG path string that correctly
    handles holes via the even-odd rule.

    Uses RETR_CCOMP hierarchy:
      level-0 (no parent)          → outer boundary
      level-1 (parent has no parent) → direct hole
    Deeper nesting is skipped; evenodd fill-rule handles it implicitly.

    Returns the 'd' string, or None if no usable contours found.
    """
    contours, hierarchy = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_TC89_KCOS,
    )
    if not contours or hierarchy is None:
        return None

    hier = hierarchy[0]   # (N, 4): [next, prev, first_child, parent]
    d_parts = []

    for idx, contour in enumerate(contours):
        parent = hier[idx][3]
        is_outer = parent == -1
        is_hole  = (parent != -1) and (hier[parent][3] == -1)
        if not (is_outer or is_hole):
            continue
        d = contour_to_bezier_d(contour, scale_x, scale_y, epsilon=epsilon)
        if d:
            d_parts.append(d)

    return " ".join(d_parts) if d_parts else None


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def image_to_layers_svg(
    input_path,
    output_svg_path,
    num_colors=6,
    smoothness_factor=0.5,
    svg_width=512,
    svg_height=512,
    min_blob_area=80,
    simplify_epsilon=1.5,
    depth_mode="y",       # "y" | "luminance"
    kmeans_n_init=3,
    no_blur=False,
):
    """
    Convert a PNG to a contoured, overlapping-layer SVG.

    Parameters
    ----------
    input_path        : source PNG (e.g. SD output).
    output_svg_path   : destination SVG.
    num_colors        : palette size (k-means clusters in Lab space).
    smoothness_factor : 0–1; Gaussian blur sigma + morphological closing radius.
    svg_width/height  : SVG canvas dimensions.
    min_blob_area     : discard blobs smaller than this many pixels.
    simplify_epsilon  : Douglas-Peucker epsilon for contour simplification.
    depth_mode        : layer ordering heuristic.
                         "y"         – lower mean Y  = background (landscapes).
                         "luminance" – higher Lab-L  = foreground (high contrast).
    kmeans_n_init     : KMeans n_init (3 = fast; 10 = higher quality).
    no_blur           : if True, skip blur and closing entirely.
    """

    # ── 1. Load ──────────────────────────────────────────────────────────────
    img_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load: {input_path}")
    h, w = img_bgr.shape[:2]

    # ── 2. Lab ───────────────────────────────────────────────────────────────
    img_rgb_f = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
    img_lab = skcolor.rgb2lab(img_rgb_f)   # kept unblurred for depth metrics

    # ── 3. Blur Lab before clustering (encourages larger, cleaner blobs) ─────
    if no_blur:
        img_lab_blur = img_lab.copy()
    else:
        blur_sigma = 0.5 + 4.5 * smoothness_factor
        img_lab_blur = np.stack([
            filters.gaussian(img_lab[:, :, c], sigma=blur_sigma)
            for c in range(3)
        ], axis=-1)

    # ── 4. K-means on blurred Lab ─────────────────────────────────────────────
    lab_flat = img_lab_blur.reshape(-1, 3)
    kmeans = KMeans(
        n_clusters=num_colors,
        init="k-means++",
        n_init=kmeans_n_init,
        random_state=42,
    )
    label_map = kmeans.fit_predict(lab_flat).reshape(h, w).astype(np.int32)

    # Centroids → RGB for SVG fill
    centroids_rgb = (
        skcolor.lab2rgb(kmeans.cluster_centers_.reshape(1, -1, 3))[0] * 255
    ).clip(0, 255).astype(np.uint8)

    # ── 5. Depth per color (on original, unblurred Lab for accuracy) ──────────
    color_depth = {}
    for cid in range(num_colors):
        px = label_map == cid
        if not px.any():
            color_depth[cid] = 0.0
        elif depth_mode == "luminance":
            color_depth[cid] = float(img_lab[px, 0].mean())
        else:  # "y"
            color_depth[cid] = float(np.where(px)[0].mean())

    color_order = sorted(range(num_colors), key=lambda c: color_depth[c])

    # ── 6. Extract blobs — paint-over model (no used-pixel mask) ─────────────
    if no_blur:
        all_blobs = []
    else:
        close_radius = max(1, int(2 + 4 * smoothness_factor))
        selem = morphology.disk(close_radius)
        all_blobs = []

    for color_id in color_order:
        mask = (label_map == color_id)
        if not no_blur:
            mask_closed = morphology.binary_closing(mask, selem).astype(np.uint8)
        else:
            mask_closed = mask.astype(np.uint8)

        num_cc, cc_map, stats, _ = cv2.connectedComponentsWithStats(
            mask_closed, connectivity=8
        )
        for cc_id in range(1, num_cc):
            area = int(stats[cc_id, cv2.CC_STAT_AREA])
            if area < min_blob_area:
                continue
            all_blobs.append({
                "mask":     (cc_map == cc_id).astype(np.uint8),
                "color":    tuple(int(x) for x in centroids_rgb[color_id]),
                "color_id": color_id,
                "depth":    color_depth[color_id],
                "area":     area,
            })

    all_blobs.sort(key=lambda b: b["depth"])

    # ── 7. SVG ────────────────────────────────────────────────────────────────
    scale_x = svg_width / w
    scale_y = svg_height / h

    dwg = svgwrite.Drawing(output_svg_path, size=(f"{svg_width}px", f"{svg_height}px"))
    dwg.viewbox(0, 0, svg_width, svg_height)

    for i, blob in enumerate(all_blobs):
        d = mask_to_compound_path(
            blob["mask"], scale_x, scale_y, epsilon=simplify_epsilon
        )
        if not d:
            continue
        g = dwg.add(dwg.g(id=f"layer_{i}_c{blob['color_id']}"))
        g.add(dwg.path(
            d=d,
            fill=f"rgb{blob['color']}",
            stroke="none",
            fill_rule="evenodd",
            opacity=1.0,
        ))

    dwg.save()

    # ── 8. Report ─────────────────────────────────────────────────────────────
    print(f"Saved  : {output_svg_path}")
    print(f"Blobs  : {len(all_blobs)}")
    print(f"Colors : {num_colors}")
    print(f"Depth  : {depth_mode}")


# ─────────────────────────────────────────────────────────────────────────────

def raster_to_contour_svg(
    png_path,
    svg_path,
    n_levels=6,
    as_stroke=False,
    smoothness=0.5,
    color_match=True,
    svg_width=512,
    svg_height=512,
    min_blob_area=80,
    simplify_epsilon=1.5,
    depth_mode="y",
    kmeans_n_init=3,
    no_blur=False,
):
    """
    Wrapper around image_to_layers_svg that matches the old raster_to_contour_svg
    signature so the Flask app can call it.

    Parameters
    ----------
    png_path      : source PNG path.
    svg_path      : destination SVG path.
    n_levels      : number of colour clusters (num_colors).
    as_stroke     : if True, render paths as strokes instead of fills.
    smoothness    : 0–1 smoothness factor.
    color_match   : (unused) kept for backward compatibility.
    svg_width     : SVG canvas width in px.
    svg_height    : SVG canvas height in px.
    min_blob_area : discard blobs smaller than this area (px²).
    simplify_epsilon: Douglas-Peucker simplification epsilon.
    depth_mode    : "y" or "luminance".
    kmeans_n_init : KMeans n_init.
    no_blur       : if True, skip blur and morphological closing entirely.
    """
    if isinstance(smoothness, int) and smoothness > 1:
        smoothness = smoothness / 100.0

    image_to_layers_svg(
        input_path=png_path,
        output_svg_path=svg_path,
        num_colors=int(n_levels),
        smoothness_factor=float(smoothness),
        svg_width=int(svg_width),
        svg_height=int(svg_height),
        min_blob_area=int(min_blob_area),
        simplify_epsilon=float(simplify_epsilon),
        depth_mode=str(depth_mode),
        kmeans_n_init=int(kmeans_n_init),
        no_blur=bool(no_blur),
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python segment_layers.py <input.png> [output.svg] [num_colors] [smoothness] [depth_mode]")
        print("  depth_mode: y | luminance   (default: y)")
        sys.exit(1)

    input_path = sys.argv[1]
    output_svg_path = sys.argv[2] if len(sys.argv) > 2 else "layers.svg"
    num_colors = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    smoothness_factor = float(sys.argv[4]) if len(sys.argv) > 4 else 0.6
    depth_mode = sys.argv[5] if len(sys.argv) > 5 else "y"

    image_to_layers_svg(
        input_path=input_path,
        output_svg_path=output_svg_path,
        num_colors=num_colors,
        smoothness_factor=smoothness_factor,
        depth_mode=depth_mode,
        svg_width=512,
        svg_height=512,
        min_blob_area=80,
        simplify_epsilon=1.5,
        kmeans_n_init=3,
        no_blur=False,
    )