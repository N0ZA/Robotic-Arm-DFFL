import pyrealsense2 as rs
import numpy as np
import cv2
from dataclasses import dataclass
import time
import math

# ============================================================
# CONFIGURATION AND CONSTANTS
# ============================================================

W = 1280
H = 720
FPS = 30

# RANSAC for table plane
TABLE_INLIER_THRESH_M = 0.005
TABLE_RANSAC_ITERS = 220
FOREGROUND_DIST_FROM_TABLE_M = 0.004  # <<< CHANGED: smaller, keep more points for lying cups

# Upright cup parameters
ELL_MIN_AREA_PX = 1200
ELL_MAX_ASPECT = 4.0
UPRIGHT_SCORE_THRESH = 0.45

# 3D point cloud & clustering for lying cups
GRID_STEP = 4
VOXEL_SIZE_M = 0.004
DBSCAN_EPS_M = 0.030       # <<< CHANGED: slightly tight but okay for tabletop scale
DBSCAN_MIN_PTS = 40        # <<< CHANGED: lower to catch sparse transparent clusters
LYING_SCORE_THRESH = 0.30  # <<< CHANGED: allow weaker lying detections

# Multi-modal transparency score weights
W_HOLES = 0.45
W_IR = 0.35
W_GEOM = 0.20

DRAW_MAX = 10
WINDOW_NAME = "Transparent Cup Detection (Camera View)"

# ============================================================
# DATA STRUCTURE FOR DETECTION RESULT
# ============================================================

@dataclass
class CupHypothesis:
    pose_type: str                 # "UPRIGHT" or "LYING"
    centroid_cam_xyz: np.ndarray   # 3D centroid [x, y, z] in camera frame
    score: float                   # Transparency+geometry confidence
    bbox_px: tuple                 # (x_min, y_min, x_max, y_max)
    meta: dict                     # Extra info

# ============================================================
# CUP SIZE MODEL (AUTO-ESTIMATION, NO HARD-CODED SIZE)
# ============================================================

class CupSizeModel:
    """
    Maintains a running estimate of typical cup diameter and height
    from detections themselves (both upright and lying). No fixed cm values.
    """

    def __init__(self):
        self.ref_diam_m = None
        self.ref_height_m = None
        self.n_samples = 0

    def _update_running(self, value_d, value_h):
        # Exponential moving average for stability
        alpha = 0.3
        if self.ref_diam_m is None or self.ref_height_m is None:
            self.ref_diam_m = float(value_d)
            self.ref_height_m = float(value_h)
        else:
            self.ref_diam_m = (1 - alpha) * self.ref_diam_m + alpha * float(value_d)
            self.ref_height_m = (1 - alpha) * self.ref_height_m + alpha * float(value_h)
        self.n_samples += 1

    def update_from_upright(self, ellipse, centroid_cam, intr, table_plane):
        """
        Estimate diameter & height for an upright cup from ellipse + table plane.
        """
        if table_plane is None:
            return

        (cx, cy), (ew, eh), ang = ellipse
        # Approximate diameter from ellipse size and depth at center
        # Try to get depth at center via plane-ray intersection (more robust if depth missing)
        n, dpl = table_plane
        ray = pixel_ray(intr, cx, cy)
        p_table = ray_plane_intersect(n, dpl, ray)
        if p_table is None:
            return

        # Approx distance from camera to rim
        z_est = float(np.linalg.norm(centroid_cam))  # magnitude as rough range
        fx = intr.fx
        # Take largest axis as rim diameter in pixels, convert to meters
        pix_diam = max(ew, eh)
        if fx <= 0 or z_est <= 0:
            return
        diam_est = (pix_diam / fx) * z_est

        # Height: distance from rim centroid to table along plane normal
        dist_to_plane = abs(point_plane_dist(n, dpl, centroid_cam))
        h_est = dist_to_plane

        if diam_est <= 0 or h_est <= 0:
            return

        self._update_running(diam_est, h_est)

    def update_from_cluster(self, extents):
        """
        Estimate diameter & height from PCA extents of lying cup cluster.
        Longest axis ≈ height (cup length),
        max of other axes ≈ diameter.
        """
        long_len = float(extents[0])
        mid_len = float(extents[1])
        short_len = float(extents[2])
        diam_est = max(mid_len, short_len)
        h_est = long_len
        if diam_est <= 0 or h_est <= 0:
            return
        self._update_running(diam_est, h_est)

    def geometry_score(self, diam_est, h_est):
        """
        Score how consistent this (diameter, height) pair is with the learned model.
        1.0 = perfect match, ~0.0 = very different. If no model yet, return neutral 0.5.
        """
        if self.ref_diam_m is None or self.ref_height_m is None or self.n_samples < 2:
            return 0.5  # neutral before we have a stable model

        # Relative errors
        rel_d = abs(diam_est - self.ref_diam_m) / (self.ref_diam_m + 1e-9)
        rel_h = abs(h_est - self.ref_height_m) / (self.ref_height_m + 1e-9)

        # Tolerate ~50–60% variation; beyond that, geometry score decays
        sigma = 0.6
        s_d = math.exp(- (rel_d ** 2) / (2 * sigma ** 2))
        s_h = math.exp(- (rel_h ** 2) / (2 * sigma ** 2))

        return (s_d + s_h) * 0.5

# ============================================================
# BASIC GEOMETRY UTILITIES
# ============================================================

def normalize(v):
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)

def depth_to_m(depth_u16, depth_scale):
    return depth_u16.astype(np.float32) * float(depth_scale)

def deproject(intr, u, v, z_m):
    x, y, z = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], float(z_m))
    return np.array([x, y, z], dtype=np.float32)

def project(intr, p_cam):
    uv = rs.rs2_project_point_to_pixel(intr, [float(p_cam[0]), float(p_cam[1]), float(p_cam[2])])
    return np.array([uv[0], uv[1]], dtype=np.float32)

def pixel_ray(intr, u, v):
    p = deproject(intr, u, v, 1.0)
    return normalize(p)

def fit_plane_from_points(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1, v2)
    nn = float(np.linalg.norm(n))
    if nn < 1e-9:
        return None
    n = (n / nn).astype(np.float32)
    d = -float(np.dot(n, p1))
    return n, d

def point_plane_dist(n, d, p):
    return float(np.dot(n, p) + d)

def ray_plane_intersect(n, d, ray_dir):
    denom = float(np.dot(n, ray_dir))
    if abs(denom) < 1e-9:
        return None
    t = -(d) / denom
    if t <= 0:
        return None
    return (t * ray_dir).astype(np.float32)

# ============================================================
# CAMERA SETUP (REAL DATA SOURCE)
# ============================================================

def start_d457():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    config.enable_stream(rs.stream.infrared, 1, W, H, rs.format.y8, FPS)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    return pipeline, align, depth_scale, intr

def get_frames(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)

    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    ir_frame = frames.get_infrared_frame(1)

    if not color_frame or not depth_frame or not ir_frame:
        return None, None, None

    color = np.asanyarray(color_frame.get_data())
    depth = np.asanyarray(depth_frame.get_data())
    ir = np.asanyarray(ir_frame.get_data())

    return color, depth, ir

# ============================================================
# TABLE PLANE ESTIMATION
# ============================================================

def calibrate_table_plane(depth_u16, intr, depth_scale):
    z = depth_to_m(depth_u16, depth_scale)
    hh, ww = z.shape
    ys, xs = np.where(z > 0.0)
    if len(xs) < 5000:
        return None

    pick = np.random.choice(len(xs), size=min(20000, len(xs)), replace=False)
    xs_s = xs[pick]
    ys_s = ys[pick]
    zs_s = z[ys_s, xs_s]
    pts = np.stack([xs_s, ys_s, zs_s], axis=1)

    best_inliers = -1
    best_plane = None

    for _ in range(TABLE_RANSAC_ITERS):
        idx = np.random.choice(len(pts), size=3, replace=False)
        u1, v1, z1 = pts[idx[0]]
        u2, v2, z2 = pts[idx[1]]
        u3, v3, z3 = pts[idx[2]]

        p1 = deproject(intr, u1, v1, z1)
        p2 = deproject(intr, u2, v2, z2)
        p3 = deproject(intr, u3, v3, z3)

        plane = fit_plane_from_points(p1, p2, p3)
        if plane is None:
            continue

        n, d = plane
        inliers = 0
        for j in range(0, len(pts), 6):
            uj, vj, zj = pts[j]
            pj = deproject(intr, uj, vj, zj)
            if abs(point_plane_dist(n, d, pj)) < TABLE_INLIER_THRESH_M:
                inliers += 1

        if inliers > best_inliers:
            best_inliers = inliers
            best_plane = (n, d)

    return best_plane

# ============================================================
# UPRIGHT CUP DETECTION
# ============================================================

def preprocess_edges(color_bgr):
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 40, 40)
    edges = cv2.Canny(blur, 60, 140)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    return edges

def ellipse_candidates(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < ELL_MIN_AREA_PX:
            continue
        if len(cnt) < 20:
            continue
        ellipse = cv2.fitEllipse(cnt)
        (cx, cy), (ew, eh), ang = ellipse
        if ew < 25 or eh < 25:
            continue
        aspect = max(ew, eh) / (min(ew, eh) + 1e-6)
        if aspect > ELL_MAX_ASPECT:
            continue
        candidates.append(ellipse)

    return candidates

def ellipse_mask(shape_hw, ellipse, scale=1.0):
    hh, ww = shape_hw
    mask = np.zeros((hh, ww), dtype=np.uint8)
    (cx, cy), (ew, eh), ang = ellipse
    ew_scaled = ew * scale
    eh_scaled = eh * scale
    cv2.ellipse(mask,
                (int(cx), int(cy)),
                (int(ew_scaled / 2), int(eh_scaled / 2)),
                ang,
                0, 360,
                255,
                -1)
    return mask

def ring_mask(shape_hw, ellipse, inner=0.85, outer=1.10):
    outer_m = ellipse_mask(shape_hw, ellipse, scale=outer)
    inner_m = ellipse_mask(shape_hw, ellipse, scale=inner)
    ring = cv2.subtract(outer_m, inner_m)
    return ring

def score_ellipse_transparency(ellipse, depth_u16, ir_u8):
    hh, ww = depth_u16.shape
    inside = ellipse_mask((hh, ww), ellipse, scale=0.90)
    ring = ring_mask((hh, ww), ellipse, inner=0.85, outer=1.10)

    d_in = depth_u16[inside > 0].astype(np.float32)
    ir_in = ir_u8[inside > 0].astype(np.float32)
    ir_ring = ir_u8[ring > 0].astype(np.float32)

    if d_in.size < 50 or ir_in.size < 50 or ir_ring.size < 50:
        return 0.0, {"reason": "too_few_pixels"}

    holes = float(np.mean(d_in == 0.0))
    s_holes = float(np.clip(holes, 0.0, 1.0))

    ir_in_mean = float(np.mean(ir_in))
    ir_ring_mean = float(np.mean(ir_ring) + 1e-6)
    ir_drop = float(np.clip((ir_ring_mean - ir_in_mean) / ir_ring_mean, 0.0, 1.0))
    s_ir = ir_drop

    (cx, cy), (ew, eh), ang = ellipse
    norm_area = float((ew * eh) / (ww * hh))
    s_geom = float(np.exp(-((norm_area - 0.03) ** 2) / (2 * (0.02 ** 2))))

    score = float(W_HOLES * s_holes + W_IR * s_ir + W_GEOM * s_geom)

    meta = {
        "holes_ratio": holes,
        "ir_in_mean": ir_in_mean,
        "ir_ring_mean": ir_ring_mean,
        "ir_drop": ir_drop,
        "geom_score": s_geom
    }
    return score, meta

def estimate_upright_centroid_3d(ellipse, depth_u16, depth_scale, intr, table_plane):
    hh, ww = depth_u16.shape
    (cx, cy), (ew, eh), ang = ellipse

    num_samples = 90
    theta = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)

    a = (ew / 2.0) * 0.98
    b = (eh / 2.0) * 0.98
    A = np.deg2rad(ang)
    cosA, sinA = np.cos(A), np.sin(A)

    rim_points_3d = []

    for t in theta:
        ex = a * np.cos(t)
        ey = b * np.sin(t)

        u = cx + ex * cosA - ey * sinA
        v = cy + ex * sinA + ey * cosA

        ui = int(round(u))
        vi = int(round(v))

        if ui < 0 or ui >= ww or vi < 0 or vi >= hh:
            continue

        d = depth_u16[vi, ui]
        if d == 0:
            continue

        z_m = float(d) * float(depth_scale)
        rim_points_3d.append(deproject(intr, ui, vi, z_m))

    rim_points_3d = np.array(rim_points_3d, dtype=np.float32)
    if rim_points_3d.shape[0] >= 18:
        centroid = np.mean(rim_points_3d, axis=0).astype(np.float32)
        return True, centroid, {"centroid_mode": "rim_depth"}

    if table_plane is None:
        return False, np.zeros(3, dtype=np.float32), {"centroid_mode": "no_table_plane"}

    n, dpl = table_plane
    ray = pixel_ray(intr, cx, cy)
    p_table = ray_plane_intersect(n, dpl, ray)
    if p_table is None:
        return False, np.zeros(3, dtype=np.float32), {"centroid_mode": "ray_parallel"}

    centroid = p_table.astype(np.float32)
    return True, centroid, {"centroid_mode": "table_fallback"}

def ellipse_bounding_box(ellipse):
    (cx, cy), (ew, eh), ang = ellipse
    x_min = int(round(cx - ew / 2.0))
    x_max = int(round(cx + ew / 2.0))
    y_min = int(round(cy - eh / 2.0))
    y_max = int(round(cy + eh / 2.0))
    return x_min, y_min, x_max, y_max

def detect_upright_cups(color, depth_u16, ir_u8, depth_scale, intr, table_plane, size_model):
    edges = preprocess_edges(color)
    ellipses = ellipse_candidates(edges)
    hypotheses = []

    for ell in ellipses:
        score, meta_s = score_ellipse_transparency(ell, depth_u16, ir_u8)
        if score < UPRIGHT_SCORE_THRESH:
            continue

        ok, centroid, meta_c = estimate_upright_centroid_3d(ell, depth_u16, depth_scale, intr, table_plane)
        if not ok:
            continue

        # Estimate diameter & height for size model (auto-scale)
        if table_plane is not None:
            size_model.update_from_upright(ell, centroid, intr, table_plane)

        bbox = ellipse_bounding_box(ell)

        hypotheses.append(
            CupHypothesis(
                pose_type="UPRIGHT",
                centroid_cam_xyz=centroid,
                score=score,
                bbox_px=bbox,
                meta={
                    "ellipse": ell,
                    **meta_s,
                    **meta_c
                }
            )
        )

    return hypotheses

# ============================================================
# LYING CUP DETECTION (3D)
# ============================================================

def depth_to_points_on_grid(depth_u16, intr, depth_scale, step):
    z = depth_to_m(depth_u16, depth_scale)
    hh, ww = z.shape
    points = []
    for v in range(0, hh, step):
        for u in range(0, ww, step):
            zz = float(z[v, u])
            if zz <= 0.0:
                continue
            points.append(deproject(intr, u, v, zz))
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.array(points, dtype=np.float32)

def voxel_downsample(points, voxel_size):
    if points.shape[0] == 0:
        return points
    q = np.floor(points / float(voxel_size)).astype(np.int32)
    _, idx = np.unique(q, axis=0, return_index=True)
    return points[idx]

def extract_foreground_points(points, table_plane):
    if table_plane is None or points.shape[0] == 0:
        return points
    n, d = table_plane
    dist = (points @ n) + float(d)
    keep = np.abs(dist) > FOREGROUND_DIST_FROM_TABLE_M
    return points[keep]

def dbscan(points, eps, min_pts):
    N = points.shape[0]
    if N == 0:
        return np.full((0,), -1, dtype=np.int32)

    labels = np.full((N,), -1, dtype=np.int32)
    visited = np.zeros((N,), dtype=bool)
    cluster_id = 0

    def neighbors(i):
        d = np.linalg.norm(points - points[i], axis=1)
        return np.where(d <= float(eps))[0]

    for i in range(N):
        if visited[i]:
            continue
        visited[i] = True
        nb = neighbors(i)
        if nb.size < int(min_pts):
            labels[i] = -1
            continue

        labels[i] = cluster_id
        seeds = list(nb)

        while seeds:
            j = seeds.pop()
            if not visited[j]:
                visited[j] = True
                nbj = neighbors(j)
                if nbj.size >= int(min_pts):
                    seeds.extend(list(nbj))
            if labels[j] == -1:
                labels[j] = cluster_id

        cluster_id += 1

    return labels

def pca_axis(points):
    centroid = np.mean(points, axis=0)
    X = points - centroid
    C = (X.T @ X) / max(1, X.shape[0])
    w, V = np.linalg.eigh(C)
    idx = np.argsort(w)[::-1]
    V = V[:, idx]
    axis = normalize(V[:, 0])
    proj = X @ V
    mins = np.min(proj, axis=0)
    maxs = np.max(proj, axis=0)
    extents = (maxs - mins).astype(np.float32)
    return centroid.astype(np.float32), axis.astype(np.float32), extents

def local_ir_contrast_score(ir_u8, uv, radius_px=22):
    hh, ww = ir_u8.shape
    u, v = int(round(uv[0])), int(round(uv[1]))
    if u < 0 or u >= ww or v < 0 or v >= hh:
        return 0.5
    u0 = max(0, u - radius_px)
    u1 = min(ww, u + radius_px + 1)
    v0 = max(0, v - radius_px)
    v1 = min(hh, v + radius_px + 1)
    patch = ir_u8[v0:v1, u0:u1].astype(np.float32)
    if patch.size < 50:
        return 0.5
    center_val = float(ir_u8[v, u])
    mean_val = float(np.mean(patch) + 1e-6)
    drop = float(np.clip((mean_val - center_val) / mean_val, 0.0, 1.0))
    return drop

def score_cluster_transparency(cluster_points, ir_u8, intr, size_model):
    centroid, axis, ext = pca_axis(cluster_points)
    long_len = float(ext[0])
    mid_len = float(ext[1])
    short_len = float(ext[2])

    diam_est = float(max(mid_len, short_len))
    h_est = float(long_len)

    # Update size model from cluster
    size_model.update_from_cluster(ext)

    # Geometry score based on learned cup size (no fixed bounds)
    s_geom = size_model.geometry_score(diam_est, h_est)

    # Density-based "holes" proxy
    volume_approx = long_len * (diam_est ** 2) + 1e-9
    density = float(cluster_points.shape[0] / volume_approx)
    s_holes = float(np.clip(1.0 - density / 2e6, 0.0, 1.0))

    uv = project(intr, centroid)
    s_ir = float(local_ir_contrast_score(ir_u8, uv, radius_px=22))

    score = float(W_HOLES * s_holes + W_IR * s_ir + W_GEOM * s_geom)

    meta = {
        "diam_est": diam_est,
        "h_est": h_est,
        "density": density,
        "geom_score": s_geom,
        "holes_score": s_holes,
        "ir_score": s_ir
    }
    return score, axis, centroid, meta

def cluster_bounding_box(cluster_points, intr):
    if cluster_points.shape[0] == 0:
        return (0, 0, 0, 0)
    uvs = []
    for p in cluster_points:
        uv = project(intr, p)
        uvs.append(uv)
    uvs = np.array(uvs)
    u_min = int(np.min(uvs[:, 0]))
    u_max = int(np.max(uvs[:, 0]))
    v_min = int(np.min(uvs[:, 1]))
    v_max = int(np.max(uvs[:, 1]))
    return u_min, v_min, u_max, v_max

def detect_lying_cups(depth_u16, ir_u8, depth_scale, intr, table_plane, size_model):
    pts = depth_to_points_on_grid(depth_u16, intr, depth_scale, step=GRID_STEP)
    pts = voxel_downsample(pts, VOXEL_SIZE_M)
    pts = extract_foreground_points(pts, table_plane)

    labels = dbscan(pts, DBSCAN_EPS_M, DBSCAN_MIN_PTS)
    if labels.shape[0] == 0:
        return []

    hypotheses = []
    for cid in np.unique(labels):
        if cid < 0:
            continue
        cluster = pts[labels == cid]
        if cluster.shape[0] < DBSCAN_MIN_PTS:
            continue

        score, axis, centroid, meta_s = score_cluster_transparency(cluster, ir_u8, intr, size_model)
        if score < LYING_SCORE_THRESH:
            continue

        bbox = cluster_bounding_box(cluster, intr)
        hypotheses.append(
            CupHypothesis(
                pose_type="LYING",
                centroid_cam_xyz=centroid,
                score=score,
                bbox_px=bbox,
                meta={
                    "cluster_id": int(cid),
                    "axis": axis,
                    **meta_s
                }
            )
        )
    return hypotheses

# ============================================================
# MERGING & DUPLICATE SUPPRESSION
# ============================================================

def merge_hypotheses(upright_list, lying_list, dist_thresh=0.06):
    all_h = upright_list[:]
    used_l = set()

    for u in upright_list:
        for i, l in enumerate(lying_list):
            if i in used_l:
                continue
            dist = float(np.linalg.norm(u.centroid_cam_xyz - l.centroid_cam_xyz))
            if dist < float(dist_thresh):
                if l.score > u.score:
                    if u in all_h:
                        all_h.remove(u)
                    all_h.append(l)
                used_l.add(i)

    for i, l in enumerate(lying_list):
        if i not in used_l:
            all_h.append(l)

    all_h.sort(key=lambda c: c.score, reverse=True)
    return all_h

def bbox_iou(b1, b2):
    x1_min, y1_min, x1_max, y1_max = b1
    x2_min, y2_min, x2_max, y2_max = b2

    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)

    if xi_max <= xi_min or yi_max <= yi_min:
        return 0.0

    inter = (xi_max - xi_min) * (yi_max - yi_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - inter + 1e-9
    return inter / union

def suppress_duplicates(cups, iou_thresh=0.6, dist_thresh=0.03):
    """
    Non-maximum suppression:
    - If two cups have high IoU and close 3D centroids (e.g. stepped upside-down rim),
      keep only the higher-score one.
    """
    if not cups:
        return []

    cups_sorted = sorted(cups, key=lambda c: c.score, reverse=True)
    kept = []

    for c in cups_sorted:
        keep = True
        for k in kept:
            iou = bbox_iou(c.bbox_px, k.bbox_px)
            dist = float(np.linalg.norm(c.centroid_cam_xyz - k.centroid_cam_xyz))
            if iou > iou_thresh and dist < dist_thresh:
                keep = False
                break
        if keep:
            kept.append(c)

    return kept

# ============================================================
# VISUALIZATION & OUTPUT
# ============================================================

def draw_bounding_box_and_label(img, cup):
    x_min, y_min, x_max, y_max = cup.bbox_px
    h, w = img.shape[:2]
    x_min = max(0, min(w - 1, x_min))
    x_max = max(0, min(w - 1, x_max))
    y_min = max(0, min(h - 1, y_min))
    y_max = max(0, min(h - 1, y_max))

    color = (0, 255, 0) if cup.pose_type == "UPRIGHT" else (0, 255, 255)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
    label = f"{cup.pose_type} {cup.score:.2f}"
    cv2.putText(img, label, (x_min, max(0, y_min - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def print_cup_info(cups):
    print("========================================")
    print("Detected cups:", len(cups))
    for idx, cup in enumerate(cups):
        print(f" Cup {idx}:")
        print(f"   Type: {cup.pose_type}")
        print(f"   Score: {cup.score:.3f}")
        print(f"   3D centroid [m]: {cup.centroid_cam_xyz}")
        print(f"   BBox (px): {cup.bbox_px}")
        for key in ["holes_ratio", "ir_drop", "geom_score", "diam_est", "h_est", "centroid_mode"]:
            if key in cup.meta:
                print(f"   {key}: {cup.meta[key]}")
    print("========================================")

# ============================================================
# MAIN LOOP
# ============================================================

def main():
    pipeline, align, depth_scale, intr = start_d457()
    table_plane = None
    last_plane_t = 0.0
    size_model = CupSizeModel()  # <<< NEW: auto cup size estimator

    try:
        while True:
            color, depth_u16, ir_u8 = get_frames(pipeline, align)
            if color is None:
                continue

            now = time.time()
            if table_plane is None or (now - last_plane_t) > 3.0:
                plane = calibrate_table_plane(depth_u16, intr, depth_scale)
                if plane is not None:
                    table_plane = plane
                    last_plane_t = now

            upright_cups = detect_upright_cups(
                color, depth_u16, ir_u8, depth_scale, intr, table_plane, size_model
            )
            lying_cups = detect_lying_cups(
                depth_u16, ir_u8, depth_scale, intr, table_plane, size_model
            )

            cups = merge_hypotheses(upright_cups, lying_cups)
            cups = suppress_duplicates(cups)  # <<< NEW: merge double detections (e.g. stepped rims)

            vis = color.copy()
            for cup in cups[:DRAW_MAX]:
                draw_bounding_box_and_label(vis, cup)

            cv2.imshow(WINDOW_NAME, vis)

            if len(cups) > 0:
                print_cup_info(cups)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
