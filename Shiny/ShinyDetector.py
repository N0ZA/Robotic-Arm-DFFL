"""
realsense_shiny_detector.py

Real-time detector that:
 - captures color (+ optional depth) frames from an Intel RealSense camera
 - applies a simplified version of the edge-based "shiny" preprocessing from multiShiny.ipynb
 - runs FastSAM (FastSAM-x.pt) to segment shiny objects
 - selects and displays only the single detected object whose centroid is closest to the screen center

Usage:
  python realsense_shiny_detector.py

Requirements (select from your environment):
  pip install pyrealsense2 opencv-python ultralytics numpy

Notes:
 - Adjust MODEL_PATH if you keep the model in a non-default location. The repository contains `Shiny/FastSAM-x.pt`.
 - If no RealSense camera is available, the script will fail to start the pipeline.
"""

import time
import argparse
import numpy as np
import cv2

try:
    import pyrealsense2 as rs
except Exception:
    rs = None

from ultralytics import FastSAM


def build_arg_parser():
    p = argparse.ArgumentParser(description="RealSense FastSAM shiny object detector")
    p.add_argument("--model", default="Shiny/FastSAM-x.pt", help="Path to FastSAM model (default: Shiny/FastSAM-x.pt)")
    p.add_argument("--width", type=int, default=640, help="Color frame width")
    p.add_argument("--height", type=int, default=360, help="Color frame height")
    p.add_argument("--fps", type=int, default=30, help="Camera FPS")
    p.add_argument("--edge-method", choices=("sobel","farid"), default="farid", help="Edge operator for preprocessing")
    p.add_argument("--display-scale", type=float, default=1.0, help="Scale factor for display window")
    return p


def farid_operator(img):
    # img: single-channel float or uint8
    farid_x = np.array([
        [-1.0, 0.0, 1.0],
        [-np.sqrt(2), 0.0, np.sqrt(2)],
        [-1.0, 0.0, 1.0]
    ], dtype=np.float32)
    farid_y = farid_x.T
    gx = cv2.filter2D(img, cv2.CV_64F, farid_x)
    gy = cv2.filter2D(img, cv2.CV_64F, farid_y)
    return gx, gy


def sobel_operator(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return gx, gy


def preprocess_shiny_mask(gray, edge_method="farid"):
    """Return a 1-channel uint8 mask (0 or 255) indicating candidate shiny regions.

    Steps (light adaptation of notebook):
    - small gaussian blur
    - compute gradients (Farid or Sobel)
    - gradient magnitude -> normalize
    - binary threshold -> find contours -> filled mask
    - mask + blur -> final threshold
    """
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    if edge_method == "farid":
        gx, gy = farid_operator(blur)
    else:
        gx, gy = sobel_operator(blur)

    gradient_magnitude = cv2.magnitude(gx, gy)
    # normalize to 0-255
    magnitude_norm = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # coarse threshold to find edges/specular candidates
    _, binary_edge = cv2.threshold(magnitude_norm, 60, 255, cv2.THRESH_BINARY)

    # find contours and fill them to create mask region
    contours, _ = cv2.findContours(binary_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_mask = np.zeros_like(gray, dtype=np.uint8)
    if len(contours) > 0:
        cv2.drawContours(contour_mask, contours, -1, 255, thickness=-1)

    # apply mask to grayscale, blur region and threshold to obtain bright metallic regions
    masked_gray = cv2.bitwise_and(gray, gray, mask=contour_mask)
    blurred = cv2.blur(masked_gray, (12, 12))
    _, final_mask = cv2.threshold(blurred, 37, 255, cv2.THRESH_BINARY)

    return final_mask


def extract_masks_from_results(results):
    """Robust extractor for mask arrays from ultralytics FastSAM results.
    Returns masks as a boolean numpy array with shape (n, H, W) or None.
    """
    if not results:
        return None
    r = results[0]
    # Many ultralytics versions expose r.masks.data as (n, H, W)
    masks = None
    if hasattr(r, 'masks'):
        masks_obj = r.masks
        # try common attributes in order
        if hasattr(masks_obj, 'data'):
            masks = np.asarray(masks_obj.data)
        elif hasattr(masks_obj, 'masks'):
            masks = np.asarray(masks_obj.masks)
        else:
            # fallback: try direct conversion
            try:
                masks = np.asarray(masks_obj)
            except Exception:
                masks = None
    return masks


def choose_mask_closest_to_center(masks, image_shape):
    """Given masks (n,H,W) boolean/uint8 and image_shape (H,W), return index of mask whose centroid
    is closest to the image center. Returns -1 if masks is empty.
    """
    if masks is None or len(masks) == 0:
        return -1

    H, W = image_shape
    cx, cy = W / 2.0, H / 2.0

    best_idx = -1
    best_dist = float('inf')

    for i in range(len(masks)):
        m = masks[i].astype(np.uint8)
        # compute centroid robustly
        ys, xs = np.where(m > 0)
        if len(xs) == 0:
            continue
        mx = xs.mean()
        my = ys.mean()
        d = (mx - cx) ** 2 + (my - cy) ** 2
        if d < best_dist:
            best_dist = d
            best_idx = i

    return best_idx


def main():
    args = build_arg_parser().parse_args()

    # Load FastSAM model
    print(f"Loading FastSAM model from: {args.model}")
    model = FastSAM(args.model)
    print("Model loaded.")

    # Setup RealSense
    if rs is None:
        print("pyrealsense2 not found. Please install pyrealsense2 to use a RealSense camera.")
        return

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    # optional depth stream (not used for selection by default)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)

    win_name = "RealSense Shiny Detector"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    try:
        last_time = time.time()
        frame_count = 0
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            # match notebook: scale up for visualization if needed
            # but here keep original size

            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

            # create candidate shiny mask
            shiny_mask = preprocess_shiny_mask(gray, edge_method=args.edge_method)

            # prepare masked color image for model
            masked_color = cv2.bitwise_and(color, color, mask=shiny_mask)

            # ensure contiguous array and BGR->RGB if required by model
            input_for_model = masked_color.copy()

            # Run model - we pass the full masked color image
            try:
                results = model(input_for_model)
            except Exception as e:
                print("Model inference error:", e)
                results = None

            masks = extract_masks_from_results(results)

            selected_idx = -1
            if masks is not None and len(masks) > 0:
                # Masks may be boolean or 0/255. Ensure boolean
                masks_bool = (masks > 0)
                selected_idx = choose_mask_closest_to_center(masks_bool, (args.height, args.width))

            overlay = color.copy()
            info_text = "No detection"
            if selected_idx >= 0:
                m = masks[selected_idx]
                # make uint8 single channel
                m_u8 = (m > 0).astype(np.uint8) * 255
                # create colored overlay (green)
                overlay[m_u8 == 255] = (0, 255, 0)

                # compute bounding box from mask
                ys, xs = np.where(m_u8 == 255)
                if len(xs) > 0:
                    x0, x1 = int(xs.min()), int(xs.max())
                    y0, y1 = int(ys.min()), int(ys.max())
                    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)
                    info_text = f"Selected object idx={selected_idx} bbox=({x0},{y0},{x1},{y1})"

            # show small preview of mask at the right side
            mask_rgb = cv2.cvtColor(shiny_mask, cv2.COLOR_GRAY2BGR)
            h, w = overlay.shape[:2]
            preview_h = int(h * 0.25)
            preview_w = int(w * 0.25)
            small_overlay = cv2.resize(overlay, (preview_w, preview_h))
            small_mask = cv2.resize(mask_rgb, (preview_w, preview_h))

            # compose display image: original with overlay and small mask on top-left corner
            display = overlay.copy()
            # place small mask at top-left
            display[0:preview_h, 0:preview_w] = cv2.addWeighted(display[0:preview_h, 0:preview_w], 0.6, small_mask, 0.4, 0)

            # draw center crosshair
            chx, chy = int(w/2), int(h/2)
            cv2.drawMarker(display, (chx, chy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=1)

            # put text
            cv2.putText(display, info_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # resize for display if requested
            if args.display_scale != 1.0:
                display = cv2.resize(display, (int(w * args.display_scale), int(h * args.display_scale)))

            cv2.imshow(win_name, display)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break

            frame_count += 1
            if frame_count % 30 == 0:
                now = time.time()
                fps = 30.0 / max(1e-6, now - last_time)
                last_time = now
                # print a lightweight status
                print(f"Processed {frame_count} frames")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
