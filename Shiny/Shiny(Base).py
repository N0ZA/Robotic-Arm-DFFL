#!/usr/bin/env python3
"""
realsense_metal_detector.py

Detect shiny metallic regions from a RealSense camera in realtime using OpenCV.

Pipeline (per user's spec / Untitled10.ipynb Approach 1):
- Compute Sobel gradients and gradient magnitude
- Normalize magnitude to 0-255 and threshold at 50 -> binary edge map
- Find closed contours and draw filled contour mask
- Mask original grayscale image with contour mask
- Blur masked image with averaging kernel (12x12)
- Threshold blurred image at 45 to produce metallic mask
- Find centroids of metallic regions and query depth from RealSense

Overlay detections and print/display x, y, z for each centroid.

Requires: pyrealsense2, opencv-python, numpy

Usage example:
    python scripts\realsense_metal_detector.py --color --width 1280 --height 720

"""
import argparse
import time
import sys
import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except Exception as e:
    print("Error: can't import pyrealsense2. Install librealsense and the pyrealsense2 wheel.")
    raise


def median_distance_in_window(depth_frame, x, y, window_size=5):
    """Return median distance (meters) inside a square window around (x,y).
    Returns None when no valid depth samples found.
    """
    if depth_frame is None:
        return None

    img_w = depth_frame.get_width()
    img_h = depth_frame.get_height()
    x = int(max(0, min(img_w - 1, x)))
    y = int(max(0, min(img_h - 1, y)))

    half = max(0, window_size // 2)
    x0 = max(0, x - half)
    y0 = max(0, y - half)
    x1 = min(img_w - 1, x + half)
    y1 = min(img_h - 1, y + half)

    distances = []
    for yy in range(y0, y1 + 1):
        for xx in range(x0, x1 + 1):
            try:
                d = depth_frame.get_distance(xx, yy)
            except Exception:
                d = 0.0
            if d and d > 0:
                distances.append(d)

    if not distances:
        return None
    return float(np.median(distances))


# Removed CLI parsing: keep default values directly in the script
# Defaults (previously provided via parse_args):
# width=424, height=240, fps=30, color=False,
# edge_thresh=50, final_thresh=45, blur_k=12, depth_window=5


def main():
    # Use fixed defaults (no CLI parsing)
    args = argparse.Namespace(
        width=480,
        height=270,
        fps=30,
        color=True,
        edge_thresh=50,
        final_thresh=37,
        blur_k=12,
        depth_window=5,
    )

    # ensure odd kernel sizes where needed
    if args.blur_k <= 0:
        args.blur_k = 6
    if args.depth_window <= 0:
        args.depth_window = 5
    if args.depth_window % 2 == 0:
        args.depth_window += 1

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    if args.color:
        config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    try:
        profile = pipeline.start(config)
    except Exception as e:
        print("Failed to start RealSense pipeline:", e)
        sys.exit(1)

    # Depth sensor info
    depth_sensor = profile.get_device().first_depth_sensor()
    try:
        depth_scale = depth_sensor.get_depth_scale()
    except Exception:
        depth_scale = None
    print(f"Depth scale: {depth_scale} m per unit" if depth_scale else "Depth scale: unknown")

    align = rs.align(rs.stream.color) if args.color else None

    window_name = "RealSense Metallic Detector"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
            except Exception:
                # no frames
                continue

            if align:
                frames = align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame() if args.color else None

            if not depth_frame:
                # nothing to do until we get depth
                continue

            if color_frame:
                color_image = np.asanyarray(color_frame.get_data())
                display_img = color_image.copy()
                gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            else:
                # Fall back to depth_colormap for visualization/detection (not ideal for specular highlights)
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                display_img = depth_colormap.copy()
                gray_image = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2GRAY)

            # --- Detection pipeline ---
            # 1) Sobel
            sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = cv2.magnitude(sobelx, sobely)

            # 2) Normalize magnitude to 0-255 uint8
            mag_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX)
            mag_uint8 = mag_norm.astype(np.uint8)

            # 3) Binary edge map (threshold on normalized magnitude)
            _, binary_edge_map = cv2.threshold(mag_uint8, args.edge_thresh, 255, cv2.THRESH_BINARY)

            # 4) Find contours and build filled contour mask
            contours, _ = cv2.findContours(binary_edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_mask = np.zeros_like(gray_image)
            if contours:
                cv2.drawContours(contour_mask, contours, -1, 255, thickness=-1)

            # 5) Mask the grayscale image
            masked_gray = cv2.bitwise_and(gray_image, gray_image, mask=contour_mask)

            # 6) Averaging blur on masked image
            k = args.blur_k
            if k % 2 == 0:
                k += 1
            blurred = cv2.blur(masked_gray, (k, k))

            # 7) Final threshold to get metallic mask
            _, metallic_mask = cv2.threshold(blurred, args.final_thresh, 255, cv2.THRESH_BINARY)

            # 8) Find metallic regions and centroids
            metal_contours, _ = cv2.findContours(metallic_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # create overlay for visualization
            overlay = display_img.copy()
            # color mask overlay: red for metallic
            red_mask = np.zeros_like(overlay, dtype=np.uint8)
            red_mask[metallic_mask == 255] = (0, 0, 255)
            cv2.addWeighted(red_mask, 0.4, overlay, 0.6, 0, overlay)

            # For each metallic region, compute centroid and query depth
            for cnt in metal_contours:
                area = cv2.contourArea(cnt)
                # Require sufficiently large region to be considered metallic
                # Only treat regions with pixel area >= 15000 as metallic
                if area < 15000:
                    # ignore small regions / noise
                    continue

                M = cv2.moments(cnt)
                if M.get('m00', 0) == 0:
                    continue
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # Depth lookup (median in small window)
                z = median_distance_in_window(depth_frame, cx, cy, args.depth_window)

                # Draw centroid and bounding box
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.circle(overlay, (cx, cy), 4, (0, 255, 255), -1)

                z_text = f"{z:.3f} m" if z else "No depth"
                text = f"x:{cx}, y:{cy}, z:{z_text}"
                # put text above the rectangle
                tx = x
                ty = y - 10 if y - 10 > 10 else y + h + 20
                cv2.putText(overlay, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Display combined view
            cv2.imshow(window_name, overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        pipeline.stop()


if __name__ == '__main__':
    main()