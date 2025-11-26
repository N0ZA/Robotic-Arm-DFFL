#!/usr/bin/env python3
"""
realsense_sachet_detector.py

Real-time sachet detector using Intel RealSense (color + depth).

Behavior:
- Streams color and depth, aligns depth to color
- Resizes color frames to 480x270 and applies the same processing used in TinyDetectorrr.ipynb
  (Sobel X/Y with ksize=3, convertScaleAbs, addWeighted, threshold at 120 using THRESH_TOZERO)
- Finds contours, computes perimeters and filters them by min/max perimeters (120.0 - 200.0)
- For each filtered contour, computes centroid in the resized image and maps it back to
  the original color resolution to query the aligned depth frame for the Z coordinate
- Overlays contours, centroid and (x, y, z) on the display and runs continuously

Requires: pyrealsense2, numpy, opencv-python
"""
import time
import argparse
import sys

try:
    import pyrealsense2 as rs
except Exception:
    print("Error: pyrealsense2 not installed or import failed. Install the RealSense SDK and Python wheel.")
    raise

import cv2
import numpy as np


def clamp(v, a, b):
    return max(a, min(b, v))


def main():
    parser = argparse.ArgumentParser(description="Real-time sachet detector using Intel RealSense")
    parser.add_argument("--min-perim", type=float, default=120.0, help="Minimum contour perimeter to consider (default 120.0)")
    parser.add_argument("--max-perim", type=float, default=200.0, help="Maximum contour perimeter to consider (default 200.0)")
    parser.add_argument("--fps", type=int, default=30, help="Stream frames per second (default 30)")
    args = parser.parse_args()

    # Processing sizes from the notebook
    proc_w, proc_h = 480, 270

    # Start RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 480, 270, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, 480, 270, rs.format.bgr8, args.fps)

    try:
        profile = pipeline.start(config)
    except Exception as e:
        print("Failed to start RealSense pipeline:", e)
        sys.exit(1)

    # Align depth to color
    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print(f"Depth scale: {depth_scale} m/unit")
    except Exception:
        depth_scale = None

    window_name = "RealSense Sachet Detector"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()

            if not depth_frame or not color_frame:
                print("Missing frames")
                continue

            color_image = np.asanyarray(color_frame.get_data())
            orig_h, orig_w = color_image.shape[:2]

            # Resize to the notebook processing size
            resized = cv2.resize(color_image, (proc_w, proc_h))

            # Convert to grayscale
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            # Sobel X and Y (as in notebook)
            sobelx_64f = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobelx = cv2.convertScaleAbs(sobelx_64f)
            sobely_64f = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobely = cv2.convertScaleAbs(sobely_64f)
            sobel_combined = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

            # Threshold (use same type as notebook: THRESH_TOZERO with thresh=120)
            _, thresh_image = cv2.threshold(sobel_combined, 120, 255, cv2.THRESH_TOZERO)

            # Find contours
            contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Compute perimeters for contours
            perimeters = [cv2.arcLength(cnt, True) for cnt in contours]

            # Draw on a copy of the resized image
            display = resized.copy()

            # --- Adaptive perimeter thresholds ---
            # initialize base and current thresholds outside the loop on first iteration
            if not hasattr(main, "base_min_perim"):
                main.base_min_perim = args.min_perim
                main.base_max_perim = args.max_perim
                # current thresholds start at base values
                main.current_min_perim = float(main.base_min_perim)
                main.current_max_perim = float(main.base_max_perim)
                main.initial_ref_z = None
                main.last_valid_z = None

            # Collect candidate contours that pass current thresholds
            candidates = []
            for i, cnt in enumerate(contours):
                perim = perimeters[i]
                if main.current_min_perim <= perim <= main.current_max_perim:
                    M = cv2.moments(cnt)
                    if M["m00"] == 0:
                        continue
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Map centroid to original color resolution to query depth frame
                    scale_x = orig_w / float(proc_w)
                    scale_y = orig_h / float(proc_h)
                    orig_x = int(clamp(round(cx * scale_x), 0, orig_w - 1))
                    orig_y = int(clamp(round(cy * scale_y), 0, orig_h - 1))

                    # Query depth (meters). get_distance returns meters (float) or 0.0 if invalid
                    try:
                        z = depth_frame.get_distance(orig_x, orig_y)
                    except Exception:
                        z = 0.0

                    # distance to center (in resized image coords)
                    center_x, center_y = proc_w // 2, proc_h // 2
                    d_center = ((cx - center_x) ** 2 + (cy - center_y) ** 2) ** 0.5

                    candidates.append({
                        "idx": i,
                        "contour": cnt,
                        "perim": perim,
                        "cx": cx,
                        "cy": cy,
                        "orig_x": orig_x,
                        "orig_y": orig_y,
                        "z": z,
                        "d_center": d_center,
                    })

            # Select the single candidate whose centroid is closest to screen center
            chosen = None
            if candidates:
                candidates.sort(key=lambda c: c["d_center"])
                chosen = candidates[0]

                # update reference depth and adaptive thresholds using chosen object's depth
                zc = chosen.get("z", 0.0)
                if zc and zc > 0:
                    if main.initial_ref_z is None:
                        main.initial_ref_z = zc
                    main.last_valid_z = zc

                    # compute scale factor: if z increases (farther) scale < 1 -> decrease perimeters
                    scale = main.initial_ref_z / zc if zc > 0 else 1.0
                    # clamp scale to avoid extreme changes
                    scale = clamp(scale, 0.5, 2.0)
                    main.current_min_perim = max(10.0, main.base_min_perim * scale)
                    main.current_max_perim = max(main.current_min_perim + 10.0, main.base_max_perim * scale)

                # Draw only the chosen contour
                cnt = chosen["contour"]
                cx = chosen["cx"]
                cy = chosen["cy"]
                perim = chosen["perim"]
                z = chosen["z"]

                cv2.drawContours(display, [cnt], -1, (0, 255, 0), 2)
                cv2.circle(display, (cx, cy), 4, (0, 0, 255), -1)
                text = f"x={cx}, y={cy}, z={z*100:.3f} cm"
                cv2.putText(display, text, (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display, f"perim={perim:.1f}", (cx + 8, cy + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # If none found, show a hint
            else:
                cv2.putText(display, "No sachets found (perimeter filter)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Always show current adaptive thresholds on the display
            cv2.putText(display, f"minP={main.current_min_perim:.1f} maxP={main.current_max_perim:.1f}", (10, proc_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # (handled above) - no-op here

            # (edge mapping thumbnail removed)

            cv2.imshow(window_name, cv2.cvtColor(display, cv2.COLOR_BGR2RGB))

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        pipeline.stop()


if __name__ == '__main__':
    main()
