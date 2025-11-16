#!/usr/bin/env python3
"""
realsense_depth_test.py

Simple test script for Intel RealSense (D455) to report depth (meters) to a surface.

Features:
- Connects to RealSense pipeline and reads depth frames
- Reports distance (in meters) at a pixel (default: image center)
- Optionally averages a small square window to reduce noise
- Continuous streaming mode or single-frame capture

Requires: pyrealsense2, numpy
"""
import argparse
import sys
import time
import signal
import cv2
import os
import tempfile
import traceback

try:
    import pyrealsense2 as rs
except Exception as e:
    print("Error: can't import pyrealsense2. Make sure librealsense and the Python wheel are installed.")
    print("See README.md for install instructions.")
    raise

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="RealSense D455 depth test - report distance in meters")
    p.add_argument("--pixel", "-p", nargs=2, type=int, metavar=("X","Y"),
                   help="XY pixel coordinates to sample (0-based). Default: image center")
    p.add_argument("--avg", "-a", type=int, default=5,
                   help="Averaging window size (odd integer). Samples a square window centered at the pixel. Default: 5")
    p.add_argument("--continuous", "-c", action="store_true",
                   help="Run continuously until Ctrl-C instead of single-shot")
    p.add_argument("--timeout", type=float, default=5.0,
                   help="Frame wait timeout in seconds (single-shot mode). Default: 5")
    p.add_argument("--width", type=int, default=424, help="Depth stream width (default 424)")
    p.add_argument("--height", type=int, default=240, help="Depth stream height (default 240)")
    p.add_argument("--fps", type=int, default=30, help="Depth stream framerate (default 30)")
    p.add_argument("--color", action="store_true", help="Enable color stream and align depth to color (optional)")
    return p.parse_args()


def get_window_coords(x, y, w, h, img_w, img_h):
    # clamp window to image bounds
    x0 = max(0, x - w//2)
    y0 = max(0, y - h//2)
    x1 = min(img_w - 1, x + w//2)
    y1 = min(img_h - 1, y + h//2)
    return x0, y0, x1, y1


def median_distance_in_window(depth_frame, x, y, window_size):
    img_w = depth_frame.get_width()
    img_h = depth_frame.get_height()
    x0, y0, x1, y1 = get_window_coords(x, y, window_size, window_size, img_w, img_h)

    distances = []
    for yy in range(y0, y1+1):
        for xx in range(x0, x1+1):
            d = depth_frame.get_distance(xx, yy)  # returns meters (float) or 0.0 if invalid
            if d > 0:
                distances.append(d)

    if not distances:
        return None
    # median is robust to outliers
    return float(np.median(distances))


def main():
    args = parse_args()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    if args.color:
        config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    try:
        profile = pipeline.start(config)
    except Exception as e:
        print("Failed to start RealSense pipeline: {}".format(e))
        print("Make sure the camera is connected, drivers/runtime are installed, and no other app is using the camera.")
        sys.exit(1)

    # Get depth sensor and scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth scale: {depth_scale} m per unit")

    align = None
    if args.color:
        # Align depth to color frame for easier pixel selection
        align = rs.align(rs.stream.color)

    # determine default pixel (center) if not provided
    # we will fetch size from an actual frame below, but pick center based on args width/height by default
    default_x = args.width // 2
    default_y = args.height // 2
    user_xy = args.pixel
    window = args.avg if args.avg and args.avg > 0 else 1
    if window % 2 == 0:
        print("Warning: --avg should be odd; incrementing to next odd value")
        window += 1

    stop = False
    show_fallback_handled = False

    def handle_sigint(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        first = True
        while not stop:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=int(max(1, args.timeout) * 1000))
            except Exception:
                print("No frames received (timeout). Is the camera attached and powered?")
                if not args.continuous:
                    break
                else:
                    time.sleep(0.1)
                    continue

            if align:
                frames = align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame() if args.color else None

            if not depth_frame:
                print("Received no depth frame")
                if not args.continuous:
                    break
                else:
                    continue

            img_w = depth_frame.get_width()
            img_h = depth_frame.get_height()

            if user_xy:
                x, y = user_xy
            else:
                x, y = img_w // 2, img_h // 2

            x = max(0, min(img_w - 1, x))
            y = max(0, min(img_h - 1, y))

            dist = median_distance_in_window(depth_frame, x, y, window)
            timestamp = time.strftime('%H:%M:%S')

            # --- Visualize ---
            if color_frame:
                color_image = np.asanyarray(color_frame.get_data())
                display_img = color_image.copy()
            else:
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(np.asanyarray(depth_frame.get_data()), alpha=0.03),
                    cv2.COLORMAP_JET
                )
                display_img = depth_colormap

            # Draw crosshair
            cv2.drawMarker(display_img, (x, y), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

            # Put text
            text = f"{dist:.3f} m" if dist else "No depth"
            cv2.putText(display_img, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(display_img, f"({x},{y})", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

            # Show the image (may fail if OpenCV wasn't built with GUI support)
            try:
                cv2.imshow("RealSense Depth View", display_img)

                # Print depth value in terminal
                if dist:
                    print(f"{timestamp} - Pixel ({x},{y}) - distance = {dist:.4f} m")
                else:
                    print(f"{timestamp} - Pixel ({x},{y}) - no valid depth samples")

                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                if not args.continuous:
                    time.sleep(0.1)
                    break
            except Exception:
                # Typical error here: cv2.error: "The function is not implemented..." when OpenCV was built without GUI
                if not show_fallback_handled:
                    print("cv2.imshow failed (likely OpenCV without GUI support). Falling back to saving a snapshot and opening it with the default image viewer.")
                    traceback.print_exc()
                    try:
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png', prefix='rs_snapshot_')
                        tmp_name = tmp.name
                        tmp.close()
                        cv2.imwrite(tmp_name, display_img)
                        print(f"Saved snapshot to: {tmp_name}")
                        # On Windows, open file with default associated app
                        try:
                            os.startfile(tmp_name)
                        except Exception as e2:
                            print(f"Could not open image automatically: {e2}. You can open it manually.")
                    except Exception as e3:
                        print(f"Failed to write snapshot fallback: {e3}")
                    show_fallback_handled = True

                # Print depth value in terminal regardless
                if dist:
                    print(f"{timestamp} - Pixel ({x},{y}) - distance = {dist:.4f} m")
                else:
                    print(f"{timestamp} - Pixel ({x},{y}) - no valid depth samples")
                # If we're not in continuous mode, break after one frame
                if not args.continuous:
                    time.sleep(0.1)
                    break

        cv2.destroyAllWindows()


    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()


if __name__ == '__main__':
    main()
