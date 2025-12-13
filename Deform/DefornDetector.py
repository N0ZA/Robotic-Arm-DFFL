#!/usr/bin/env python3
"""
realsense_Object (deformed)_detector.py

Real-time Object (deformed) detector using Intel RealSense (color + depth) with YOLO11-seg.

Behavior:
- Streams color and depth, aligns depth to color
- Resizes color frames to 480x270 and applies the same processing used in TinyDetectorrr.ipynb
  (Sobel X/Y with ksize=3, convertScaleAbs, addWeighted, threshold at 120 using THRESH_TOZERO)
- Converts white parts of the processed image to RGB by masking the original color image
- Runs YOLO11-seg on the masked RGB image for segmentation
- Finds contours from YOLO segmentation masks
- Selects only ONE object (closest to center) and displays its XYZ coordinates
- Overlays contours, centroid and (x, y, z) on the display and runs continuously

Requires: pyrealsense2, numpy, opencv-python, ultralytics
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

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed. Install with: pip install ultralytics")
    sys.exit(1)


def clamp(v, a, b):
    return max(a, min(b, v))


def main():
    parser = argparse.ArgumentParser(description="Real-time Object (deformed) detector using Intel RealSense with YOLO11-seg")
    parser.add_argument("--min-perim", type=float, default=150.0, help="Minimum contour perimeter to consider (default 150.0)")
    parser.add_argument("--max-perim", type=float, default=220.0, help="Maximum contour perimeter to consider (default 220.0)")
    parser.add_argument("--fps", type=int, default=30, help="Stream frames per second (default 30)")
    parser.add_argument("--yolo-model", type=str, default="yolo11s-seg.pt", help="YOLO11-seg model path (default yolo11s-seg.pt)")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO confidence threshold (default 0.25)")
    args = parser.parse_args()

    # Processing sizes from the notebook
    proc_w, proc_h = 480, 270

    # Load YOLO11-seg model
    print(f"Loading YOLO11-seg model: {args.yolo_model}")
    try:
        yolo_model = YOLO(args.yolo_model)
        print("YOLO11-seg model loaded successfully")
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        print("Make sure you have the correct model file. You can download it with:")
        print("  from ultralytics import YOLO; YOLO('yolo11s-seg.pt')")
        sys.exit(1)

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

    window_name = "RealSense Object (deformed) Detector with YOLO11-seg"
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

            # Create a binary mask from the thresholded image (white parts = 255, black parts = 0)
            # Any pixel value > 0 in thresh_image is considered "white"
            binary_mask = (thresh_image > 0).astype(np.uint8) * 255

            # Convert the white parts to RGB by masking the original resized color image
            # Create a 3-channel mask
            mask_3ch = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
            
            # Apply mask to get RGB image with only white parts from processed image
            # Where mask is white (255), keep original color; where black (0), set to black
            masked_rgb = cv2.bitwise_and(resized, mask_3ch)

            # Run YOLO11-seg on the masked RGB image for segmentation only
            yolo_results = yolo_model(masked_rgb, conf=args.yolo_conf, verbose=False)

            # Draw on a copy of the resized image
            display = resized.copy()

            # Extract contours from YOLO segmentation masks
            chosen = None
            if yolo_results and len(yolo_results) > 0:
                result = yolo_results[0]
                
                if result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    
                    # Calculate center of screen
                    center_x, center_y = proc_w // 2, proc_h // 2
                    
                    # Find the mask closest to center
                    best_distance = float('inf')
                    best_mask = None
                    best_contour = None
                    
                    for mask in masks:
                        # Resize mask to match display size
                        mask_resized = cv2.resize(mask, (proc_w, proc_h))
                        # Convert to binary mask
                        mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                        
                        # Find contours in this mask
                        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if len(contours) > 0:
                            # Use the largest contour from this mask
                            cnt = max(contours, key=cv2.contourArea)
                            
                            # Calculate centroid
                            M = cv2.moments(cnt)
                            if M["m00"] > 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                
                                # Calculate distance to center
                                d_center = ((cx - center_x) ** 2 + (cy - center_y) ** 2) ** 0.5
                                
                                # Keep track of closest mask
                                if d_center < best_distance:
                                    best_distance = d_center
                                    best_mask = mask_resized
                                    best_contour = cnt
                                    
                                    # Map centroid to original color resolution to query depth frame
                                    scale_x = orig_w / float(proc_w)
                                    scale_y = orig_h / float(proc_h)
                                    orig_x = int(clamp(round(cx * scale_x), 0, orig_w - 1))
                                    orig_y = int(clamp(round(cy * scale_y), 0, orig_h - 1))
                                    
                                    # Query depth (meters)
                                    try:
                                        z = depth_frame.get_distance(orig_x, orig_y)
                                    except Exception:
                                        z = 0.0
                                    
                                    chosen = {
                                        "contour": cnt,
                                        "cx": cx,
                                        "cy": cy,
                                        "orig_x": orig_x,
                                        "orig_y": orig_y,
                                        "z": z,
                                        "mask": best_mask
                                    }
            
            # Draw the chosen object (closest to center)
            if chosen:
                cnt = chosen["contour"]
                cx = chosen["cx"]
                cy = chosen["cy"]
                z = chosen["z"]
                
                # Draw the segmentation mask overlay
                colored_mask = np.zeros_like(display)
                colored_mask[chosen["mask"] > 0.5] = [0, 255, 0]  # Green color for selected mask
                display = cv2.addWeighted(display, 1.0, colored_mask, 0.4, 0)
                
                # Draw contour
                cv2.drawContours(display, [cnt], -1, (0, 255, 0), 2)
                
                # Draw centroid
                cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)
                
                # Draw crosshair at centroid
                cv2.line(display, (cx - 10, cy), (cx + 10, cy), (0, 0, 255), 2)
                cv2.line(display, (cx, cy - 10), (cx, cy + 10), (0, 0, 255), 2)
                
                # Display XYZ coordinates
                text = f"X={cx}, Y={cy}, Z={z*100:.2f} cm"
                cv2.putText(display, text, (cx + 15, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display, text, (cx + 15, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
            else:
                # No objects detected
                cv2.putText(display, "No objects detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

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
