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
    p.add_argument("--model", default="yolo11s-seg.pt", help="Path to FastSAM model (default: Shiny/FastSAM-x.pt)")
    p.add_argument("--width", type=int, default=848, help="Color fram255 e width")
    p.add_argument("--height", type=int, default=480, help="Color frame height")
    p.add_argument("--fps", type=int, default=60,help="Camera FPS")
    p.add_argument("--edge-method", choices=("sobel","farid"), default="sobel", help="Edge operator for preprocessing")
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


def preprocess_shiny_mask(gray, color_bgr, edge_method="sobel"):
    """Return a 1-channel uint8 mask (0 or 255) indicating candidate shiny regions.
    
    Improved algorithm using:
    1. HSV color-space analysis for specular reflectance modeling
    2. Adaptive thresholding for robustness to lighting variations
    3. Combined gradient, brightness, and saturation criteria
    
    Args:
        gray: Grayscale image (H, W) uint8
        color_bgr: Color image (H, W, 3) BGR uint8
        edge_method: "farid" or "sobel"
    
    Returns:
        Binary mask (H, W) uint8 with values 0 or 255
    """
    # Step 1: Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Step 2: Compute gradients (edge detection)
    if edge_method == "farid":
        gx, gy = farid_operator(blur)
    else:
        gx, gy = sobel_operator(blur)

    gradient_magnitude = cv2.magnitude(gx, gy)
    magnitude_norm = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Step 3: Adaptive thresholding for gradient magnitude (instead of fixed threshold)
    # Use Otsu's method to automatically determine optimal threshold
    _, binary_edge = cv2.threshold(magnitude_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply morphological operations to clean up edge mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_edge = cv2.morphologyEx(binary_edge, cv2.MORPH_CLOSE, kernel)

    # Step 4: HSV color-space analysis for specular highlights
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Mask 1: High brightness (Value channel) - specular highlights are bright
    # Use adaptive threshold for robustness
    _, high_brightness_mask = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Further refine: only keep very bright regions (top 20% after Otsu)
    bright_threshold = max(200, int(cv2.minMaxLoc(v)[1] * 0.8))
    _, high_brightness_mask = cv2.threshold(v, bright_threshold, 255, cv2.THRESH_BINARY)
    
    # Mask 2: Low saturation - metallic/shiny surfaces reflect light with low color saturation
    _, low_saturation_mask = cv2.threshold(s, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Step 5: Combine masks using bitwise AND
    # Shiny regions must have: strong edges AND high brightness AND low saturation
    combined_mask = cv2.bitwise_and(binary_edge, high_brightness_mask)
    combined_mask = cv2.bitwise_and(combined_mask, low_saturation_mask)
    
    # Step 6: Find contours and fill them to create solid regions
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_mask = np.zeros_like(gray, dtype=np.uint8)
    if len(contours) > 0:
        cv2.drawContours(contour_mask, contours, -1, 255, thickness=-1)
    
    # Step 7: ROI-based refinement - focus on center region
    H, W = gray.shape
    center_x, center_y = W // 2, H // 2
    roi_radius = 200  # pixels from center
    
    # Create circular ROI mask
    y_coords, x_coords = np.ogrid[:H, :W]
    roi_mask = ((x_coords - center_x)**2 + (y_coords - center_y)**2 <= roi_radius**2).astype(np.uint8) * 255
    
    # Apply ROI mask to focus processing on center region
    roi_contour_mask = cv2.bitwise_and(contour_mask, roi_mask)
    
    # Step 8: Final refinement with blur and threshold on brightness
    masked_v = cv2.bitwise_and(v, v, mask=roi_contour_mask)
    blurred = cv2.blur(masked_v, (12, 12))
    
    # Use adaptive threshold for final mask
    _, final_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Fallback: if Otsu gives empty result, use fixed threshold
    if cv2.countNonZero(final_mask) == 0:
        _, final_mask = cv2.threshold(blurred, 37, 255, cv2.THRESH_BINARY)
    
    return final_mask


def filter_masks_by_constraints(masks, image_shape, min_area_ratio=0.001, max_area_ratio=0.8, 
                                 min_solidity=0.3, max_aspect_ratio=10.0):
    """Filter masks based on size and shape constraints.
    
    Args:
        masks: (n, H, W) boolean/uint8 array of masks
        image_shape: (H, W) tuple
        min_area_ratio: Minimum mask area as fraction of image area
        max_area_ratio: Maximum mask area as fraction of image area
        min_solidity: Minimum solidity (area / convex_hull_area)
        max_aspect_ratio: Maximum bounding box aspect ratio
    
    Returns:
        List of valid mask indices
    """
    if masks is None or len(masks) == 0:
        return []
    
    H, W = image_shape
    image_area = H * W
    min_area = int(image_area * min_area_ratio)
    max_area = int(image_area * max_area_ratio)
    
    valid_indices = []
    
    for i in range(len(masks)):
        m = masks[i].astype(np.uint8)
        
        # Calculate area
        area = cv2.countNonZero(m)
        
        # Filter by area
        if area < min_area or area > max_area:
            continue
        
        # Get contours for shape analysis
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
        
        # Use the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate solidity (area / convex hull area)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            if solidity < min_solidity:
                continue
        
        # Calculate aspect ratio from bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        if h > 0:
            aspect_ratio = w / h
            # Check both orientations
            if aspect_ratio > max_aspect_ratio and (1.0 / aspect_ratio) > max_aspect_ratio:
                continue
        
        valid_indices.append(i)
    
    return valid_indices

def extract_masks_from_results(results):
    """Robust extractor for mask arrays from ultralytics FastSAM results.
    Returns masks as a boolean numpy array with shape (n, H, W) or None.
    """
    if not results:
        return None
    r = results[0]
    
    masks = None
    if hasattr(r, 'masks') and r.masks is not None:
        masks_obj = r.masks
        
        # 1. Get the raw data container
        data = None
        if hasattr(masks_obj, 'data'):
            data = masks_obj.data
        elif hasattr(masks_obj, 'masks'):
            data = masks_obj.masks
        else:
            # Fallback: try using the object itself
            data = masks_obj

        # 2. Convert to numpy (handling GPU Tensors)
        if data is not None:
            if hasattr(data, 'cpu'):
                # It is a torch Tensor, move to cpu first
                masks = data.cpu().numpy()
            else:
                # It is likely already a list or numpy array
                try:
                    masks = np.asarray(data)
                except Exception:
                    masks = None
                    
    return masks


def choose_mask_closest_to_center(masks, image_shape, min_area_ratio=0.001, max_area_ratio=0.8):
    """Given masks (n,H,W) boolean/uint8 and image_shape (H,W), return index of mask whose centroid
    is closest to the image center, with size and shape filtering. Returns -1 if masks is empty.
    
    Args:
        masks: (n,H,W) boolean/uint8 array
        image_shape: (H, W) tuple
        min_area_ratio: Minimum mask area as fraction of image area
        max_area_ratio: Maximum mask area as fraction of image area
    
    Returns:
        Index of selected mask or -1
    """
    if masks is None or len(masks) == 0:
        return -1

    # Filter masks by size and shape constraints
    valid_indices = filter_masks_by_constraints(
        masks, image_shape,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        min_solidity=0.3,
        max_aspect_ratio=10.0
    )
    
    if len(valid_indices) == 0:
        return -1

    H, W = image_shape
    cx, cy = W / 2.0, H / 2.0

    best_idx = -1
    best_dist = float('inf')

    for i in valid_indices:
        m = masks[i].astype(np.uint8)
        # Compute centroid robustly
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


def choose_mask_lowest_depth_within_radius(masks, depth_img, image_shape, radius=80,
                                           min_area_ratio=0.001, max_area_ratio=0.8):
    """Select the mask whose centroid lies within `radius` pixels from image center
    and has the lowest median depth value, with size and shape filtering.
    
    Args:
        masks: (n,H,W) boolean array
        depth_img: 2D ndarray of depth in meters (or raw units) with shape (H,W), or None
        image_shape: (H, W) tuple
        radius: Maximum distance from center in pixels
        min_area_ratio: Minimum mask area as fraction of image area
        max_area_ratio: Maximum mask area as fraction of image area
    
    Returns:
        Selected index or -1 if none found
    """
    if masks is None or len(masks) == 0:
        return -1

    # Step 1: Filter masks by size and shape constraints
    valid_indices = filter_masks_by_constraints(
        masks, image_shape, 
        min_area_ratio=min_area_ratio, 
        max_area_ratio=max_area_ratio,
        min_solidity=0.3,
        max_aspect_ratio=10.0
    )
    
    if len(valid_indices) == 0:
        return -1

    H, W = image_shape
    cx, cy = W / 2.0, H / 2.0

    best_idx = -1
    best_val = float('inf')

    # Step 2: Among valid masks, select based on depth and proximity to center
    for i in valid_indices:
        m = masks[i].astype(np.uint8)
        ys, xs = np.where(m > 0)
        if len(xs) == 0:
            continue

        mx = xs.mean()
        my = ys.mean()
        dist_to_center = np.hypot(mx - cx, my - cy)
        
        # Only consider masks whose centroid is within the radius
        if dist_to_center > radius:
            continue

        if depth_img is not None:
            # Make sure depth image matches shape; if not, try to resize
            try:
                if depth_img.shape != (H, W):
                    depth_resized = cv2.resize(depth_img, (W, H), interpolation=cv2.INTER_NEAREST)
                else:
                    depth_resized = depth_img
            except Exception:
                depth_resized = depth_img

            depths = depth_resized[ys, xs]
            # Ignore zero/invalid depth values
            valid = depths > 0
            if not np.any(valid):
                continue
            median_depth = float(np.median(depths[valid]))
            
            # Smaller median_depth means closer to camera
            if median_depth < best_val:
                best_val = median_depth
                best_idx = i
        else:
            # No depth available: pick the one closest to center (as fallback)
            if dist_to_center < best_val:
                best_val = dist_to_center
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
    # Print the actual camera stream resolution (some devices will pick the nearest supported size)
    try:
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        actual_width = getattr(color_profile, 'width', None)
        actual_height = getattr(color_profile, 'height', None)
        if actual_width is not None and actual_height is not None:
            print(f"Camera streaming color at: {actual_width}x{actual_height} (requested {args.width}x{args.height})")
    except Exception:
        # non-fatal: just continue if profile introspection isn't available
        pass
    # Try to obtain depth scale (meters per unit) for converting depth frame to meters
    depth_scale = None
    try:
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print(f"Depth scale: {depth_scale} meters per unit")
    except Exception:
        depth_scale = None
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Create RealSense filters: spatial (reduces spatial noise) and temporal (smooths over time)
    try:
        spatial = rs.spatial_filter()
        temporal = rs.temporal_filter()
        # You can tune filter options here if desired, e.g. spatial.set_option(...)
    except Exception:
        spatial = None
        temporal = None

    win_name = "RealSense Shiny Detector"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # Force the window to match the camera resolution
    cv2.resizeWindow(win_name, args.width, args.height)


    try:
        last_time = time.time()
        frame_count = 0
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            # Apply depth filters in order: decimation -> spatial -> temporal -> hole_filling
            if depth_frame is not None:
                try:
                    if spatial is not None:
                        depth_frame = spatial.process(depth_frame)
                    if temporal is not None:
                        depth_frame = temporal.process(depth_frame)
                except Exception:
                    # non-fatal: continue with unfiltered depth if filters fail
                    pass
            if not color_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            # match notebook: scale up for visualization if needed
            # Some RealSense devices do not support the exact requested (width,height).
            # Resize the captured frame to the requested size so the display window
            # and subsequent processing use the expected resolution (e.g. 640x360).
            h0, w0 = color.shape[:2]
            if (w0, h0) != (args.width, args.height):
                color = cv2.resize(color, (args.width, args.height), interpolation=cv2.INTER_LINEAR)

            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

            # create candidate shiny mask with improved algorithm
            shiny_mask = preprocess_shiny_mask(gray, color, edge_method=args.edge_method)

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

                # Prepare a depth image (in meters if depth_scale known) for selection
                depth_img = None
                if depth_frame:
                    try:
                        depth_img = np.asanyarray(depth_frame.get_data()).astype(np.float32)
                        if depth_scale is not None:
                            depth_img *= float(depth_scale)
                    except Exception:
                        depth_img = None

                # Select mask with lowest median depth whose centroid is within 80px of center
                selected_idx = choose_mask_lowest_depth_within_radius(masks_bool, depth_img, (args.height, args.width), radius=80)
                # Fallback to previous center-based selection if no mask met the depth/radius criteria
                if selected_idx == -1:
                    selected_idx = choose_mask_closest_to_center(masks_bool, (args.height, args.width))

            overlay = color.copy()
            info_text = "No detection"
            if selected_idx >= 0:
                m = masks[selected_idx]
                
                # Make uint8 single channel
                m_u8 = (m > 0).astype(np.uint8) * 255

                # FIX: Resize mask to match current image dimensions if they differ
                # cv2.resize expects (width, height)
                h, w = overlay.shape[:2]
                if m_u8.shape[:2] != (h, w):
                    m_u8 = cv2.resize(m_u8, (w, h), interpolation=cv2.INTER_NEAREST)

                # Create colored overlay (green)
                overlay[m_u8 == 255] = (255, 0, 0)

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
