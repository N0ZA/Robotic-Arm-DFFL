import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

def metallic_edge_filter(src_bgr):
    """Filters for high-contrast, shiny metallic regions."""
    gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=1)
    gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=1)

    magnitude = cv2.magnitude(gx, gy)
    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, binary_edge = cv2.threshold(magnitude_norm, 50, 255, cv2.THRESH_BINARY)

    kernel = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(binary_edge, kernel, iterations=1)
    refined_edge_map = cv2.dilate(eroded, kernel, iterations=1)

    contours, _ = cv2.findContours(refined_edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_mask = np.zeros_like(gray)
    cv2.drawContours(contour_mask, contours, -1, 255, -1)

    masked_gray = cv2.bitwise_and(gray, gray, mask=contour_mask)
    blurred_masked = cv2.blur(masked_gray, (12, 12))
    _, metallic_mask = cv2.threshold(blurred_masked, 37, 255, cv2.THRESH_BINARY)

    return metallic_mask

def main():
    # 1. Setup RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)

    # 2. Load YOLOv8 Segmentation Model
    print("[INFO] Loading YOLOv8-Seg model...")
    model = YOLO("yolov8s-seg.pt")

    # Screen constants
    screen_center = np.array([320, 240])
    MAX_PIXEL_AREA = 5000

    try:
        while True:
            # Wait for coherent frames
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert to numpy arrays
            frame = np.asanyarray(color_frame.get_data())
            
            # 3. Generate Metallic Mask
            metallic_mask = metallic_edge_filter(frame)

            # 4. Run YOLO Inference on full frame
            results = model(frame, verbose=False)

            closest_obj_index = -1
            min_dist = float('inf')
            best_stats = None # Will hold (cX, cY, depth, area)

            # 5. Filter and select the "Winner"
            if results[0].masks is not None:
                for i in range(len(results[0].masks.data)):
                    # Get individual mask
                    m_data = results[0].masks.data[i].cpu().numpy()
                    m_resized = cv2.resize(m_data, (640, 480))
                    binary_mask = (m_resized > 0.5).astype(np.uint8) * 255

                    # Rule A: Area must be < 800 pixels
                    pixel_area = cv2.countNonZero(binary_mask)
                    if pixel_area >= MAX_PIXEL_AREA:
                        continue

                    # Rule B: Must overlap with metallic highlights
                    overlap = cv2.bitwise_and(binary_mask, metallic_mask)
                    if not np.any(overlap > 0):
                        continue

                    # Rule C: Find the one closest to screen center
                    M = cv2.moments(binary_mask)
                    if M["m00"] == 0: continue
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    dist_to_center = np.linalg.norm(np.array([cX, cY]) - screen_center)

                    if dist_to_center < min_dist:
                        min_dist = dist_to_center
                        closest_obj_index = i
                        # Sample depth from the centroid
                        depth_z = depth_frame.get_distance(cX, cY)
                        best_stats = (cX, cY, depth_z, pixel_area)

            # 6. Visualization
            annotated_frame = frame.copy()
            
            # Draw a subtle crosshair at screen center
            cv2.line(annotated_frame, (310, 240), (330, 240), (255, 255, 255), 1)
            cv2.line(annotated_frame, (320, 230), (320, 250), (255, 255, 255), 1)

            if closest_obj_index != -1:
                # Plot only the winner's mask (no box, no label)
                annotated_frame = results[0][closest_obj_index].plot(
                    img=annotated_frame, 
                    labels=False, 
                    boxes=False, 
                    conf=False
                )
                
                # Draw the centroid and the Z coordinate text
                cX, cY, z, area = best_stats
                cv2.circle(annotated_frame, (cX, cY), 4, (0, 255, 0), -1)
                
                # Display text near the object
                telemetry = f"X:{cX} Y:{cY} Z:{z:.2f}m Area:{area}px"
                cv2.putText(annotated_frame, telemetry, (cX + 10, cY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Display the result
            cv2.imshow('Center-Targeted Metallic Filter', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()