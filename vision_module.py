import threading
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO, FastSAM

class RealSenseVision:
    def __init__(self, tiny_model="yolo11s-seg.pt", shiny_model="FastSAM-x.pt"):
        """
        Initializes the RealSense camera and loads the AI models.
        """
        self.running = False
        self.lock = threading.Lock()
        self.latest_detection = None
        self.current_mode = "tiny"  # Default mode
        
        # --- Load Models ---
        print("[Vision] Loading AI Models...")
        try:
            self.model_tiny = YOLO(tiny_model)  # For Tiny & Deform
            self.model_shiny = FastSAM(shiny_model)  # For Shiny
            print("[Vision] Models loaded successfully.")
        except Exception as e:
            print(f"[Vision] Error loading models: {e}")
            raise

        # --- Configure RealSense ---
        print("[Vision] Starting RealSense Camera...")
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        
        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        
        # Get Depth Scale (to convert raw units to meters)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        
        # Get Camera Intrinsics
        self.depth_intrinsics = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        print(f"[Vision] Camera ready. Depth Scale: {self.depth_scale}")

    def get_intrinsics(self):
        """Returns the camera intrinsics for deprojection."""
        return self.depth_intrinsics

    def start(self):
        """Starts the background processing thread."""
        self.running = True
        self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.thread.start()
        print("[Vision] Background processing started.")

    def stop(self):
        """Stops the camera and processing."""
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.pipeline.stop()
        print("[Vision] Stopped.")

    def set_mode(self, mode):
        """Switches detection mode: 'tiny', 'deform', or 'shiny'."""
        if mode in ["tiny", "deform", "shiny"]:
            with self.lock:
                self.current_mode = mode
            print(f"[Vision] Switched to mode: {mode}")
        else:
            print(f"[Vision] Invalid mode: {mode}")

    def get_latest_coordinates(self):
        """Returns the most recent (X, Y, Z) coordinates safely."""
        with self.lock:
            return self.latest_detection

    def _processing_loop(self):
        while self.running:
            try:
                # 1. Capture Frames
                frames = self.pipeline.wait_for_frames()
                aligned = self.align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                color_img = np.asanyarray(color_frame.get_data())
                
                # 2. Determine Logic based on Mode
                mode = self.current_mode
                result = None

                if mode == "shiny":
                    result = self._process_shiny(color_img, depth_frame)
                else:
                    # 'tiny' and 'deform' share the same core logic
                    result = self._process_tiny_deform(color_img, depth_frame, mode)

                # 3. Update Shared State
                with self.lock:
                    self.latest_detection = result

            except Exception as e:
                print(f"[Vision] Error in loop: {e}")

    # --- Specific Detection Algorithms ---

    def _process_tiny_deform(self, img, depth_frame, mode):
        """
        Logic from Tiny(Base).py and DefornDetector.py
        """
        # Resize to match training/notebook consistency
        proc_w, proc_h = 480, 270
        resized = cv2.resize(img, (proc_w, proc_h))
        orig_h, orig_w = img.shape[:2]

        # Sobel + Threshold + Masking
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
        sobely = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
        combined = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
        
        _, thresh = cv2.threshold(combined, 120, 255, cv2.THRESH_TOZERO)
        binary_mask = (thresh > 0).astype(np.uint8) * 255
        mask_3ch = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        masked_input = cv2.bitwise_and(resized, mask_3ch)

        # AI Inference
        results = self.model_tiny(masked_input, verbose=False, conf=0.25)
        
        # Find Object Closest to Center
        if results and results[0].masks:
            masks = results[0].masks.data.cpu().numpy()
            center_x, center_y = proc_w // 2, proc_h // 2
            best_obj = None
            best_dist = float('inf')

            for mask in masks:
                mask_resized = cv2.resize(mask, (proc_w, proc_h))
                mask_bin = (mask_resized > 0.5).astype(np.uint8)
                
                contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours: continue
                
                cnt = max(contours, key=cv2.contourArea)
                
                # Deform mode specific: Filter by perimeter? 
                # (Optional: Add perimeter check here if strict adherence to Deforn(Base).py is needed)
                
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                    dist = ((cx - center_x)**2 + (cy - center_y)**2)**0.5
                    
                    if dist < best_dist:
                        best_dist = dist
                        # Map back to full resolution
                        scale_x, scale_y = orig_w / proc_w, orig_h / proc_h
                        fx, fy = int(cx * scale_x), int(cy * scale_y)
                        
                        # Get Depth
                        z = depth_frame.get_distance(fx, fy)
                        best_obj = {"x": fx, "y": fy, "z": z, "type": mode}

            return best_obj
        return None

    def _process_shiny(self, img, depth_frame):
        """
        Logic from ShinyDetector(HSV).py
        """
        # Preprocessing: Blur -> Sobel -> Normalize -> Otsu
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        mag = cv2.normalize(cv2.magnitude(gx, gy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, edges = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # HSV Masking
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)
        bright_thresh = max(200, int(cv2.minMaxLoc(v)[1] * 0.8))
        _, high_v = cv2.threshold(v, bright_thresh, 255, cv2.THRESH_BINARY)
        _, low_s = cv2.threshold(s, 50, 255, cv2.THRESH_BINARY_INV)
        
        combined = cv2.bitwise_and(edges, high_v)
        combined = cv2.bitwise_and(combined, low_s)
        
        # FastSAM Inference on Masked Image
        masked_img = cv2.bitwise_and(img, img, mask=combined)
        results = self.model_shiny(masked_img, verbose=False)
        
        # Logic: Lowest Depth + Center Proximity
        if results and results[0].masks:
            masks = results[0].masks.data.cpu().numpy()
            h, w = img.shape[:2]
            best_obj = None
            best_z = float('inf')
            
            # Convert depth frame to array once
            depth_arr = np.asanyarray(depth_frame.get_data()).astype(np.float32)
            
            for mask in masks:
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h))
                
                m_u8 = (mask > 0).astype(np.uint8)
                ys, xs = np.where(m_u8 > 0)
                if len(xs) == 0: continue
                
                # Center Filter (80px radius)
                cx, cy = xs.mean(), ys.mean()
                if np.hypot(cx - w/2, cy - h/2) > 80: continue
                
                # Median Depth
                z_vals = depth_arr[ys, xs]
                valid_z = z_vals[z_vals > 0]
                if len(valid_z) == 0: continue
                
                median_z = np.median(valid_z) * self.depth_scale # Meters
                
                if median_z < best_z:
                    best_z = median_z
                    best_obj = {"x": int(cx), "y": int(cy), "z": median_z, "type": "shiny"}
            
            return best_obj
        return None
