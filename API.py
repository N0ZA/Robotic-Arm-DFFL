import time
from vision_module import RealSenseVision

# 1. Initialize the Vision System
# Ensure your .pt model files are in the same folder or provide full paths
vision = RealSenseVision(tiny_model="yolo11s-seg.pt", shiny_model="FastSAM-x.pt")
vision.start()

try:
    # 2. Set the detection mode (tiny, deform, or shiny)
    vision.set_mode("tiny") 
    
    print("Robot System Ready. Press Ctrl+C to stop.")
    
    while True:
        # 3. Get coordinates instantly (non-blocking)
        coords = vision.get_latest_coordinates()
        
        if coords:
            print(f"Object Detected [{coords['type']}]: "
                  f"X={coords['x']}, Y={coords['y']}, Z={coords['z']:.3f}m")
            
            # --- INSERT ROBOT MOVEMENT CODE HERE ---
            # robot.move_to(coords['x'], coords['y'], coords['z'])
            
        else:
            print("Scanning...")
            
        time.sleep(0.1) # Loop rate

except KeyboardInterrupt:
    print("Stopping...")

finally:
    vision.stop()