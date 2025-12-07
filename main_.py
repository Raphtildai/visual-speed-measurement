# main.py

from speedometer.pipeline_ import run_pipeline_corrected
import os
import pickle

ALL_IMAGES_DIR = 'dataset'
ANNOTATIONS_DIR = 'annotations'
WORK_DIR = 'speed_work'

# --- CONFIGURATION SWITCH ---
# OPTIONS: "FULL", "HALF1", "HALF2", "CHUNK", "TEST"
RUN_MODE = "TEST"   
# --------------------------

# Based on your 3m spacing world coordinates:
# Your world grid is 3m wide × 6m long (points at [0,6], [0,3], [0,0], [3,6], [3,3], [3,0])
WORLD_WIDTH = 3.0   # meters (from your world coordinates)
WORLD_LENGTH = 6.0  # meters (from your world coordinates)

# Scale factor - critical for proper BEV size
# This determines how many pixels represent 1 meter in BEV
SCALE_PX_PER_M = 100.0  # 100 pixels per meter = reasonable BEV size

# BEV output should MATCH your world coordinates
# If you want to see your 3m×6m world grid, use these dimensions
BEV_WIDTH_M = WORLD_WIDTH + 1.0   # Add 0.5m border on each side
BEV_LENGTH_M = WORLD_LENGTH + 2.0 # Add 1m border on front/back

# Total number of frames (based on earlier pipeline output)
TOTAL_FRAMES = 3405 
HALF_FRAMES = TOTAL_FRAMES // 2 

# Configuration based on RUN_MODE
if RUN_MODE == "FULL":
    START_FRAME = 0
    FRAME_COUNT = TOTAL_FRAMES
    OUTPUT_VIDEO = 'results/result_speedometer_full.mp4'
elif RUN_MODE == "HALF1":
    START_FRAME = 0
    FRAME_COUNT = HALF_FRAMES
    OUTPUT_VIDEO = 'results/result_speedometer_part1.mp4'
elif RUN_MODE == "HALF2":
    START_FRAME = HALF_FRAMES
    FRAME_COUNT = TOTAL_FRAMES - HALF_FRAMES
    OUTPUT_VIDEO = 'results/result_speedometer_part2.mp4'
elif RUN_MODE == "CHUNK":
    START_FRAME = 0
    FRAME_COUNT = 100 # Run only the first 100 frames for quick testing
    OUTPUT_VIDEO = 'results/result_speedometer_test_chunk.mp4'
elif RUN_MODE == "TEST":
    START_FRAME = 0
    FRAME_COUNT = 4  # Only the first two frames for quick debugging
    # CRITICAL CHANGE: Set output_video to None to enable cv2.imshow interactive mode
    OUTPUT_VIDEO = None
else:
    raise ValueError(f"Invalid RUN_MODE: {RUN_MODE}")

# Parameters for the pipeline
flow_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
# SCALE_PX_PER_M = 100.0 # Increased for better BEV clarity

# --- Pipeline Execution ---
print(f"Running pipeline in {RUN_MODE} mode...")
print(f"Processing frames {START_FRAME} to {START_FRAME + FRAME_COUNT - 1} ({FRAME_COUNT} frames total).")

# Run with auto-detection
results = run_pipeline_corrected(
    all_images_dir=ALL_IMAGES_DIR,
    annotations=ANNOTATIONS_DIR,
    working_dir=WORK_DIR,
    start_index=START_FRAME,    # Set start index
    num_frames=FRAME_COUNT,     # Set frame count
    scale_px_per_m=100, #SCALE_PX_PER_M,  # 120.0,
    fps=25.0,
    bev_out_meters=(BEV_WIDTH_M, BEV_LENGTH_M),  # Match your needs #(30.0, 30.0), #(15.0, 15.0),
    output_video='./results/corrected_result.mp4',
    annotation_order='grid',  # column!
    homography_debug=True,
    use_opencv_as_reference=True
)

print('Pipeline finished. Number of frames processed:', len(results))