# main.py
from speedometer.pipeline_with_world_coords import run_pipeline_with_world_coords
import os
import pickle

ALL_IMAGES_DIR = 'dataset'
ANNOTATIONS_DIR = 'annotations'
WORLD_COORDS_DIR = '../world_coordinates'  # Directory with world coordinate JSON files
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

# World coordinate strategy: "column" or "row"
# Use "column" since that's what you annotated with
WORLD_COORDS_STRATEGY = "column"

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
    FRAME_COUNT = 20  # Run only the first 100 frames for quick testing
    OUTPUT_VIDEO = 'results/result_speedometer_test_chunk.mp4'
elif RUN_MODE == "TEST":
    START_FRAME = 0
    FRAME_COUNT = 4    # Only a few frames for quick debugging
    # Set output_video to None to enable cv2.imshow interactive mode
    OUTPUT_VIDEO = None
else:
    raise ValueError(f"Invalid RUN_MODE: {RUN_MODE}")

# Parameters for the pipeline
flow_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
# SCALE_PX_PER_M = 150.0  # Increased for better BEV clarity

# --- Pipeline Execution ---
print(f"Running pipeline in {RUN_MODE} mode...")
print(f"World coordinates strategy: {WORLD_COORDS_STRATEGY}")
print(f"Processing frames {START_FRAME} to {START_FRAME + FRAME_COUNT - 1} ({FRAME_COUNT} frames total).")

# Ensure output directory exists
os.makedirs('results', exist_ok=True)

# Test if world coordinate files exist
world_coords_file = os.path.join(WORLD_COORDS_DIR, f"world_coordinates_{WORLD_COORDS_STRATEGY}.json")
if not os.path.exists(world_coords_file):
    print(f"\nWARNING: World coordinate file not found: {world_coords_file}")
    print("Please run generate_world_coordinates.py first.")
    print("Generating world coordinates with default values...")
    
    # Create world coordinates directory if it doesn't exist
    os.makedirs(WORLD_COORDS_DIR, exist_ok=True)
    
    # Generate default world coordinates
    import json
    import numpy as np
    
    if WORLD_COORDS_STRATEGY == "column":
        world_points = np.array([
            [0.0, 6.0], [0.0, 3.0], [0.0, 0.0],
            [3.0, 6.0], [3.0, 3.0], [3.0, 0.0]
        ], dtype=np.float32)
        description = "2 columns × 3 rows grid with 3m spacing"
    else:  # row strategy
        world_points = np.array([
            [0.0, 3.0], [3.0, 3.0], [6.0, 3.0],
            [0.0, 0.0], [3.0, 0.0], [6.0, 0.0]
        ], dtype=np.float32)
        description = "3 columns × 2 rows grid with 3m spacing"
    
    # Create JSON structure
    data = {
        "strategy": WORLD_COORDS_STRATEGY,
        "description": description,
        "units": "meters",
        "spacing": 3.0,
        "points": [
            {
                "name": f"Point-{i+1}",
                "world_coordinates": world_points[i].tolist() + [0.0],
                "annotation_order": i + 1
            } for i in range(6)
        ],
        "grid_dimensions": {
            "width_meters": 3.0 if WORLD_COORDS_STRATEGY == "column" else 6.0,
            "height_meters": 6.0 if WORLD_COORDS_STRATEGY == "column" else 3.0,
            "columns": 2 if WORLD_COORDS_STRATEGY == "column" else 3,
            "rows": 3 if WORLD_COORDS_STRATEGY == "column" else 2
        }
    }
    
    with open(world_coords_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Created default world coordinates at: {world_coords_file}")
else:
    print(f"Found world coordinate file: {world_coords_file}")

# Run pipeline with world coordinates
results = run_pipeline_with_world_coords(
    all_images_dir=ALL_IMAGES_DIR,
    annotations=ANNOTATIONS_DIR,
    world_coords_dir=WORLD_COORDS_DIR,
    working_dir=WORK_DIR,
    start_index=START_FRAME,    # Set start index
    num_frames=FRAME_COUNT,     # Set frame count
    scale_px_per_m=SCALE_PX_PER_M,
    fps=25.0,
    bev_out_meters=(BEV_WIDTH_M, BEV_LENGTH_M),  # Match your needs #(30.0, 30.0), #(15.0, 15.0),
    output_video=OUTPUT_VIDEO,
    flow_params=flow_params,
    world_coords_strategy=WORLD_COORDS_STRATEGY,
    homography_debug=True,
    use_opencv_as_reference=True
)

print('Pipeline finished. Number of frames processed:', len(results) if results else 0)

# Save results to pickle file for analysis
if results and len(results) > 0:
    results_file = f'results/results_{RUN_MODE}_{WORLD_COORDS_STRATEGY}.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f'Results saved to {results_file}')
    
    # Calculate speed statistics
    speeds = [r['speed_kmh'] for r in results if r['speed_kmh'] > 0]
    if speeds:
        print(f'\nSpeed statistics:')
        print(f'  Average: {sum(speeds)/len(speeds):.1f} km/h')
        print(f'  Maximum: {max(speeds):.1f} km/h')
        print(f'  Minimum: {min(speeds):.1f} km/h')
        print(f'  Total frames with speed > 0: {len(speeds)}/{len(results)}')
    
    # Save a simple CSV with speeds
    csv_file = f'results/speeds_{RUN_MODE}_{WORLD_COORDS_STRATEGY}.csv'
    with open(csv_file, 'w') as f:
        f.write('frame,speed_kmh\n')
        for i, r in enumerate(results):
            f.write(f'{i},{r["speed_kmh"]:.1f}\n')
    print(f'Speed data saved to {csv_file}')