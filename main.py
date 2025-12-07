# main.py
from speedometer.pipeline_with_world_coords import run_pipeline_with_world_coords
import os
import pickle
import numpy as np
import json # Explicitly imported for the default world coord generation
import sys # For clean exit on failure

# --- DIRECTORIES ---
ALL_IMAGES_DIR = 'dataset'
ANNOTATIONS_DIR = 'annotations'
WORLD_COORDS_DIR = '../world_coordinates'  # Directory with world coordinate JSON files
WORK_DIR = 'speed_work'

# --- CONFIGURATION SWITCH ---
# OPTIONS: "FULL", "HALF1", "HALF2", "CHUNK", "TEST"
RUN_MODE = "CHUNK"
# --------------------------

# --- GEOMETRY CONFIGURATION ---
WORLD_WIDTH = 3.0   # meters (from your world coordinates)
WORLD_LENGTH = 6.0  # meters (from your world coordinates)

# OPTIMIZATION 1: REDUCE BEV RESOLUTION (Most significant speedup)
# 50 pixels per meter is half the original scale (100.0), reducing pixel count by 4x.
SCALE_PX_PER_M = 50.0  # 50 pixels per meter = faster processing

# BEV output size in meters (with 0.5m side border, 1.0m front/back border)
BEV_WIDTH_M = WORLD_WIDTH + 1.0
BEV_LENGTH_M = WORLD_LENGTH + 2.0

# World coordinate strategy: "column" or "row"
WORLD_COORDS_STRATEGY = "column"

# --- FRAME COUNT & OUTPUT SETUP ---
# NOTE: TOTAL_FRAMES should be calculated robustly or set explicitly if constant.
TOTAL_FRAMES = 3405 
HALF_FRAMES = TOTAL_FRAMES // 2 

# Configuration based on RUN_MODE
START_FRAME = 0
FRAME_COUNT = 0
OUTPUT_VIDEO = None

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
    START_FRAME = 50
    FRAME_COUNT = 100  # Run only the first 20 frames for quick testing
    OUTPUT_VIDEO = 'results/result_speedometer_test_chunk.mp4'
elif RUN_MODE == "TEST":
    START_FRAME = 0
    FRAME_COUNT = 4    # Only a few frames for quick debugging
    OUTPUT_VIDEO = 'results/result_speedometer_test.mp4' # Enable cv2.imshow interactive mode
else:
    raise ValueError(f"Invalid RUN_MODE: {RUN_MODE}")

# OPTIMIZATION 2: FASTER OPTICAL FLOW PARAMETERS
# Reduced levels and iterations for speed. winsize=13 is slightly smaller.
flow_params = dict(pyr_scale=0.5, levels=2, winsize=13, iterations=2, poly_n=5, poly_sigma=1.2, flags=0)

# --- World Coordinate File Check & Generation ---
print(f"Running pipeline in {RUN_MODE} mode...")
print(f"Processing frames {START_FRAME} to {START_FRAME + FRAME_COUNT - 1} ({FRAME_COUNT} frames total).")
os.makedirs('results', exist_ok=True)

world_coords_file = os.path.join(WORLD_COORDS_DIR, f"world_coordinates_{WORLD_COORDS_STRATEGY}.json")

if not os.path.exists(world_coords_file):
    print(f"\nWARNING: World coordinate file not found: {world_coords_file}")
    print("Generating world coordinates with default values...")
    os.makedirs(WORLD_COORDS_DIR, exist_ok=True)
    
    # Generate default world coordinates based on strategy
    if WORLD_COORDS_STRATEGY == "column":
        world_points = np.array([[0.0, 6.0], [0.0, 3.0], [0.0, 0.0], [3.0, 6.0], [3.0, 3.0], [3.0, 0.0]], dtype=np.float32)
        description = "2 columns × 3 rows grid with 3m spacing"
        width, height, cols, rows = 3.0, 6.0, 2, 3
    else: # row strategy
        world_points = np.array([[0.0, 3.0], [3.0, 3.0], [6.0, 3.0], [0.0, 0.0], [3.0, 0.0], [6.0, 0.0]], dtype=np.float32)
        description = "3 columns × 2 rows grid with 3m spacing"
        width, height, cols, rows = 6.0, 3.0, 3, 2
    
    # Create JSON structure
    data = {
        "strategy": WORLD_COORDS_STRATEGY,
        "description": description,
        "units": "meters",
        "spacing": 3.0,
        "points": [
            {
                "name": f"Point-{i+1}",
                "world_coordinates": world_points[i].tolist() + [0.0], # Adding Z=0.0
                "annotation_order": i + 1
            } for i in range(6)
        ],
        "grid_dimensions": {"width_meters": width, "height_meters": height, "columns": cols, "rows": rows}
    }
    
    with open(world_coords_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Created default world coordinates at: {world_coords_file}")
else:
    print(f"Found world coordinate file: {world_coords_file}")

# --- Pipeline Execution ---
results = run_pipeline_with_world_coords(
    all_images_dir=ALL_IMAGES_DIR,
    annotations=ANNOTATIONS_DIR,
    world_coords_dir=WORLD_COORDS_DIR,
    working_dir=WORK_DIR,
    start_index=START_FRAME,
    num_frames=FRAME_COUNT,
    scale_px_per_m=SCALE_PX_PER_M,
    fps=25.0,
    bev_out_meters=(BEV_WIDTH_M, BEV_LENGTH_M),
    output_video=OUTPUT_VIDEO,
    flow_params=flow_params,
    # OPTIMIZATION 3: DISABLE HOMOGRAPHY REFINEMENT
    refine_H_every_N=0, 
    world_coords_strategy=WORLD_COORDS_STRATEGY,
    homography_debug=False,
    # OPTIMIZATION 4: SKIP EXPENSIVE OPENCV RANSAC CHECK
    use_opencv_as_reference=False, 
    test_mode=(RUN_MODE == "TEST"),
    checkpoint_every=2 # Increased checkpointing to reduce write overhead
)

if results is None:
    print("Pipeline failed or returned no results.")
    sys.exit(1)

print('Pipeline finished. Number of frames processed:', len(results))

# --- Save and Analyze Results ---
if len(results) > 0:
    # Save to pickle
    results_file = f'results/results_{RUN_MODE}_{WORLD_COORDS_STRATEGY}.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f'Results saved to {results_file}')
    
    # Calculate speed statistics
    speeds = [r['speed_kmh'] for r in results if r['speed_kmh'] > 0]
    if speeds:
        print(f'\nSpeed statistics:')
        print(f'  Average: {np.mean(speeds):.1f} km/h')
        print(f'  Maximum: {np.max(speeds):.1f} km/h')
        print(f'  Minimum: {np.min(speeds):.1f} km/h')
        print(f'  Total frames with speed > 0: {len(speeds)}/{len(results)}')
    
    # Save a simple CSV with speeds
    csv_file = f'results/speeds_{RUN_MODE}_{WORLD_COORDS_STRATEGY}.csv'
    with open(csv_file, 'w') as f:
        f.write('frame,speed_kmh\n')
        for r in results:
            f.write(f'{r["frame_idx"]},{r["speed_kmh"]:.1f}\n')
    print(f'Speed data saved to {csv_file}')