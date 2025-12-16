# main.py
from speedometer.pipeline_with_world_coords import run_pipeline_with_world_coords
import os
import pickle
import numpy as np
import json 
import sys 

# --- DIRECTORIES ---
ALL_IMAGES_DIR = 'dataset'
ANNOTATIONS_DIR = 'annotations'
WORLD_COORDS_DIR = '../world_coordinates'  
WORK_DIR = 'speed_work'

# --- CONFIGURATION SWITCH ---
# OPTIONS: "FULL", "HALF1", "HALF2", "CHUNK", "TEST"
RUN_MODE = "TEST"
# --------------------------

# --- GEOMETRY CONFIGURATION ---
WORLD_WIDTH = 3.0   
WORLD_LENGTH = 6.0  

# OPTIMIZATION 1: REDUCE BEV RESOLUTION 
SCALE_PX_PER_M = 150.0  

# BEV output size in meters
BEV_WIDTH_M = WORLD_WIDTH + 1.5
BEV_LENGTH_M = WORLD_LENGTH + 2.0

# World coordinate strategy: "column" or "row"
WORLD_COORDS_STRATEGY = "column"

# --- FRAME COUNT & OUTPUT SETUP ---
TOTAL_FRAMES = 3405 
HALF_FRAMES = TOTAL_FRAMES // 2 

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
    OUTPUT_VIDEO = 'results/result_speedometer_half_1.mp4'
elif RUN_MODE == "HALF2":
    START_FRAME = HALF_FRAMES
    FRAME_COUNT = TOTAL_FRAMES - HALF_FRAMES
    OUTPUT_VIDEO = 'results/result_speedometer_half_2.mp4'
elif RUN_MODE == "CHUNK":
    START_FRAME = 0
    FRAME_COUNT = 50 
    OUTPUT_VIDEO = 'results/result_speedometer_chunk.mp4'
elif RUN_MODE == "TEST":
    START_FRAME = 0
    FRAME_COUNT = 100    # 100 so that we can see the speed change
    OUTPUT_VIDEO = 'results/result_speedometer_test.mp4' 
else:
    raise ValueError(f"Invalid RUN_MODE: {RUN_MODE}")

# OPTIMIZATION 2: FASTER OPTICAL FLOW PARAMETERS
flow_params = dict(pyr_scale=0.5, levels=2, winsize=13, iterations=2, poly_n=5, poly_sigma=1.2, flags=0)

# # OPTIMIZATION 2: FASTER OPTICAL FLOW PARAMETERS
# flow_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=7, poly_sigma=1.5, flags=0)

# --- World Coordinate File Check ---
print(f"Running pipeline in {RUN_MODE} mode...")
print(f"Processing frames {START_FRAME} to {START_FRAME + FRAME_COUNT - 1} ({FRAME_COUNT} frames total).")
os.makedirs('results', exist_ok=True)
os.makedirs(WORLD_COORDS_DIR, exist_ok=True) # Ensure dir exists

world_coords_file = os.path.join(WORLD_COORDS_DIR, f"world_coordinates_{WORLD_COORDS_STRATEGY}.json")

# Generate default world coordinates if missing
if not os.path.exists(world_coords_file):
    print(f"\nWARNING: World coordinate file not found: {world_coords_file}")
    print("Generating world coordinates with default values...")
    
    if WORLD_COORDS_STRATEGY == "column":
        world_points = np.array([[0.0, 6.0], [0.0, 3.0], [0.0, 0.0], [3.0, 6.0], [3.0, 3.0], [3.0, 0.0]], dtype=np.float32)
        description = "2 columns x 3 rows grid with 3m spacing"
        width, height, cols, rows = 3.0, 6.0, 2, 3
    else: # row strategy
        world_points = np.array([[0.0, 3.0], [3.0, 3.0], [6.0, 3.0], [0.0, 0.0], [3.0, 0.0], [6.0, 0.0]], dtype=np.float32)
        description = "3 columns x 2 rows grid with 3m spacing"
        width, height, cols, rows = 6.0, 3.0, 3, 2
    
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
        "grid_dimensions": {"width_meters": width, "height_meters": height, "columns": cols, "rows": rows}
    }
    
    with open(world_coords_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Created default world coordinates at: {world_coords_file}")
else:
    print(f"Found world coordinate file: {world_coords_file}")

# --- RUN PIPELINE ---
results = run_pipeline_with_world_coords(
    all_images_dir=ALL_IMAGES_DIR,
    annotations=ANNOTATIONS_DIR,
    world_coords_dir=WORLD_COORDS_DIR,
    working_dir=WORK_DIR,
    start_index=START_FRAME,
    num_frames=FRAME_COUNT,
    scale_px_per_m=SCALE_PX_PER_M,
    fps=4.0,
    bev_out_meters=(BEV_WIDTH_M, BEV_LENGTH_M),
    output_video=OUTPUT_VIDEO,
    flow_params=flow_params,
    refine_H_every_N=0, 
    world_coords_strategy=WORLD_COORDS_STRATEGY,
    enable_auto_roll_correction=False, # Set to True to try the auto-tilt fix again
    checkpoint_every=50,  # More frequent checkpoints for safety
)

if results is None:
    print("Pipeline failed or returned no results.")
    sys.exit(1)

print('Pipeline finished. Number of frames processed:', len(results))

# --- Save and Analyze Results ---
if len(results) > 0:
    results_file = f'results/results_{RUN_MODE}_{WORLD_COORDS_STRATEGY}.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f'Results saved to {results_file}')
    
    speeds = [r['speed_kmh'] for r in results if r['speed_kmh'] > 0]
    if speeds:
        print(f'\nSpeed statistics:')
        print(f'  Average: {np.mean(speeds):.1f} km/h')
        print(f'  Maximum: {np.max(speeds):.1f} km/h')
        print(f'  Minimum: {np.min(speeds):.1f} km/h')
        print(f'  Total frames with speed > 0: {len(speeds)}/{len(results)}')
    
    csv_file = f'results/speeds_{RUN_MODE}_{WORLD_COORDS_STRATEGY}.csv'
    with open(csv_file, 'w') as f:
        f.write('frame,speed_kmh\n')
        for r in results:
            f.write(f'{r["frame_idx"]},{r["speed_kmh"]:.1f}\n')
    print(f'Speed data saved to {csv_file}')