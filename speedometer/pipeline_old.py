# pipeline.py
import os
import glob
import cv2
import numpy as np
import imageio
from tqdm import tqdm
import logging
import re

from .annotate import load_annotation_or_annotate
from .homographies import build_homography, warp_to_bev, stitch_two_images
from .bev_and_flow import dense_flow_farneback
from .ransac_speed import ransac_translation_improved

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def ensure_dir(d):
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# NOTE: The refine_homography function is left here but is not used
# in the current pipeline (refine_H_every_N=0).
def refine_homography(H, img_prev, img_curr):
    """
    Refines the homography H (img_prev -> world/BEV) by tracking features
    between img_prev and img_curr, and applying the found image-to-image
    transformation to H.
    """
    # ... (refinement logic, left untouched)
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(img_prev, None)
    kp2, des2 = orb.detectAndCompute(img_curr, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return H

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
    
    num_matches = min(len(matches), 50) 
    matches = matches[:num_matches]

    if num_matches < 4:
        return H

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    T, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if T is None:
        return H

    H_refined = H @ np.linalg.inv(T)
    return H_refined

# NEW FUNCTION: Key for natural sorting based on 'fn' number
def natural_sort_key(filepath):
    """Extracts the integer frame number from 'fnX.jpg' for sorting."""
    filename = os.path.basename(filepath)
    # Regex to find 'fn' followed by digits
    match = re.search(r'fn(\d+)\.jpg$', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0 # Fallback for safety

def run_pipeline_v2(
    all_images_dir=".",
    annotations="./annotations",
    working_dir='./speed_work',
    start_index=0,
    num_frames=100,
    scale_px_per_m=120.0,
    fps=25.0,
    bev_out_meters=(15.0,15.0),
    output_video='./results/result_speedometer.mp4',
    # Exposed Optical Flow parameters
    flow_params=None,
    # Homography Refinement flag (currently disabled, see refine_homography)
    refine_H_every_N=0
):
    # Ensure working_dir exists
    ensure_dir(working_dir)
    
    # Image list loading and sorting
    dev0_files = glob.glob(os.path.join(all_images_dir, "Dev0", "*.jpg")) + \
                 glob.glob(os.path.join(all_images_dir, "Dev0", "*.png"))
    dev3_files = glob.glob(os.path.join(all_images_dir, "Dev3", "*.jpg")) + \
                 glob.glob(os.path.join(all_images_dir, "Dev3", "*.png"))

    dev0_list = sorted(dev0_files, key=natural_sort_key)
    dev3_list = sorted(dev3_files, key=natural_sort_key)

    if num_frames is None:
        num_frames = min(len(dev0_list), len(dev3_list))

    sel0 = dev0_list[start_index:start_index+num_frames]
    sel3 = dev3_list[start_index:start_index+num_frames]

    logging.info(f"Using {len(sel0)} Dev0 frames and {len(sel3)} Dev3 frames")

    # --- Homography Calculation ---
    dev0_json = os.path.join(annotations, 'dev0_pts.json')
    dev3_json = os.path.join(annotations, 'dev3_pts.json')
    pts0, pts3 = load_annotation_or_annotate(sel0[0], sel3[0], dev0_json=dev0_json, dev3_json=dev3_json)

    # 90 DEGREE ROTATION FIX: Swap X and Y coordinates
    world_pts_m = np.array([
        # TOP ROW (X=0, Y changes)
        [0, 6],  # 3. Top-Right (6m away)
        [0, 3],  # 2. Top-Middle (3m away)
        [0, 0],  # 1. Top-Left
        
        # BOTTOM ROW (X=3, Y changes)
        [3, 6],   # 6. Bottom-Right
        [3, 3],  # 5. Bottom-Middle
        [3, 0],  # 4. Bottom-Left
    ], dtype=np.float32)

    # # 90 DEGREE ROTATION FIX: Swap X and Y coordinates
    # world_pts_m = np.array([
    #     # TOP ROW (X=0, Y changes)
    #     [3, 6],   # 6. Bottom-Right
    #     [0, 6],  # 3. Top-Right (6m away)
    #     [3, 3],  # 5. Bottom-Middle
    #     [0, 3],  # 2. Top-Middle (3m away)
    #     [3, 0],  # 4. Bottom-Left
    #     [0, 0]  # 1. Top-Left

    #     # BOTTOM ROW (X=3, Y changes)
    # ], dtype=np.float32)

    H0, _ = build_homography(pts0, world_pts_m, scale_px_per_m=scale_px_per_m)
    H3, _ = build_homography(pts3, world_pts_m, scale_px_per_m=scale_px_per_m)
    H3_to_0 = H0 @ np.linalg.inv(H3)

    bev_w = int(bev_out_meters[0]*scale_px_per_m)
    bev_h = int(bev_out_meters[1]*scale_px_per_m)
    out_size = (bev_w, bev_h)

    prev_bev0_gray = None
    results = []
    FLOW_DOWNSCALE_FACTOR = 0.5 

    # --- Video Writer and Interactive Mode Setup ---
    is_interactive_test = (output_video is None) and (num_frames <= 10)
    writer = None

    if output_video:
        ensure_dir(os.path.dirname(output_video))
        writer = imageio.get_writer(output_video, fps=fps)

    # --- Main Processing Loop ---
    for i in tqdm(range(num_frames)):
        img0 = cv2.imread(sel0[i])
        img3 = cv2.imread(sel3[i])
        
        if img0 is None or img3 is None:
            logging.warning(f"Skipping frame {i}: Failed to load one or both images.")
            continue

        # --- Processing Frame ---
        stitched = stitch_two_images(img0, img3, H3_to_0)
        bev0 = warp_to_bev(img0, H0, out_size)
        bev3 = warp_to_bev(img3, H3, out_size)

        translation_px = np.array([0.0,0.0])
        speed_kmh = 0.0

        bev0_low_res = cv2.resize(bev0, None, fx=FLOW_DOWNSCALE_FACTOR, fy=FLOW_DOWNSCALE_FACTOR, interpolation=cv2.INTER_LINEAR)

        if i>0 and prev_bev0_gray is not None:
            gray = cv2.cvtColor(bev0_low_res, cv2.COLOR_BGR2GRAY) 
            
            flow = dense_flow_farneback(prev_bev0_gray, gray, params=flow_params) 
            
            flow *= (1.0 / FLOW_DOWNSCALE_FACTOR)
            
            mag = np.linalg.norm(flow, axis=2)
            valid_mask = mag > 0.3
            
            t_px, _ = ransac_translation_improved(flow, mask=valid_mask)
            translation_px = t_px
            speed_m_per_frame = np.linalg.norm(t_px)/scale_px_per_m
            speed_kmh = speed_m_per_frame*fps*3.6
            prev_bev0_gray = gray
        else:
            prev_bev0_gray = cv2.cvtColor(bev0_low_res, cv2.COLOR_BGR2GRAY) 
            
        # Store results
        results.append({'stitched': stitched, 'bev0': bev0, 'bev3': bev3,
                         'translation_px': translation_px, 'speed_kmh': float(speed_kmh)})

        # --- Visualization ---
        vis_frame = compose_visual_frame_v2(stitched, bev0, bev3, translation_px, speed_kmh)

        if writer:
            # Video mode: Write frame to file
            writer.append_data(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB))
        
        elif is_interactive_test:
            # Interactive Test Mode: Display frame and wait for keypress
            # Define a scaling factor for the display window
            DISPLAY_SCALE = 0.5 
            
            # Resize the visualization frame for comfortable viewing
            display_frame = cv2.resize(vis_frame, None, 
                                     fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, 
                                     interpolation=cv2.INTER_LINEAR)

            window_title = f"Frame {start_index + i + 1} of {start_index + num_frames} (Press Any Key or 'q' to Quit)"
            cv2.imshow(window_title, display_frame) # Use the scaled frame here
            
            # Wait indefinitely until a key is pressed.
            key = cv2.waitKey(0) 
            
            # Check for quit key
            if key == ord('q'): 
                break # Exit the processing loop

    # --- Cleanup ---
    if writer:
        writer.close()
        logging.info('Video written to %s', output_video)
    elif is_interactive_test:
        cv2.destroyAllWindows() 
        
    return results

def compose_visual_frame_v2(stitched_img, bev0, bev3, translation_px, speed_kmh):
    # --- 1. Define Sizing Parameters ---
    MAIN_VIEW_WIDTH = 1080 
    SIDE_VIEW_SCALE = 0.5 
    SIDE_VIEW_WIDTH = int(MAIN_VIEW_WIDTH * SIDE_VIEW_SCALE)
    SCOREBOARD_HEIGHT = 80
    
    # --- 2. Resize and Prepare Images ---
    h_s, w_s = stitched_img.shape[:2]
    stitched_h = int(h_s * (MAIN_VIEW_WIDTH / w_s))
    stitched_resized = cv2.resize(stitched_img, (MAIN_VIEW_WIDTH, stitched_h))

    bev_h = stitched_h
    bev0_r = cv2.resize(bev0, (SIDE_VIEW_WIDTH, bev_h))
    bev3_r = cv2.resize(bev3, (SIDE_VIEW_WIDTH, bev_h))

    # --- 3. Create Canvases and Combine ---
    H_total = stitched_h + SCOREBOARD_HEIGHT
    W_total = SIDE_VIEW_WIDTH + MAIN_VIEW_WIDTH + SIDE_VIEW_WIDTH
    combined = np.zeros((H_total, W_total, 3), dtype=np.uint8)
    
    mid_start = SIDE_VIEW_WIDTH
    mid_end = mid_start + MAIN_VIEW_WIDTH
    right_start = mid_end
    
    # Placement
    combined[:bev_h, 0:SIDE_VIEW_WIDTH] = bev0_r                       # Left Panel: BEV0
    combined[:stitched_h, mid_start:mid_end] = stitched_resized        # Middle Panel: Stitched Image
    combined[:bev_h, right_start:W_total] = bev3_r                     # Right Panel: BEV3

    # --- 4. Draw Scoreboard on the Main View ---
    sb_y_start = stitched_h
    sb_y_end = H_total
    
    # Draw dark boxes for all scoreboard areas
    cv2.rectangle(combined, (0, sb_y_start), (W_total, sb_y_end), (20, 20, 20), -1)

    # Calculate center Y for text alignment
    sb_center_y = sb_y_start + SCOREBOARD_HEIGHT // 2
    
    # A. Draw Speed Text (Centered on the Main View scoreboard)
    speed_text = f"{speed_kmh:.1f} km/h"
    text_size, baseline = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)
    text_x = mid_start + (MAIN_VIEW_WIDTH - text_size[0]) // 2 # Center X
    text_y = sb_center_y + text_size[1] // 2
    
    cv2.putText(combined, speed_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4) # Yellow text

    # B. Draw Motion Vector Arrow (on the BEV0 panel)
    arrow_center_x = SIDE_VIEW_WIDTH // 2
    arrow_center_y = bev_h + SCOREBOARD_HEIGHT // 2
    arrow_scale = 10.0 # Scaling flow for visualization
    arrow_vis = (int(translation_px[0]*arrow_scale), int(translation_px[1]*arrow_scale))
    
    # Draw label
    cv2.putText(combined, "MOTION VECTOR", (arrow_center_x - 70, stitched_h + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
    # Draw arrow
    cv2.arrowedLine(combined, (arrow_center_x, arrow_center_y + 10),
                    (arrow_center_x+arrow_vis[0], arrow_center_y+arrow_vis[1] + 10),
                    (0, 0, 255), 3, tipLength=0.3) # Red arrow
                    
    # C. Labeling the Views
    cv2.putText(combined, "BEV LEFT (Dev0)", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(combined, "STITCHED MAIN VIEW", (mid_start + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(combined, "BEV RIGHT (Dev3)", (right_start + 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


    return combined