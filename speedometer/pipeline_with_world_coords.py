# pipeline_with_world_coords.py
import os
import glob
import cv2
import numpy as np
import imageio
from tqdm import tqdm
import logging
import re
import json
import numpy.linalg

from .annotate import load_annotation_or_annotate_final
from .bev_and_flow import dense_flow_farneback
from .ransac_speed import ransac_translation_improved
from .homographies import stitch_two_images

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class ExponentialMovingAverage:
    """Simple class for smoothing a signal using EMA."""
    def __init__(self, alpha):
        # Alpha (smoothing factor) closer to 1 means less smoothing (more reactive)
        # Closer to 0 means more smoothing (less reactive)
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

def load_world_coordinates(strategy="column", world_coords_dir="../world_coordinates"):
    """Load world coordinates (Unchanged)."""
    if strategy == "column":
        json_file = os.path.join(world_coords_dir, "world_coordinates_column.json")
        npy_file = os.path.join(world_coords_dir, "world_coordinates_column.npy")
    elif strategy == "row":
        json_file = os.path.join(world_coords_dir, "world_coordinates_row.json")
        npy_file = os.path.join(world_coords_dir, "world_coordinates_row.npy")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    if os.path.exists(npy_file):
        world_points = np.load(npy_file)
        metadata = {}
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                metadata = json.load(f)
        return world_points[:, :2], metadata
    
    elif os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        world_points = []
        for point_data in data.get("points", []):
            world_coords = point_data.get("world_coordinates", [0, 0, 0])
            world_points.append(world_coords[:2])
        return np.array(world_points, dtype=np.float32), data
    else:
        raise FileNotFoundError(f"World coordinate files not found")

# --- Standard Homography Functions ---
def calculate_normalization_matrix(points):
    points = np.asarray(points, dtype=np.float32)
    mean = np.mean(points, axis=0)
    std = np.std(points, axis=0)
    std_avg = np.mean(std) if np.mean(std) > 1e-10 else 1.0
    scale = np.sqrt(2) / std_avg
    offset = -scale * mean
    return np.array([[scale, 0, offset[0]], [0, scale, offset[1]], [0, 0, 1]])

def normalize_points(points, T):
    points = np.asarray(points, dtype=np.float32)
    pts_h = np.column_stack((points, np.ones(len(points))))
    norm_pts = (T @ pts_h.T).T
    return norm_pts[:, :2]

def calculate_homography_dlt(point_pairs):
    A = []
    for (x, y), (xp, yp) in point_pairs:
        A.append([x, y, 1, 0, 0, 0, -x*xp, -y*xp, -xp])
        A.append([0, 0, 0, x, y, 1, -x*yp, -y*yp, -yp])
    U, S, Vt = np.linalg.svd(np.array(A))
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2] if abs(H[2, 2]) > 1e-10 else H

def calculate_normalized_homography(pts_img, pts_world):
    try:
        T_img = calculate_normalization_matrix(pts_img)
        T_world = calculate_normalization_matrix(pts_world)
        norm_img = normalize_points(pts_img, T_img)
        norm_world = normalize_points(pts_world, T_world)
        H_norm = calculate_homography_dlt(list(zip(norm_img, norm_world)))
        H = np.linalg.inv(T_world) @ H_norm @ T_img
        return H / H[2, 2]
    except Exception as e:
        logging.error(f"Homography error: {e}")
        return None

def ensure_dir(d):
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# OPTIMIZATION: Faster Color Matching
def match_color_fast(src, ref):
    """
    Matches the color distribution of 'src' to 'ref' using mean/std.
    Applied on raw images BEFORE warping.
    FIXED: Explicitly casts numpy array values to scalars for cv2.addWeighted.
    """
    src = src.astype(np.float32)
    ref = ref.astype(np.float32)
    
    # Compute stats
    mean_s, std_s = cv2.meanStdDev(src)
    mean_r, std_r = cv2.meanStdDev(ref)
    
    res = np.zeros_like(src)
    for c in range(3):
        # ERROR FIX: meanStdDev returns shape (3, 1). 
        # We must extract the scalar float value using [0]
        s_std_val = std_s[c][0]
        r_std_val = std_r[c][0]
        s_mean_val = mean_s[c][0]
        r_mean_val = mean_r[c][0]

        if s_std_val > 1.0:
            gain = r_std_val / (s_std_val + 1e-6)
            bias = r_mean_val - s_mean_val * gain
            
            # cv2.addWeighted requires simple scalars (floats), not numpy arrays
            res[..., c] = cv2.addWeighted(src[..., c], float(gain), src[..., c], 0, float(bias))
        else:
            res[..., c] = src[..., c]
            
    return np.clip(res, 0, 255).astype(np.uint8)

# OPTIMIZATION: Optimized Stitching
def stitch_two_images_optimized(img0, img3, H3_to_0):
    """
    Optimized stitching.
    """
    h0, w0 = img0.shape[:2]
    h3, w3 = img3.shape[:2]

    # Calculate dimensions of warped img3
    corners3 = np.float32([[0,0], [w3,0], [w3,h3], [0,h3]]).reshape(-1,1,2)
    corners3_warped = cv2.perspectiveTransform(corners3, H3_to_0)
    
    # Calculate union bounding box
    all_x = np.concatenate(([0, w0], corners3_warped[:,:,0].flatten()))
    all_y = np.concatenate(([0, h0], corners3_warped[:,:,1].flatten()))
    min_x, max_x = int(np.floor(all_x.min())), int(np.ceil(all_x.max()))
    min_y, max_y = int(np.floor(all_y.min())), int(np.ceil(all_y.max()))

    # Translation matrix to shift everything to positive coordinates
    tx, ty = -min_x, -min_y
    T_shift = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
    
    final_w, final_h = max_x - min_x, max_y - min_y
    
    # Combine H with shift
    H_final_3 = T_shift @ H3_to_0
    
    # Warp img3
    warped3 = cv2.warpPerspective(img3, H_final_3, (final_w, final_h))
    
    # Warp img0
    warped0 = cv2.warpPerspective(img0, T_shift, (final_w, final_h))

    # Create masks (check for non-black pixels)
    mask0 = warped0[..., 0] > 0
    mask3 = warped3[..., 0] > 0
    
    # Initial result is just warped0
    result = warped0.copy()
    
    # Where mask3 exists but mask0 doesn't, copy warped3
    np.copyto(result, warped3, where=mask3[..., None] & ~mask0[..., None])
    
    # Where they overlap, blend
    overlap = mask0 & mask3
    if overlap.any():
        blended = cv2.addWeighted(warped0[overlap], 0.5, warped3[overlap], 0.5, 0)
        result[overlap] = blended

    # Crop black borders (ROI calculation)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(thresh)
    
    pad = 10
    x, y = max(0, x-pad), max(0, y-pad)
    w, h = min(final_w-x, w+2*pad), min(final_h-y, h+2*pad)
    
    cropped = result[y:y+h, x:x+w]
    
    # # Final resize to 1920x1080
    # target_w, target_h = 1920, 1080
    
    # # Use INTER_LINEAR for speed
    # if cropped.size == 0:
    #     return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    # return resized

    # Return the cropped union directly (preserves natural aspect ratio)
    if cropped.size == 0:
        return np.zeros((720, 1280, 3), dtype=np.uint8)  # Fallback
    return cropped

def refine_homography_with_features(H, img_prev, img_curr):
    """Refines homography."""
    if img_prev is None or img_curr is None: return H
    
    scale = 0.5
    small_prev = cv2.resize(img_prev, None, fx=scale, fy=scale)
    small_curr = cv2.resize(img_curr, None, fx=scale, fy=scale)
    
    detector = cv2.ORB_create(nfeatures=500)
    kp1, des1 = detector.detectAndCompute(small_prev, None)
    kp2, des2 = detector.detectAndCompute(small_curr, None)
    
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4: return H
    
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:50]
    
    if len(matches) < 4: return H
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2) / scale
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2) / scale
    
    T, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if T is None: return H
    return H @ np.linalg.inv(T)

def natural_sort_key(filepath):
    filename = os.path.basename(filepath)
    match = re.search(r'fn(\d+)\.(jpg|png|jpeg)$', filename, re.IGNORECASE)
    return int(match.group(1)) if match else 0

def run_pipeline_with_world_coords(
    all_images_dir=".",
    annotations="./annotations",
    world_coords_dir="../world_coordinates",
    working_dir='./speed_work',
    start_index=0,
    num_frames=100,
    scale_px_per_m=150.0,
    fps=25.0,
    bev_out_meters=(15.0, 15.0),
    output_video='./results/result_speedometer.mp4',
    flow_params=None,
    refine_H_every_N=0,
    world_coords_strategy="column",
    enable_auto_roll_correction=False,
    checkpoint_every=100
):
    ensure_dir(working_dir)

    # --- Load images ---
    dev0_files = sorted(glob.glob(os.path.join(all_images_dir, "Dev0", "*.*")), key=natural_sort_key)
    dev3_files = sorted(glob.glob(os.path.join(all_images_dir, "Dev3", "*.*")), key=natural_sort_key)
    
    dev0_files = [f for f in dev0_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    dev3_files = [f for f in dev3_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if num_frames is None: num_frames = min(len(dev0_files), len(dev3_files))
    sel0 = dev0_files[start_index:start_index + num_frames]
    sel3 = dev3_files[start_index:start_index + num_frames]

    if not sel0 or not sel3:
        logging.error("No images found!")
        return []

    # --- Load Coordinates & Homography ---
    world_points_meters, _ = load_world_coordinates(world_coords_strategy, world_coords_dir)
    world_points_pixels = world_points_meters * scale_px_per_m

    pts0, pts3, _ = load_annotation_or_annotate_final(sel0[0], sel3[0],
                                                      dev0_json=os.path.join(annotations, 'dev0_pts.json'),
                                                      dev3_json=os.path.join(annotations, 'dev3_pts.json'))
    
    n_points = min(len(pts0), len(world_points_meters))
    pts0, pts3, world_points_pixels = pts0[:n_points], pts3[:n_points], world_points_pixels[:n_points]

    H0 = calculate_normalized_homography(pts0, world_points_pixels)
    H3 = calculate_normalized_homography(pts3, world_points_pixels)
    
    if H0 is None or H3 is None: return []

    # --- Roll Correction Logic ---
    if enable_auto_roll_correction:
        logging.info("Calculating auto-roll correction...")
        img0 = cv2.imread(sel0[0], cv2.IMREAD_GRAYSCALE)
        img3 = cv2.imread(sel3[0], cv2.IMREAD_GRAYSCALE)
        orb = cv2.ORB_create(nfeatures=1000)
        kp0, des0 = orb.detectAndCompute(img0, None)
        kp3, des3 = orb.detectAndCompute(img3, None)
        
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des0, des3)
        matches = sorted(matches, key=lambda x: x.distance)[:50]
        
        if len(matches) > 10:
            src = np.float32([kp0[m.queryIdx].pt for m in matches])
            dst = np.float32([kp3[m.trainIdx].pt for m in matches])
            M, _ = cv2.estimateAffinePartial2D(src, dst)
            if M is not None:
                angle = np.arctan2(M[1,0], M[0,0]) * 180 / np.pi
                logging.info(f"Roll correction found: {angle:.2f} degrees")
                h, w = img3.shape
                R_fix = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
                R_fix_homo = np.vstack([R_fix, [0, 0, 1]])
                H3 = H3 @ np.linalg.inv(R_fix_homo)

    H3_to_0 = np.linalg.inv(H0) @ H3
    H0_curr, H3_curr, H3_to_0_curr = H0.copy(), H3.copy(), H3_to_0.copy()

    # --- BEV Setup ---
    bev_w = int(bev_out_meters[0] * scale_px_per_m)
    bev_h = int(bev_out_meters[1] * scale_px_per_m)
    out_size = (bev_w, bev_h)

    # --- Video Writer ---
    writer = None
    if output_video:
        ensure_dir(os.path.dirname(output_video))
        writer = imageio.get_writer(output_video, fps=fps, codec='libx264', quality=8, pixelformat='yuv420p')

    # --- Processing Loop ---
    prev_bev0_gray = None
    frames_buffer = []
    results = []
    
    flow_scale = 0.5 

    # Initialize the EMA filter for speed smoothing (alpha=0.1 for strong smoothing)
    speed_smoother = ExponentialMovingAverage(alpha=0.1) 
    # Initialize a smoother for the translation vector (alpha=0.2 for less smoothing)
    translation_smoother = ExponentialMovingAverage(alpha=0.2)
    
    # ----------------------------------------------------
    # START: TRY...FINALLY BLOCK FOR SAFE VIDEO CLOSURE
    # ----------------------------------------------------
    try:
        for i in tqdm(range(num_frames), desc="Processing"):
            img0 = cv2.imread(sel0[i])
            img3 = cv2.imread(sel3[i])
            
            if img0 is None or img3 is None: continue

            # Refine Homography
            if refine_H_every_N > 0 and i > 0 and i % refine_H_every_N == 0:
                img0_prev = cv2.imread(sel0[i-1])
                img3_prev = cv2.imread(sel3[i-1])
                H0_curr = refine_homography_with_features(H0_curr, img0_prev, img0)
                H3_curr = refine_homography_with_features(H3_curr, img3_prev, img3)
                H3_to_0_curr = np.linalg.inv(H0_curr) @ H3_curr

            # OPTIMIZATION: Color Match 
            img3 = match_color_fast(img3, img0)
            img0 = match_color_fast(img0, img3)  

            # OPTIMIZATION: Stitching
            stitched = stitch_two_images(img0, img3, H3_to_0_curr)

            # After (using INTER_CUBIC for better quality at the cost of slight speed):
            bev0 = cv2.warpPerspective(img0, H0_curr, out_size, flags=cv2.INTER_CUBIC)
            bev3 = cv2.warpPerspective(img3, H3_curr, out_size, flags=cv2.INTER_CUBIC)

            # --- Speed Calculation (Optimized) ---
            translation_px = np.array([0.0, 0.0])
            speed_kmh = 0.0
            
            # Downscale BEV for flow calculation
            bev0_small_gray = cv2.resize(cv2.cvtColor(bev0, cv2.COLOR_BGR2GRAY), None, fx=flow_scale, fy=flow_scale)
            
            if i > 0 and prev_bev0_gray is not None:
                flow = dense_flow_farneback(prev_bev0_gray, bev0_small_gray, params=flow_params or {})
                mag = np.linalg.norm(flow, axis=2)
                valid = mag > 0.5
                if np.sum(valid) > 10:
                    t_px, inliers = ransac_translation_improved(flow, mask=valid)
                    if inliers is not None and np.sum(inliers) > 5:
                        raw_translation_px = t_px / flow_scale

                        # Smooth the translation vector
                        translation_px = translation_smoother.update(raw_translation_px)
                        
                        # Apply minimum translation threshold based on scale
                        if scale_px_per_m <= 60:
                            min_translation_threshold_px = 0.8
                        elif scale_px_per_m <= 120:
                            min_translation_threshold_px = 1.2
                        else:
                            min_translation_threshold_px = 1.5  # Safe for 150+

                        if np.linalg.norm(translation_px) < min_translation_threshold_px:
                            translation_px = np.array([0.0, 0.0])
                            
                        raw_speed_kmh = np.linalg.norm(translation_px) / scale_px_per_m * fps * 3.6
                        # Speed clipping for better smoothing
                        raw_speed_kmh = min(raw_speed_kmh, 200.0)  # Cap at 200 km/h 
                        # Smooth the final speed value
                        speed_kmh = speed_smoother.update(raw_speed_kmh)
            
            prev_bev0_gray = bev0_small_gray

            # Vis
            # vis_frame = compose_visual_frame(stitched, bev0, bev3, translation_px, speed_kmh)
            vis_frame = compose_visual_frame(stitched, bev0, bev3, translation_px, speed_kmh, out_size)
            frames_buffer.append(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB))
            results.append({'speed_kmh': float(speed_kmh), 'frame_idx': start_index + i})

            # Save/Flush
            # FLUSH THE REMAINING FRAMES IN THE BUFFER BEFORE EXITING IF INTERRUPTED
            if (i + 1) % checkpoint_every == 20 or (i + 1) == num_frames:
                if writer:
                    for f in frames_buffer: writer.append_data(f)
                    frames_buffer = []

            # Preview
            disp = cv2.resize(vis_frame, (960, 540), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Speedometer", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                # Using break to jump to the finally block for safe closing
                break

    except KeyboardInterrupt:
        logging.warning("Pipeline interrupted by user. Finalizing video file...")
        
    finally:
        # This code always executes, even after a break, return, or KeyboardInterrupt
        
        # 1. Flush any remaining frames in the buffer one last time
        if writer and frames_buffer:
             for f in frames_buffer: writer.append_data(f)
             frames_buffer = [] # Clear for safety

        # 2. CRITICAL: Close the writer to write the video file footer/metadata
        if writer:
            writer.close()
            logging.info(f"Video file closed successfully. Partial video saved to {output_video}.")
            
        # 3. Close the OpenCV display windows
        cv2.destroyAllWindows()
    # ----------------------------------------------------
    # END: TRY...FINALLY BLOCK
    # ----------------------------------------------------

    return results

def compose_visual_frame(stitched_img, bev0, bev3, translation_px, speed_kmh, bev_size):
    """Optimized visual frame composition, now including motion vector visualization."""
    H_total, W_total = 720, 1280 
    
    combined = np.zeros((H_total, W_total, 3), dtype=np.uint8)
    
    sb_h = 60
    main_h = H_total - sb_h
    
    side_w = int(W_total * 0.25)
    main_w = W_total - (2 * side_w)

    # --- 1. Draw Motion Vector on BEV0 ---
    bev_w, bev_h = bev_size
    
    # Calculate vector start (center of BEV)
    center_x, center_y = bev_w // 2, bev_h // 2
    
    # Vector scaling for VISUAL CLARITY
    vector_magnitude = np.linalg.norm(translation_px)
    if vector_magnitude > 0:
        # Scale the vector by a factor (e.g., 5-10 times) for better visibility
        # Clamp the visible length to prevent it from dominating the BEV
        visual_scale_factor = 8.0 
        
        # Calculate scaled and clamped translation
        scaled_translation_px = translation_px * visual_scale_factor
        scaled_magnitude = np.linalg.norm(scaled_translation_px)

        # Clamp the vector length to max_vis_length (e.g., 100 pixels)
        max_vis_length = 100
        if scaled_magnitude > max_vis_length:
            scaled_translation_px = scaled_translation_px * (max_vis_length / scaled_magnitude)

        # Calculate vector end
        end_x = int(center_x + scaled_translation_px[0])
        end_y = int(center_y + scaled_translation_px[1])
    
        # Draw the main translation vector (Yellow line)
        cv2.arrowedLine(bev0, (center_x, center_y), (end_x, end_y), 
                        (0, 255, 255), 5, tipLength=0.3)
    
    # Draw a small center dot for reference (Cyan circle)
    cv2.circle(bev0, (center_x, center_y), 5, (255, 255, 0), -1)
    # --- End Draw Motion Vector ---
    
    # Resize and combine frames
    bev0_r = cv2.resize(bev0, (side_w, main_h), interpolation=cv2.INTER_LINEAR)
    bev3_r = cv2.resize(bev3, (side_w, main_h), interpolation=cv2.INTER_LINEAR)
    # stitched_r = cv2.resize(stitched_img, (main_w, main_h), interpolation=cv2.INTER_LINEAR)
    # 
    h_st, w_st = stitched_img.shape[:2]
    scale = min(main_w / w_st, main_h / h_st)
    new_w = int(w_st * scale)
    new_h = int(h_st * scale)
    stitched_resized = cv2.resize(stitched_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Center it in the main panel
    stitched_r = np.zeros((main_h, main_w, 3), dtype=np.uint8)
    offset_x = (main_w - new_w) // 2
    offset_y = (main_h - new_h) // 2
    stitched_r[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = stitched_resized

    combined[0:main_h, 0:side_w] = bev0_r
    combined[0:main_h, side_w:side_w+main_w] = stitched_r
    combined[0:main_h, side_w+main_w:] = bev3_r
    
    # Status bar
    cv2.rectangle(combined, (0, main_h), (W_total, H_total), (20, 20, 20), -1)

    speed_text = f"{speed_kmh:.1f} km/h"

    motion_text = f"Motion: X={translation_px[0]:.1f} | Y={translation_px[1]:.1f} px"

    # Draw speed text
    cv2.putText(combined, speed_text, (W_total//2 - 150, main_h + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)

    # Draw motion text
    cv2.putText(combined, motion_text, (50, main_h + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                
    cv2.putText(combined, "Left BEV (Motion)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(combined, "Stitched View", (side_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(combined, "Right BEV", (side_w + main_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    return combined

if __name__ == "__main__":
    # Example usage
    run_pipeline_with_world_coords()