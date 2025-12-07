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

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_world_coordinates(strategy="column", world_coords_dir="../world_coordinates"):
    """
    Load world coordinates from JSON file based on strategy.
    
    Args:
        strategy: "column" or "row"
        world_coords_dir: Directory containing world coordinate files
    
    Returns:
        world_points: Array of world coordinates (n, 2) in meters
        metadata: Dictionary with additional information
    """
    if strategy == "column":
        json_file = os.path.join(world_coords_dir, "world_coordinates_column.json")
        npy_file = os.path.join(world_coords_dir, "world_coords_column.npy")
    elif strategy == "row":
        json_file = os.path.join(world_coords_dir, "world_coordinates_row.json")
        npy_file = os.path.join(world_coords_dir, "world_coords_row.npy")
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'column' or 'row'")
    
    # Try to load from numpy file first (simpler)
    if os.path.exists(npy_file):
        world_points = np.load(npy_file)
        logging.info(f"Loaded world coordinates from {npy_file}")
        logging.info(f"Shape: {world_points.shape}")
        
        # Load metadata from JSON
        metadata = {}
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                metadata = json.load(f)
        return world_points[:, :2], metadata  # Use only X,Y for 2D homography
    
    # Fallback to JSON
    elif os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        world_points = []
        for point_data in data.get("points", []):
            world_coords = point_data.get("world_coordinates", [0, 0, 0])
            world_points.append(world_coords[:2])  # Use only X,Y for 2D homography
        
        world_points = np.array(world_points, dtype=np.float32)
        logging.info(f"Loaded world coordinates from {json_file}")
        logging.info(f"Strategy: {data.get('strategy', 'unknown')}")
        logging.info(f"Spacing: {data.get('spacing', 'unknown')} meters")
        
        return world_points, data
    
    else:
        raise FileNotFoundError(f"World coordinate files not found for strategy: {strategy}")

def calculate_normalization_matrix(points):
    """Calculate normalization matrix for points."""
    points = np.asarray(points, dtype=np.float32)
    mean = np.mean(points, axis=0)
    std = np.std(points, axis=0)
    std_avg = np.mean(std)
    
    if std_avg < 1e-10:
        std_avg = 1.0
    
    scale = np.sqrt(2) / std_avg
    offset = -scale * mean
    
    T = np.array([[scale, 0, offset[0]],
                  [0, scale, offset[1]],
                  [0, 0, 1]])
    return T

def normalize_points(points, T):
    """Normalize points using transformation matrix T."""
    points = np.asarray(points, dtype=np.float32)
    normalized_points = []
    for point in points:
        point_homogeneous = np.append(point, 1)
        normalized_point = T @ point_homogeneous
        normalized_points.append(normalized_point[:2] / normalized_point[2])
    return np.array(normalized_points)

def calculate_homography_dlt(point_pairs):
    """Calculate homography using normalized DLT with proper SVD."""
    A = []
    for point1, point2 in point_pairs:
        x, y = point1[0], point1[1]
        xp, yp = point2[0], point2[1]
        A.append([x, y, 1, 0, 0, 0, -x*xp, -y*xp, -xp])
        A.append([0, 0, 0, x, y, 1, -x*yp, -y*yp, -yp])
    A = np.array(A)
    
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    
    if abs(H[2, 2]) > 1e-10:
        H = H / H[2, 2]
    return H

def calculate_normalized_homography(pts_img, pts_world, debug=False):
    """
    Calculate homography using normalized DLT.
    
    Args:
        pts_img: Image points (pixels)
        pts_world: World points (meters, will be converted to pixels)
        debug: Enable debug output
    
    Returns:
        Homography matrix H
    """
    if len(pts_img) < 4:
        print(f"ERROR: Need at least 4 points, got {len(pts_img)}")
        return None
    
    if len(pts_img) != len(pts_world):
        print(f"ERROR: Number of points mismatch: {len(pts_img)} image points vs {len(pts_world)} world points")
        return None
    
    try:
        # Normalize image and world points separately
        T_img = calculate_normalization_matrix(pts_img)
        T_world = calculate_normalization_matrix(pts_world)
        
        normalized_pts_img = normalize_points(pts_img, T_img)
        normalized_pts_world = normalize_points(pts_world, T_world)
        
        normalized_point_pairs = list(zip(normalized_pts_img, normalized_pts_world))
        H_normalized = calculate_homography_dlt(normalized_point_pairs)
        
        H = np.linalg.inv(T_world) @ H_normalized @ T_img
        
        if abs(H[2, 2]) > 1e-10:
            H = H / H[2, 2]
        
        return H
        
    except np.linalg.LinAlgError as e:
        print(f"ERROR in homography calculation: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def validate_homography(H, src_pts, dst_pts, name="Homography", threshold=5.0):
    """Validate homography by projecting points and checking error."""
    if H is None:
        print(f"{name}: ERROR - Homography is None!")
        return False, float('inf')
    
    src_homo = np.column_stack([src_pts, np.ones(len(src_pts))])
    projected = H @ src_homo.T
    projected = projected[:2] / projected[2]
    projected = projected.T
    
    errors = np.linalg.norm(projected - dst_pts, axis=1)
    avg_error = np.mean(errors)
    max_error = np.max(errors)
    
    print(f"\n{name} validation:")
    print(f"  Average reprojection error: {avg_error:.4f} px")
    print(f"  Maximum reprojection error: {max_error:.4f} px")
    print(f"  Individual errors: {errors}")
    
    is_valid = avg_error < threshold
    if not is_valid:
        print(f"  WARNING: Reprojection error exceeds threshold of {threshold} px!")
    
    return is_valid, avg_error

def ensure_dir(d):
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def warp_to_bev(img, H, out_size):
    """Warps an image to the BEV using homography H."""
    return cv2.warpPerspective(img, H, out_size)

def stitch_two_images_proper(img0, img3, H3_to_0=None, pts0=None, pts3=None):
    """Natural human-eye-like panoramic stitch - clean 1920x1080 output"""
    if H3_to_0 is None:
        return np.hstack([img0, img3])

    H = H3_to_0.astype(np.float32)
    h0, w0 = img0.shape[:2]
    h3, w3 = img3.shape[:2]

    corners = np.float32([[0,0], [w3,0], [w3,h3], [0,h3]]).reshape(-1,1,2)
    corners_warped = cv2.perspectiveTransform(corners, H)
    all_x = np.concatenate(([0, w0], corners_warped[:,:,0].flatten()))
    all_y = np.concatenate(([0, h0], corners_warped[:,:,1].flatten()))
    min_x, min_y = int(np.floor(all_x.min())), int(np.floor(all_y.min()))
    max_x, max_y = int(np.ceil(all_x.max())), int(np.ceil(all_y.max()))
    tx = max(0, -min_x)
    ty = max(0, -min_y)
    translation = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
    final_w = max_x - min_x + 1
    final_h = max_y - min_y + 1
    H_final = translation @ H

    warped3 = cv2.warpPerspective(img3, H_final, (final_w, final_h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0,0,0))
    warped0 = cv2.warpPerspective(img0, translation, (final_w, final_h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0,0,0))

    def match_color(src, ref):
        src = src.astype(np.float32)
        ref = ref.astype(np.float32)
        for c in range(3):
            s_mean, s_std = src[..., c].mean(), src[..., c].std()
            r_mean, r_std = ref[..., c].mean(), ref[..., c].std()
            if s_std > 1.0:
                src[..., c] = (src[..., c] - s_mean) * (r_std / (s_std + 1e-6)) + r_mean
        return np.clip(src, 0, 255).astype(np.uint8)
    warped3 = match_color(warped3, warped0)

    mask0 = (warped0 > 30).any(axis=2)
    mask3 = (warped3 > 30).any(axis=2)

    result = warped0.copy()
    overlap = mask0 & mask3
    if overlap.any():
        blended = cv2.addWeighted(warped0[overlap], 0.6, warped3[overlap], 0.4, 0)
        result[overlap] = blended
    result[mask3 & ~mask0] = warped3[mask3 & ~mask0]

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(np.vstack(contours))
        pad = 30
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(final_w - x, w + 2*pad)
        h = min(final_h - y, h + 2*pad)
        cropped = result[y:y+h, x:x+w]
    else:
        cropped = result

    target_w, target_h = 1920, 1080
    scale = min(target_w / cropped.shape[1], target_h / cropped.shape[0])
    new_w, new_h = int(cropped.shape[1] * scale), int(cropped.shape[0] * scale)
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    final = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    oy = (target_h - new_h) // 2
    ox = (target_w - new_w) // 2
    final[oy:oy+new_h, ox:ox+new_w] = resized
    return final
    
def refine_homography_with_features(H, img_prev, img_curr, method='orb'):
    """
    Refines the homography H (img_prev -> world/BEV) by tracking features.
    """
    if img_prev is None or img_curr is None:
        return H
    
    if len(img_prev.shape) == 3:
        gray_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(img_curr, cv2.COLOR_BGR2GRAY)
    else:
        gray_prev = img_prev
        gray_curr = img_curr
    
    detector = cv2.ORB_create(nfeatures=500)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    kp1, des1 = detector.detectAndCompute(gray_prev, None)
    kp2, des2 = detector.detectAndCompute(gray_curr, None)
    
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return H
    
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:50]
    
    if len(matches) < 4:
        return H
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    T, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if T is None:
        return H
    
    H_refined = H @ np.linalg.inv(T)
    return H_refined

def natural_sort_key(filepath):
    """Extracts the integer frame number from 'fnX.jpg' for sorting."""
    filename = os.path.basename(filepath)
    match = re.search(r'fn(\d+)\.(jpg|png|jpeg)$', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0

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
    homography_debug=False,
    use_opencv_as_reference=True,
    test_mode=False
):
    def dprint(*args, **kwargs):
        if test_mode:
            print(*args, **kwargs)

    ensure_dir(working_dir)

    # --- Load images ---
    dev0_files = glob.glob(os.path.join(all_images_dir, "Dev0", "*.jpg")) + \
                 glob.glob(os.path.join(all_images_dir, "Dev0", "*.png")) + \
                 glob.glob(os.path.join(all_images_dir, "Dev0", "*.jpeg"))
    dev3_files = glob.glob(os.path.join(all_images_dir, "Dev3", "*.jpg")) + \
                 glob.glob(os.path.join(all_images_dir, "Dev3", "*.png")) + \
                 glob.glob(os.path.join(all_images_dir, "Dev3", "*.jpeg"))
    dev0_list = sorted(dev0_files, key=natural_sort_key)
    dev3_list = sorted(dev3_files, key=natural_sort_key)
    if num_frames is None:
        num_frames = min(len(dev0_list), len(dev3_list))
    sel0 = dev0_list[start_index:start_index + num_frames]
    sel3 = dev3_list[start_index:start_index + num_frames]

    if len(sel0) == 0 or len(sel3) == 0:
        logging.error("No images found!")
        return []

    # --- Load world coordinates & annotations ---
    world_points_meters, _ = load_world_coordinates(world_coords_strategy, world_coords_dir)
    world_points_pixels = world_points_meters * scale_px_per_m

    pts0, pts3, _ = load_annotation_or_annotate_final(sel0[0], sel3[0],
                                                      dev0_json=os.path.join(annotations, 'dev0_pts.json'),
                                                      dev3_json=os.path.join(annotations, 'dev3_pts.json'))

    # Trim points if needed
    n_points = min(len(pts0), len(world_points_meters))
    pts0 = pts0[:n_points]
    pts3 = pts3[:n_points]
    world_points_pixels = world_points_pixels[:n_points]

    # --- Calculate homographies ---
    H0 = calculate_normalized_homography(pts0, world_points_pixels)
    H3 = calculate_normalized_homography(pts3, world_points_pixels)

    if use_opencv_as_reference and H0 is not None and H3 is not None:
        H0_cv, _ = cv2.findHomography(pts0, world_points_pixels, cv2.RANSAC, 5.0)
        H3_cv, _ = cv2.findHomography(pts3, world_points_pixels, cv2.RANSAC, 5.0)
        if H0_cv is not None and H3_cv is not None:
            err0_cv = np.mean(np.linalg.norm(cv2.perspectiveTransform(pts0.reshape(-1,1,2), H0_cv).reshape(-1,2) - world_points_pixels, axis=1))
            err0_dlt = np.mean(np.linalg.norm(cv2.perspectiveTransform(pts0.reshape(-1,1,2), H0).reshape(-1,2) - world_points_pixels, axis=1))
            if err0_cv < err0_dlt: H0 = H0_cv
            if np.mean(np.linalg.norm(cv2.perspectiveTransform(pts3.reshape(-1,1,2), H3_cv).reshape(-1,2) - world_points_pixels, axis=1)) < \
               np.mean(np.linalg.norm(cv2.perspectiveTransform(pts3.reshape(-1,1,2), H3).reshape(-1,2) - world_points_pixels, axis=1)):
                H3 = H3_cv

    if H0 is None or H3 is None:
        logging.error("Failed to compute homographies!")
        return []

    H3_to_0 = np.linalg.inv(H0) @ H3
    H0_current = H0.copy()
    H3_current = H3.copy()
    H3_to_0_current = H3_to_0.copy()

    # --- BEV size ---
    bev_w = int(bev_out_meters[0] * scale_px_per_m)
    bev_h = int(bev_out_meters[1] * scale_px_per_m)
    out_size = (bev_w, bev_h)

    # --- Roll correction (computed once) ---
    global_roll_correction = None
    roll_correction_computed = False

    prev_bev0_gray = None
    results = []
    writer = None
    if output_video:
        ensure_dir(os.path.dirname(output_video))
        writer = imageio.get_writer(output_video, fps=fps)

    for i in tqdm(range(num_frames), desc="Processing frames"):
        img0 = cv2.imread(sel0[i])
        img3 = cv2.imread(sel3[i])
        if img0 is None or img3 is None:
            logging.warning(f"Skipping frame {i}")
            continue

        # --- Auto roll correction (first frame only) ---
        if not roll_correction_computed:
            gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
            gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
            orb = cv2.ORB_create(nfeatures=2000)
            kp0, des0 = orb.detectAndCompute(gray0, None)
            kp3, des3 = orb.detectAndCompute(gray3, None)
            if des0 is not None and des3 is not None and len(kp0) > 20 and len(kp3) > 20:
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = matcher.match(des0, des3)
                matches = sorted(matches, key=lambda x: x.distance)[:100]
                if len(matches) > 15:
                    src = np.float32([kp0[m.queryIdx].pt for m in matches])
                    dst = np.float32([kp3[m.trainIdx].pt for m in matches])
                    M, _ = cv2.estimateAffinePartial2D(src, dst)
                    if M is not None:
                        angle = np.arctan2(M[1,0], M[0,0]) * 180 / np.pi
                        dprint(f"Roll correction applied: {angle:+.3f}°")
                        h, w = img3.shape[:2]
                        global_roll_correction = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            roll_correction_computed = True

        if global_roll_correction is not None:
            img3 = cv2.warpAffine(img3, global_roll_correction, (img3.shape[1], img3.shape[0]),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # --- Refine homographies ---
        if refine_H_every_N > 0 and i > 0 and i % refine_H_every_N == 0:
            img0_prev = cv2.imread(sel0[i-1])
            img3_prev = cv2.imread(sel3[i-1])
            if img0_prev is not None and img3_prev is not None:
                H0_current = refine_homography_with_features(H0_current, img0_prev, img0)
                H3_current = refine_homography_with_features(H3_current, img3_prev, img3)
                H3_to_0_current = np.linalg.inv(H0_current) @ H3_current

        # --- Process ---
        stitched = stitch_two_images_proper(img0, img3, H3_to_0_current)
        bev0 = warp_to_bev(img0, H0_current, out_size)
        bev3 = warp_to_bev(img3, H3_current, out_size)

        # --- Speed calculation ---
        translation_px = np.array([0.0, 0.0])
        speed_kmh = 0.0
        if i > 0 and prev_bev0_gray is not None:
            gray = cv2.cvtColor(bev0, cv2.COLOR_BGR2GRAY)
            flow = dense_flow_farneback(prev_bev0_gray, gray, params=flow_params or {})
            mag = np.linalg.norm(flow, axis=2)
            valid = mag > 0.5
            if np.sum(valid) > 10:
                t_px, inliers = ransac_translation_improved(flow, mask=valid)
                if inliers is not None and np.sum(inliers) > 5:
                    translation_px = t_px
                    speed_kmh = np.linalg.norm(t_px) / scale_px_per_m * fps * 3.6
            prev_bev0_gray = gray
        else:
            prev_bev0_gray = cv2.cvtColor(bev0, cv2.COLOR_BGR2GRAY)

        results.append({
            'stitched': stitched, 'bev0': bev0, 'bev3': bev3,
            'translation_px': translation_px, 'speed_kmh': float(speed_kmh),
            'frame_idx': start_index + i
        })

        vis_frame = compose_visual_frame(stitched, bev0, bev3, translation_px, speed_kmh)
        if writer:
            writer.append_data(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB))
        elif test_mode and num_frames <= 10:
            disp = cv2.resize(vis_frame, None, fx=0.5, fy=0.5)
            cv2.imshow("Speedometer", disp)
            if cv2.waitKey(1) == ord('q'): break

    if writer:
        writer.close()
        logging.info(f"Video saved: {output_video}")
    cv2.destroyAllWindows()
    return results

def compose_visual_frame(stitched_img, bev0, bev3, translation_px, speed_kmh):
    """
    Compose visualization frame - FIXED VERSION.
    Makes sure stitched image is centered properly.
    """
    # Target display properties
    MAIN_VIEW_WIDTH = 1080
    SIDE_VIEW_SCALE = 0.5
    SIDE_VIEW_WIDTH = int(MAIN_VIEW_WIDTH * SIDE_VIEW_SCALE)
    SCOREBOARD_HEIGHT = 80
    
    # Get dimensions
    h_s, w_s = stitched_img.shape[:2]
    h0, w0 = bev0.shape[:2]
    h3, w3 = bev3.shape[:2]
    
    # Resize BEV images to fit SIDE_VIEW_WIDTH while maintaining aspect ratio
    bev0_scale = SIDE_VIEW_WIDTH / w0
    bev0_height = int(h0 * bev0_scale)
    bev0_r = cv2.resize(bev0, (SIDE_VIEW_WIDTH, bev0_height))
    
    bev3_scale = SIDE_VIEW_WIDTH / w3
    bev3_height = int(h3 * bev3_scale)
    bev3_r = cv2.resize(bev3, (SIDE_VIEW_WIDTH, bev3_height))
    
    # Resize stitched image to fit MAIN_VIEW_WIDTH while maintaining aspect ratio
    stitched_scale = MAIN_VIEW_WIDTH / w_s
    stitched_height = int(h_s * stitched_scale)
    stitched_resized = cv2.resize(stitched_img, (MAIN_VIEW_WIDTH, stitched_height))
    
    # Determine total height (max of all three heights)
    max_height = max(stitched_height, bev0_height, bev3_height)
    
    # Create canvas
    H_total = max_height + SCOREBOARD_HEIGHT
    W_total = SIDE_VIEW_WIDTH + MAIN_VIEW_WIDTH + SIDE_VIEW_WIDTH
    combined = np.zeros((H_total, W_total, 3), dtype=np.uint8)
    
    # Calculate vertical offsets to center each image
    bev0_y_offset = (max_height - bev0_height) // 2
    stitched_y_offset = (max_height - stitched_height) // 2
    bev3_y_offset = (max_height - bev3_height) // 2
    
    # Place images
    combined[bev0_y_offset:bev0_y_offset+bev0_height, 0:SIDE_VIEW_WIDTH] = bev0_r
    combined[stitched_y_offset:stitched_y_offset+stitched_height, SIDE_VIEW_WIDTH:SIDE_VIEW_WIDTH+MAIN_VIEW_WIDTH] = stitched_resized
    combined[bev3_y_offset:bev3_y_offset+bev3_height, SIDE_VIEW_WIDTH+MAIN_VIEW_WIDTH:W_total] = bev3_r
    
    # Add scoreboard
    sb_y_start = max_height
    sb_y_end = H_total
    cv2.rectangle(combined, (0, sb_y_start), (W_total, sb_y_end), (20, 20, 20), -1)
    
    # Add speed text
    speed_text = f"{speed_kmh:.1f} km/h"
    sb_center_y = sb_y_start + SCOREBOARD_HEIGHT // 2
    text_size = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)[0]
    text_x = SIDE_VIEW_WIDTH + (MAIN_VIEW_WIDTH - text_size[0]) // 2
    text_y = sb_center_y + text_size[1] // 2
    cv2.putText(combined, speed_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4)
    
    # Add motion vector (on left side, below BEV0)
    arrow_center_x = SIDE_VIEW_WIDTH // 2
    arrow_center_y = max_height + SCOREBOARD_HEIGHT // 2
    arrow_scale = 10.0
    arrow_vis = (int(translation_px[0] * arrow_scale), int(translation_px[1] * arrow_scale))
    
    cv2.putText(combined, "MOTION VECTOR", (arrow_center_x - 70, max_height + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.arrowedLine(combined, (arrow_center_x, arrow_center_y + 10),
                    (arrow_center_x + arrow_vis[0], arrow_center_y + arrow_vis[1] + 10),
                    (0, 0, 255), 3, tipLength=0.3)
    
    # Add labels at the top
    cv2.putText(combined, "BEV LEFT (Dev0)", (5, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(combined, "STITCHED VIEW", (SIDE_VIEW_WIDTH + 10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(combined, "BEV RIGHT (Dev3)", (SIDE_VIEW_WIDTH + MAIN_VIEW_WIDTH + 5, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return combined

# For backward compatibility
def run_pipeline_v4(*args, **kwargs):
    """Alias for the pipeline with world coordinates."""
    return run_pipeline_with_world_coords(*args, **kwargs)

# Simple test function
def test_world_coordinates_loading():
    """Test world coordinates loading."""
    print("Testing world coordinates loading...")
    
    try:
        # Test column strategy
        world_points_column, metadata_column = load_world_coordinates("column")
        print(f"\nColumn strategy loaded successfully:")
        print(f"  Points shape: {world_points_column.shape}")
        print(f"  Strategy: {metadata_column.get('strategy', 'unknown')}")
        print(f"  Points:")
        for i, pt in enumerate(world_points_column):
            print(f"    {i+1}: {pt}")
        
        # Test row strategy
        world_points_row, metadata_row = load_world_coordinates("row")
        print(f"\nRow strategy loaded successfully:")
        print(f"  Points shape: {world_points_row.shape}")
        print(f"  Strategy: {metadata_row.get('strategy', 'unknown')}")
        print(f"  Points:")
        for i, pt in enumerate(world_points_row):
            print(f"    {i+1}: {pt}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    # Test the world coordinates loading
    test_world_coordinates_loading()
