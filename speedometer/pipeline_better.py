# pipeline.py
import os
import glob
import cv2
import numpy as np
import imageio
from tqdm import tqdm
import logging
import re
import numpy.linalg

from .annotate import load_annotation_or_annotate
from .bev_and_flow import dense_flow_farneback
from .ransac_speed import ransac_translation_improved

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- START: Improved Normalized DLT Homography Functions ---

def calculate_normalization_matrix(points):
    """Calculate normalization matrix for points."""
    points = np.asarray(points, dtype=np.float32)
    mean = np.mean(points, axis=0)
    std = np.std(points, axis=0)
    std_avg = np.mean(std)
    
    # Avoid division by zero
    if std_avg < 1e-10:
        std_avg = 1.0
    
    # The scale factor for normalization
    scale = np.sqrt(2) / std_avg
    # The offset for translation
    offset = -scale * mean
    
    # Transformation matrix T (Scale and Translate)
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
        # Divide by the last coordinate to convert back to inhomogeneous (2D)
        normalized_points.append(normalized_point[:2] / normalized_point[2])
    return np.array(normalized_points)

def calculate_homography_dlt(point_pairs):
    """Calculate homography using normalized DLT with proper SVD."""
    A = []
    for point1, point2 in point_pairs:
        x, y = point1[0], point1[1]
        xp, yp = point2[0], point2[1]
        # Standard DLT equations (x' = Hx)
        A.append([x, y, 1, 0, 0, 0, -x*xp, -y*xp, -xp])
        A.append([0, 0, 0, x, y, 1, -x*yp, -y*yp, -yp])
    A = np.array(A)
    
    # Use SVD to find the solution
    U, S, Vt = np.linalg.svd(A)
    # Solution is the last row of Vt (corresponding to smallest singular value)
    H = Vt[-1].reshape(3, 3)
    
    return H / H[2, 2]

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

def calculate_normalized_homography(pts_img, pts_world_m, scale_px_per_m, debug=False):
    """Calculate homography using normalized DLT."""
    if len(pts_img) < 4:
        print(f"ERROR: Need at least 4 points, got {len(pts_img)}")
        return None
    
    # 1. Scale the World Coordinates (m) to Pixels
    pts_world_px = pts_world_m * scale_px_per_m
    
    if debug:
        print(f"\nImage points (pixels):")
        for i, pt in enumerate(pts_img):
            print(f"  Point {i+1}: {pt}")
        
        print(f"\nWorld points (meters):")
        for i, pt in enumerate(pts_world_m):
            print(f"  Point {i+1}: {pt}")
        
        print(f"\nWorld points (pixels @ {scale_px_per_m} px/m):")
        for i, pt in enumerate(pts_world_px):
            print(f"  Point {i+1}: {pt}")
    
    # 2. Normalize image and world points separately
    try:
        T_img = calculate_normalization_matrix(pts_img)
        T_world = calculate_normalization_matrix(pts_world_px)
        
        normalized_pts_img = normalize_points(pts_img, T_img)
        normalized_pts_world = normalize_points(pts_world_px, T_world)
        
        # 3. Calculate the homography (H_norm) using normalized points
        normalized_point_pairs = list(zip(normalized_pts_img, normalized_pts_world))
        H_normalized = calculate_homography_dlt(normalized_point_pairs)
        
        # 4. Denormalize: H = T_world_inv @ H_normalized @ T_img
        # This matrix H maps img_pixels -> world_pixels
        H_denormalized = np.linalg.inv(T_world) @ H_normalized @ T_img
        
        # Normalize by bottom-right element
        H_denormalized = H_denormalized / H_denormalized[2, 2]
        
        # Also calculate using OpenCV for comparison
        H_opencv, mask = cv2.findHomography(pts_img, pts_world_px, cv2.RANSAC, 5.0)
        
        if debug and H_opencv is not None:
            # Compare with OpenCV
            diff = np.linalg.norm(H_denormalized - H_opencv)
            print(f"\nMatrix difference (DLT vs OpenCV): {diff:.6f}")
            
            # Validate both
            print("\n" + "="*50)
            validate_homography(H_denormalized, pts_img, pts_world_px, "DLT Homography")
            validate_homography(H_opencv, pts_img, pts_world_px, "OpenCV Homography")
        
        return H_denormalized
        
    except np.linalg.LinAlgError as e:
        print(f"ERROR in homography calculation: {e}")
        # Fall back to OpenCV
        print("Falling back to OpenCV findHomography...")
        H_opencv, mask = cv2.findHomography(pts_img, pts_world_px, cv2.RANSAC, 5.0)
        return H_opencv
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# --- END: Improved Normalized DLT Homography Functions ---

def warp_to_bev(img, H, out_size):
    """Warps an image to the BEV using homography H."""
    return cv2.warpPerspective(img, H, out_size)

def stitch_two_images(img0, img3, H3_to_0, blend=True):
    """Warps img3 onto img0 using homography H3_to_0."""
    h0, w0 = img0.shape[:2]
    h3, w3 = img3.shape[:2]
    
    if H3_to_0 is None:
        print("WARNING: H3_to_0 is None, returning empty stitch")
        return np.zeros((h0, w0, 3), dtype=np.uint8)
    
    try:
        # Get corners of img3 after transformation
        corners_src = np.float32([[0, 0], [w3, 0], [w3, h3], [0, h3]]).reshape(-1, 1, 2)
        corners_dst = cv2.perspectiveTransform(corners_src, H3_to_0)
        
        # Find bounding box
        min_x = min(corners_dst[:, 0, 0].min(), 0)
        min_y = min(corners_dst[:, 0, 1].min(), 0)
        max_x = max(corners_dst[:, 0, 0].max(), w0)
        max_y = max(corners_dst[:, 0, 1].max(), h0)
        
        # Calculate translation
        tx = -min_x
        ty = -min_y
        
        # Create translation matrix
        T = np.array([[1, 0, tx],
                      [0, 1, ty],
                      [0, 0, 1]])
        
        # Warp both images
        output_width = int(max_x - min_x)
        output_height = int(max_y - min_y)
        
        warped_img3 = cv2.warpPerspective(img3, T @ H3_to_0, (output_width, output_height))
        warped_img0 = cv2.warpPerspective(img0, T, (output_width, output_height))
        
        if blend:
            # Create masks for blending
            mask3 = (warped_img3 > 0).any(axis=2).astype(np.uint8) * 255
            mask0 = (warped_img0 > 0).any(axis=2).astype(np.uint8) * 255
            
            # Find overlap region
            overlap = cv2.bitwise_and(mask3, mask0)
            
            # Create blended result
            result = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            
            # Non-overlapping regions
            non_overlap3 = cv2.bitwise_and(mask3, cv2.bitwise_not(overlap))
            non_overlap0 = cv2.bitwise_and(mask0, cv2.bitwise_not(overlap))
            
            result[non_overlap3 > 0] = warped_img3[non_overlap3 > 0]
            result[non_overlap0 > 0] = warped_img0[non_overlap0 > 0]
            
            # Overlapping region - blend
            overlap_indices = np.where(overlap > 0)
            if len(overlap_indices[0]) > 0:
                alpha = 0.5  # Blend factor
                result[overlap_indices] = (
                    alpha * warped_img0[overlap_indices] + 
                    (1 - alpha) * warped_img3[overlap_indices]
                ).astype(np.uint8)
            
            return result
        else:
            # Simple overlay
            mask = (warped_img3 == 0).all(axis=2)
            warped_img3[mask] = warped_img0[mask]
            return warped_img3
            
    except Exception as e:
        print(f"Error in stitching: {e}")
        # Fallback to simple stitch
        return np.hstack([img0, img3])

def ensure_dir(d):
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def refine_homography_with_features(H, img_prev, img_curr, method='orb'):
    """
    Refines the homography H (img_prev -> world/BEV) by tracking features.
    
    Args:
        H: Current homography matrix
        img_prev: Previous frame
        img_curr: Current frame
        method: Feature detection method ('orb', 'sift', or 'akaze')
    
    Returns:
        Refined homography matrix
    """
    if img_prev is None or img_curr is None:
        return H
    
    # Convert to grayscale if needed
    if len(img_prev.shape) == 3:
        gray_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(img_curr, cv2.COLOR_BGR2GRAY)
    else:
        gray_prev = img_prev
        gray_curr = img_curr
    
    # Choose feature detector
    if method == 'sift':
        detector = cv2.SIFT_create()
        matcher = cv2.BFMatcher(cv2.NORM_L2)
    elif method == 'akaze':
        detector = cv2.AKAZE_create()
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    else:  # ORB default
        detector = cv2.ORB_create(nfeatures=1000)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Detect and compute features
    kp1, des1 = detector.detectAndCompute(gray_prev, None)
    kp2, des2 = detector.detectAndCompute(gray_curr, None)
    
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return H
    
    # Match features
    if method in ['orb', 'akaze']:
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        num_matches = min(len(matches), 100)
        matches = matches[:num_matches]
    else:  # SIFT with ratio test
        matches = matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        matches = good_matches
        num_matches = len(matches)
    
    if num_matches < 4:
        return H
    
    # Extract matched points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Find homography between frames
    T, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if T is None:
        return H
    
    # Refine the homography: H_refined = H @ inv(T)
    H_refined = H @ np.linalg.inv(T)
    
    return H_refined

def natural_sort_key(filepath):
    """Extracts the integer frame number from 'fnX.jpg' for sorting."""
    filename = os.path.basename(filepath)
    match = re.search(r'fn(\d+)\.(jpg|png|jpeg)$', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0

def get_world_coordinates_from_annotation(annotation_order='top-left-first'):
    """
    Define world coordinates based on annotation order.
    
    Args:
        annotation_order: How points are annotated
            'top-left-first': Points annotated from top-left to bottom-right
            'clockwise': Points annotated clockwise from top-left
            'grid': Points in a grid pattern
    
    Returns:
        World coordinates in meters
    """
    # Standard road grid: 3m wide, 6m deep
    # X: left-right (positive right)
    # Y: forward-backward (positive forward)
    
    if annotation_order == 'top-left-first':
        # Points are annotated: top-left, top-right, middle-left, middle-right, bottom-left, bottom-right
        world_pts_m = np.array([
            [0.0, 6.0],    # 1. Top-Left: Left, Far
            [3.0, 6.0],    # 2. Top-Right: Right, Far
            [0.0, 3.0],    # 3. Middle-Left: Left, Middle
            [3.0, 3.0],    # 4. Middle-Right: Right, Middle
            [0.0, 0.0],    # 5. Bottom-Left: Left, Near
            [3.0, 0.0]     # 6. Bottom-Right: Right, Near
        ], dtype=np.float32)
        
    elif annotation_order == 'clockwise':
        # Points annotated clockwise starting from top-left
        world_pts_m = np.array([
            [0.0, 6.0],    # 1. Top-Left
            [3.0, 6.0],    # 2. Top-Right
            [3.0, 3.0],    # 3. Middle-Right
            [3.0, 0.0],    # 4. Bottom-Right
            [0.0, 0.0],    # 5. Bottom-Left
            [0.0, 3.0]     # 6. Middle-Left
        ], dtype=np.float32)
        
    else:  # 'grid' default
        # 3x2 grid: left column then right column
        world_pts_m = np.array([
            [0.0, 6.0],    # Top-Left
            [0.0, 3.0],    # Middle-Left
            [0.0, 0.0],    # Bottom-Left
            [3.0, 6.0],    # Top-Right
            [3.0, 3.0],    # Middle-Right
            [3.0, 0.0]     # Bottom-Right
        ], dtype=np.float32)
    
    # Apply offsets to center the road
    # Center in X (shift 1.5m right so road is centered at X=1.5)
    # Start at Y=1.0m (so closest point is 1m from camera)
    world_pts_m[:, 0] += 1.5  # Center horizontally
    world_pts_m[:, 1] += 1.0  # Add offset from camera
    
    return world_pts_m

def run_pipeline_v3(
    all_images_dir=".",
    annotations="./annotations",
    working_dir='./speed_work',
    start_index=0,
    num_frames=100,
    scale_px_per_m=120.0,
    fps=25.0,
    bev_out_meters=(15.0, 15.0),
    output_video='./results/result_speedometer.mp4',
    flow_params=None,
    refine_H_every_N=0,
    annotation_order='top-left-first',
    homography_debug=True,
    use_opencv_fallback=True
):
    """
    Improved pipeline with better homography calculation.
    
    Args:
        all_images_dir: Directory containing Dev0 and Dev3 folders
        annotations: Directory for annotation files
        working_dir: Working directory for intermediate files
        start_index: Starting frame index
        num_frames: Number of frames to process
        scale_px_per_m: Pixels per meter for BEV
        fps: Frames per second for video output
        bev_out_meters: BEV output size in meters (width, height)
        output_video: Output video path
        flow_params: Parameters for optical flow
        refine_H_every_N: Refine homography every N frames (0 = never)
        annotation_order: Order of annotated points
        homography_debug: Enable debug output for homography
        use_opencv_fallback: Fall back to OpenCV if DLT fails
    """
    ensure_dir(working_dir)
    
    # Find image files
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
        logging.error("No images found in the specified directories!")
        return []

    logging.info(f"Using {len(sel0)} Dev0 frames and {len(sel3)} Dev3 frames")
    logging.info(f"First Dev0: {os.path.basename(sel0[0])}")
    logging.info(f"First Dev3: {os.path.basename(sel3[0])}")

    # --- Homography Calculation ---
    dev0_json = os.path.join(annotations, 'dev0_pts.json')
    dev3_json = os.path.join(annotations, 'dev3_pts.json')
    
    # Load or create annotations
    pts0, pts3 = load_annotation_or_annotate(sel0[0], sel3[0], 
                                            dev0_json=dev0_json, 
                                            dev3_json=dev3_json)
    
    # Check if we have enough points
    if len(pts0) < 4 or len(pts3) < 4:
        logging.error(f"Need at least 4 points, got {len(pts0)} for Dev0 and {len(pts3)} for Dev3")
        return []
    
    # Get world coordinates based on annotation order
    world_pts_m = get_world_coordinates_from_annotation(annotation_order)
    
    if homography_debug:
        print("\n" + "="*60)
        print("HOMOGRAPHY CALCULATION DEBUG")
        print("="*60)
        
        print(f"\nAnnotation order: {annotation_order}")
        print(f"\nWorld points (meters):")
        for i, pt in enumerate(world_pts_m):
            print(f"  Point {i+1}: {pt}")
    
    # Calculate homographies with improved function
    H0 = calculate_normalized_homography(pts0, world_pts_m, scale_px_per_m, debug=homography_debug)
    H3 = calculate_normalized_homography(pts3, world_pts_m, scale_px_per_m, debug=homography_debug)
    
    if H0 is None or H3 is None:
        logging.error("Failed to calculate homographies!")
        return []
    
    # Calculate stitching homography: H3_to_0 = inv(H0) @ H3
    # This maps Dev3 -> World -> Dev0
    H3_to_0 = np.linalg.inv(H0) @ H3
    
    # Validate stitching homography
    if homography_debug:
        print("\n" + "="*50)
        print("STITCHING HOMOGRAPHY VALIDATION")
        print("="*50)
        
        # Test projection of Dev3 corners to Dev0 space
        h3, w3 = cv2.imread(sel3[0]).shape[:2] if os.path.exists(sel3[0]) else (1200, 1920)
        corners_src = np.float32([[0, 0], [w3, 0], [w3, h3], [0, h3]])
        corners_src_homo = np.column_stack([corners_src, np.ones(4)])
        corners_dst = (H3_to_0 @ corners_src_homo.T).T
        corners_dst = corners_dst[:, :2] / corners_dst[:, 2:]
        
        print(f"\nDev3 corners projected to Dev0 space:")
        for i, (src, dst) in enumerate(zip(corners_src, corners_dst)):
            print(f"  Corner {i+1}: {src.astype(int)} -> {dst.astype(int)}")
    
    # BEV output size
    bev_w = int(bev_out_meters[0] * scale_px_per_m)
    bev_h = int(bev_out_meters[1] * scale_px_per_m)
    out_size = (bev_w, bev_h)
    
    if homography_debug:
        print(f"\nBEV output size: {bev_w}x{bev_h} pixels")
        print(f"BEV world size: {bev_out_meters[0]}x{bev_out_meters[1]} meters")
    
    # Initialize tracking variables
    prev_bev0_gray = None
    results = []
    FLOW_DOWNSCALE_FACTOR = 0.5
    
    # Homography refinement tracking
    H0_current = H0.copy()
    H3_current = H3.copy()
    H3_to_0_current = H3_to_0.copy()
    
    # Video writer setup
    is_interactive_test = (output_video is None) and (num_frames <= 10)
    writer = None
    
    if output_video:
        ensure_dir(os.path.dirname(output_video))
        try:
            writer = imageio.get_writer(output_video, fps=fps)
        except Exception as e:
            logging.error(f"Failed to create video writer: {e}")
            writer = None
    
    # --- Main Processing Loop ---
    for i in tqdm(range(num_frames)):
        img0 = cv2.imread(sel0[i])
        img3 = cv2.imread(sel3[i])
        
        if img0 is None or img3 is None:
            logging.warning(f"Skipping frame {i}: Failed to load one or both images.")
            continue
        
        # Homography refinement if enabled
        if refine_H_every_N > 0 and i > 0 and i % refine_H_every_N == 0:
            # Load previous frame for refinement
            img0_prev = cv2.imread(sel0[i-1])
            img3_prev = cv2.imread(sel3[i-1])
            
            if img0_prev is not None and img3_prev is not None:
                H0_current = refine_homography_with_features(H0_current, img0_prev, img0, method='orb')
                H3_current = refine_homography_with_features(H3_current, img3_prev, img3, method='orb')
                H3_to_0_current = np.linalg.inv(H0_current) @ H3_current
        
        # --- Process Frame ---
        stitched = stitch_two_images(img0, img3, H3_to_0_current, blend=True)
        bev0 = warp_to_bev(img0, H0_current, out_size)
        bev3 = warp_to_bev(img3, H3_current, out_size)
        
        # Optical flow and speed calculation
        translation_px = np.array([0.0, 0.0])
        speed_kmh = 0.0
        
        bev0_low_res = cv2.resize(bev0, None, 
                                 fx=FLOW_DOWNSCALE_FACTOR, 
                                 fy=FLOW_DOWNSCALE_FACTOR, 
                                 interpolation=cv2.INTER_LINEAR)
        
        if i > 0 and prev_bev0_gray is not None:
            gray = cv2.cvtColor(bev0_low_res, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            flow = dense_flow_farneback(prev_bev0_gray, gray, params=flow_params)
            
            # Scale flow back to original size
            flow *= (1.0 / FLOW_DOWNSCALE_FACTOR)
            
            # Calculate flow magnitude and create valid mask
            mag = np.linalg.norm(flow, axis=2)
            valid_mask = mag > 0.3  # Threshold to remove noise
            
            # Use RANSAC to find dominant translation
            t_px, inliers = ransac_translation_improved(flow, mask=valid_mask)
            
            if inliers is not None and np.sum(inliers) > 10:  # Need enough inliers
                translation_px = t_px
                speed_m_per_frame = np.linalg.norm(t_px) / scale_px_per_m
                speed_kmh = speed_m_per_frame * fps * 3.6
            else:
                speed_kmh = 0.0
            
            prev_bev0_gray = gray
        else:
            prev_bev0_gray = cv2.cvtColor(bev0_low_res, cv2.COLOR_BGR2GRAY)
        
        # Store results
        results.append({
            'stitched': stitched,
            'bev0': bev0,
            'bev3': bev3,
            'translation_px': translation_px,
            'speed_kmh': float(speed_kmh),
            'frame_idx': start_index + i
        })
        
        # Visualization
        vis_frame = compose_visual_frame_v2(stitched, bev0, bev3, translation_px, speed_kmh)
        
        if writer:
            # Write frame to video
            writer.append_data(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB))
        
        elif is_interactive_test:
            # Interactive display
            DISPLAY_SCALE = 0.5
            display_frame = cv2.resize(vis_frame, None,
                                     fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
                                     interpolation=cv2.INTER_LINEAR)
            
            window_title = f"Frame {start_index + i + 1}/{start_index + num_frames} | Speed: {speed_kmh:.1f} km/h"
            cv2.imshow(window_title, display_frame)
            
            key = cv2.waitKey(1 if i == 0 else 0)
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                # Pause
                cv2.waitKey(0)
    
    # Cleanup
    if writer:
        writer.close()
        logging.info(f'Video written to {output_video}')
    elif is_interactive_test:
        cv2.destroyAllWindows()
    
    # Save homography matrices for future use
    homography_file = os.path.join(working_dir, 'homography_matrices.npz')
    np.savez(homography_file, H0=H0, H3=H3, H3_to_0=H3_to_0, 
             pts0=pts0, pts3=pts3, world_pts_m=world_pts_m,
             scale_px_per_m=scale_px_per_m)
    logging.info(f'Homography matrices saved to {homography_file}')
    
    return results

# Keep the existing compose_visual_frame_v2 function
def compose_visual_frame_v2(stitched_img, bev0, bev3, translation_px, speed_kmh):
    """Compose visualization frame (unchanged)."""
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
    combined[:bev_h, 0:SIDE_VIEW_WIDTH] = bev0_r                      
    combined[:stitched_h, mid_start:mid_end] = stitched_resized       
    combined[:bev_h, right_start:W_total] = bev3_r                    

    # --- 4. Draw Scoreboard on the Main View ---
    sb_y_start = stitched_h
    sb_y_end = H_total
    
    # Draw dark boxes for all scoreboard areas
    cv2.rectangle(combined, (0, sb_y_start), (W_total, sb_y_end), (20, 20, 20), -1)

    sb_center_y = sb_y_start + SCOREBOARD_HEIGHT // 2
    
    # A. Draw Speed Text (Centered on the Main View scoreboard)
    speed_text = f"{speed_kmh:.1f} km/h"
    text_size, baseline = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)
    text_x = mid_start + (MAIN_VIEW_WIDTH - text_size[0]) // 2 
    text_y = sb_center_y + text_size[1] // 2
    
    cv2.putText(combined, speed_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4)

    # B. Draw Motion Vector Arrow (on the BEV0 panel)
    arrow_center_x = SIDE_VIEW_WIDTH // 2
    arrow_center_y = bev_h + SCOREBOARD_HEIGHT // 2
    arrow_scale = 10.0 
    arrow_vis = (int(translation_px[0]*arrow_scale), int(translation_px[1]*arrow_scale))
    
    cv2.putText(combined, "MOTION VECTOR", (arrow_center_x - 70, stitched_h + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
    cv2.arrowedLine(combined, (arrow_center_x, arrow_center_y + 10),
                    (arrow_center_x+arrow_vis[0], arrow_center_y+arrow_vis[1] + 10),
                    (0, 0, 255), 3, tipLength=0.3)
                    
    # C. Labeling the Views
    cv2.putText(combined, "BEV LEFT (Dev0)", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(combined, "STITCHED MAIN VIEW", (mid_start + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(combined, "BEV RIGHT (Dev3)", (right_start + 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return combined

# Add a helper function to load saved homographies
def load_saved_homographies(homography_file):
    """Load saved homography matrices."""
    if not os.path.exists(homography_file):
        return None
    
    data = np.load(homography_file)
    return {
        'H0': data['H0'],
        'H3': data['H3'],
        'H3_to_0': data['H3_to_0'],
        'pts0': data['pts0'],
        'pts3': data['pts3'],
        'world_pts_m': data['world_pts_m'],
        'scale_px_per_m': data['scale_px_per_m']
    }