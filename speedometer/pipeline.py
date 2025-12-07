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

from .annotate import load_annotation_or_annotate_improved
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
    
    # Normalize
    if abs(H[2, 2]) > 1e-10:
        H = H / H[2, 2]
    return H

def calculate_normalized_homography(pts_img, pts_world_px, debug=False):
    """
    Calculate homography using normalized DLT.
    
    Args:
        pts_img: Image points (pixels)
        pts_world_px: World points (pixels)
        debug: Enable debug output
    
    Returns:
        Homography matrix H
    """
    if len(pts_img) < 4:
        print(f"ERROR: Need at least 4 points, got {len(pts_img)}")
        return None
    
    if len(pts_img) != len(pts_world_px):
        print(f"ERROR: Number of points mismatch: {len(pts_img)} image points vs {len(pts_world_px)} world points")
        return None
    
    try:
        # Normalize image and world points separately
        T_img = calculate_normalization_matrix(pts_img)
        T_world = calculate_normalization_matrix(pts_world_px)
        
        normalized_pts_img = normalize_points(pts_img, T_img)
        normalized_pts_world = normalize_points(pts_world_px, T_world)
        
        # Calculate the homography using normalized points
        normalized_point_pairs = list(zip(normalized_pts_img, normalized_pts_world))
        H_normalized = calculate_homography_dlt(normalized_point_pairs)
        
        # Denormalize: H = T_world_inv @ H_normalized @ T_img
        H = np.linalg.inv(T_world) @ H_normalized @ T_img
        
        # Normalize by bottom-right element
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

def determine_world_coordinate_order(pts_img):
    """
    Determine the order of points based on their image positions.
    
    Returns:
        Order string: 'horizontal-first', 'vertical-first', or 'grid'
    """
    if len(pts_img) < 6:
        return 'grid'  # Default
    
    # Calculate statistics
    x_coords = pts_img[:, 0]
    y_coords = pts_img[:, 1]
    
    # Check if points form two vertical columns
    # Sort by x, then by y
    sorted_by_x = pts_img[np.argsort(x_coords)]
    
    # Check if we have two clear columns
    x_diff = np.diff(sorted_by_x[:, 0])
    median_x_diff = np.median(x_diff)
    
    # If there's a large gap in x, it's probably two columns
    if median_x_diff > 50:  # Threshold for column separation
        # Points are in left column then right column
        return 'grid'
    
    # Check if points form two horizontal rows
    sorted_by_y = pts_img[np.argsort(y_coords)]
    y_diff = np.diff(sorted_by_y[:, 1])
    median_y_diff = np.median(y_diff)
    
    if median_y_diff > 50:  # Threshold for row separation
        # Points are in top row then bottom row
        return 'horizontal-first'
    
    # Default: assume grid pattern
    return 'grid'

def get_world_coordinates_corrected(pts_img, scale_px_per_m, annotation_order='auto'):
    """
    Get world coordinates that match the actual point distribution in the image.
    
    Args:
        pts_img: Image points (pixels)
        scale_px_per_m: Pixels per meter
        annotation_order: 'auto', 'grid', 'horizontal-first', 'vertical-first'
    
    Returns:
        World coordinates in meters that match image point distribution
    """
    if annotation_order == 'auto':
        annotation_order = determine_world_coordinate_order(pts_img)
    
    print(f"\nDetected annotation order: {annotation_order}")
    
    if annotation_order == 'grid':
        # Left column then right column (3x2 grid)
        # Points 1-3: Left column (top to bottom)
        # Points 4-6: Right column (top to bottom)
        world_pts_m = np.array([
            [0.0, 6.0],    # 1. Top-Left: Left, Far
            [0.0, 3.0],    # 2. Middle-Left: Left, Middle
            [0.0, 0.0],    # 3. Bottom-Left: Left, Near
            [3.0, 6.0],    # 4. Top-Right: Right, Far
            [3.0, 3.0],    # 5. Middle-Right: Right, Middle
            [3.0, 0.0]     # 6. Bottom-Right: Right, Near
        ], dtype=np.float32)
        
    elif annotation_order == 'horizontal-first':
        # Top row then bottom row (2x3 grid)
        # Points 1-3: Top row (left to right)
        # Points 4-6: Bottom row (left to right)
        world_pts_m = np.array([
            [0.0, 6.0],    # 1. Top-Left
            [1.5, 6.0],    # 2. Top-Middle
            [3.0, 6.0],    # 3. Top-Right
            [0.0, 0.0],    # 4. Bottom-Left
            [1.5, 0.0],    # 5. Bottom-Middle
            [3.0, 0.0]     # 6. Bottom-Right
        ], dtype=np.float32)
        
    elif annotation_order == 'vertical-first':
        # Two columns, top to bottom in each column
        # This matches your current annotation
        world_pts_m = np.array([
            [0.0, 6.0],    # 1. Top-Left
            [3.0, 6.0],    # 2. Top-Right
            [0.0, 3.0],    # 3. Middle-Left
            [3.0, 3.0],    # 4. Middle-Right
            [0.0, 0.0],    # 5. Bottom-Left
            [3.0, 0.0]     # 6. Bottom-Right
        ], dtype=np.float32)
    
    else:
        # Default to grid
        world_pts_m = np.array([
            [0.0, 6.0], [0.0, 3.0], [0.0, 0.0],
            [3.0, 6.0], [3.0, 3.0], [3.0, 0.0]
        ], dtype=np.float32)
    
    # Apply offsets to center the road
    # Center in X (shift 1.5m right so road is centered at X=1.5)
    # Start at Y=1.0m (so closest point is 1m from camera)
    world_pts_m[:, 0] += 1.5  # Center horizontally
    world_pts_m[:, 1] += 1.0  # Add offset from camera
    
    # Convert to pixels for debugging
    world_pts_px = world_pts_m * scale_px_per_m
    
    return world_pts_m, world_pts_px, annotation_order

# --- END: Improved Normalized DLT Homography Functions ---

def warp_to_bev(img, H, out_size):
    """Warps an image to the BEV using homography H."""
    return cv2.warpPerspective(img, H, out_size)

def stitch_two_images_simple(img0, img3, H3_to_0):
    """Simple stitching that works with identity or near-identity homographies."""
    h0, w0 = img0.shape[:2]
    h3, w3 = img3.shape[:2]
    
    if H3_to_0 is None:
        print("WARNING: H3_to_0 is None, using simple side-by-side")
        return np.hstack([img0, img3])
    
    # Check if H3_to_0 is close to identity
    H_identity = np.eye(3)
    diff_from_identity = np.linalg.norm(H3_to_0 - H_identity)
    
    if diff_from_identity < 1.0:  # Very close to identity
        print(f"WARNING: H3_to_0 is close to identity (diff={diff_from_identity:.4f})")
        # Simple side-by-side stitching
        return np.hstack([img0, img3])
    
    # Regular stitching
    try:
        # Warp img3 to img0's coordinate system
        warped_img3 = cv2.warpPerspective(img3, H3_to_0, (w0, h0))
        
        # Create mask for warped image
        mask = (warped_img3 > 0).any(axis=2)
        
        # Blend images
        result = img0.copy()
        result[mask] = cv2.addWeighted(img0[mask], 0.5, warped_img3[mask], 0.5, 0)
        
        return result
        
    except Exception as e:
        print(f"Error in stitching: {e}")
        return np.hstack([img0, img3])

def ensure_dir(d):
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def refine_homography_with_features(H, img_prev, img_curr, method='orb'):
    """
    Refines the homography H (img_prev -> world/BEV) by tracking features.
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
    
    # Use ORB for speed
    detector = cv2.ORB_create(nfeatures=500)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Detect and compute features
    kp1, des1 = detector.detectAndCompute(gray_prev, None)
    kp2, des2 = detector.detectAndCompute(gray_curr, None)
    
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return H
    
    # Match features
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:50]  # Use top 50 matches
    
    if len(matches) < 4:
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

def auto_reorder_points_column_first(points):
    """Automatically reorder points to column-first order (2x3 grid)."""
    if len(points) != 6:
        return points  # Can't reorder if not 6 points
    
    # Cluster by X to find left/right columns
    x_coords = points[:, 0]
    x_sorted_indices = np.argsort(x_coords)
    
    left_indices = x_sorted_indices[:3]
    right_indices = x_sorted_indices[3:]
    
    left_points = points[left_indices]
    right_points = points[right_indices]
    
    # Sort each column by Y (top to bottom)
    left_points = left_points[np.argsort(left_points[:, 1])]
    right_points = right_points[np.argsort(right_points[:, 1])]
    
    # Combine in column-first order
    reordered = np.vstack([
        left_points[0],   # Left column, top
        left_points[1],   # Left column, middle
        left_points[2],   # Left column, bottom
        right_points[0],  # Right column, top
        right_points[1],  # Right column, middle
        right_points[2]   # Right column, bottom
    ])
    
    return reordered

def run_pipeline_corrected(
    all_images_dir=".",
    annotations="./annotations",
    working_dir='./speed_work',
    start_index=0,
    num_frames=100,
    scale_px_per_m=150.0,  # Adjusted to match your output
    fps=25.0,
    bev_out_meters=(15.0, 15.0),
    output_video='./results/result_speedometer.mp4',
    flow_params=None,
    refine_H_every_N=0,
    annotation_order='auto',  # Auto-detect order
    homography_debug=True,
    use_opencv_as_reference=True  # Use OpenCV as reference to validate
):
    """
    Corrected pipeline with proper coordinate system handling.
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
    pts0, pts3 = load_annotation_or_annotate_improved(sel0[0], sel3[0], 
                                            dev0_json=dev0_json, 
                                            dev3_json=dev3_json)
    
    # # Reorder points to column-first order
    # pts0 = auto_reorder_points_column_first(pts0)
    # pts3 = auto_reorder_points_column_first(pts3)
    
    # Check if we have enough points
    if len(pts0) < 4 or len(pts3) < 4:
        logging.error(f"Need at least 4 points, got {len(pts0)} for Dev0 and {len(pts3)} for Dev3")
        return []
    
    # Get world coordinates that match the actual point distribution
    world_pts_m0, world_pts_px0, order0 = get_world_coordinates_corrected(
        pts0, scale_px_per_m, annotation_order)
    
    world_pts_m3, world_pts_px3, order3 = get_world_coordinates_corrected(
        pts3, scale_px_per_m, annotation_order)
    
    if homography_debug:
        print("\n" + "="*60)
        print("HOMOGRAPHY CALCULATION - CORRECTED")
        print("="*60)
        
        print(f"\nDetected annotation orders:")
        print(f"  Dev0: {order0}")
        print(f"  Dev3: {order3}")
        
        print(f"\nWorld points for Dev0 (meters):")
        for i, (img_pt, world_pt) in enumerate(zip(pts0, world_pts_m0)):
            print(f"  Point {i+1}: Image {img_pt.astype(int)} -> World {world_pt}")
        
        print(f"\nWorld points for Dev0 (pixels @ {scale_px_per_m} px/m):")
        for i, world_pt_px in enumerate(world_pts_px0):
            print(f"  Point {i+1}: {world_pt_px}")
    
    # Calculate homographies using corrected function
    H0 = calculate_normalized_homography(pts0, world_pts_px0, debug=homography_debug)
    H3 = calculate_normalized_homography(pts3, world_pts_px3, debug=homography_debug)
    
    # Also calculate using OpenCV for comparison
    if use_opencv_as_reference:
        H0_opencv, mask0 = cv2.findHomography(pts0, world_pts_px0, cv2.RANSAC, 5.0)
        H3_opencv, mask3 = cv2.findHomography(pts3, world_pts_px3, cv2.RANSAC, 5.0)
        
        if homography_debug and H0_opencv is not None and H3_opencv is not None:
            print("\n" + "="*50)
            print("OPENCV REFERENCE VALIDATION")
            print("="*50)
            
            validate_homography(H0, pts0, world_pts_px0, "DLT H0")
            validate_homography(H0_opencv, pts0, world_pts_px0, "OpenCV H0")
            
            print("\n" + "-"*50)
            validate_homography(H3, pts3, world_pts_px3, "DLT H3")
            validate_homography(H3_opencv, pts3, world_pts_px3, "OpenCV H3")
            
            # Compare matrices
            if H0 is not None:
                diff0 = np.linalg.norm(H0 - H0_opencv)
                print(f"\nMatrix difference H0 (DLT vs OpenCV): {diff0:.6f}")
            
            if H3 is not None:
                diff3 = np.linalg.norm(H3 - H3_opencv)
                print(f"Matrix difference H3 (DLT vs OpenCV): {diff3:.6f}")
            
            # Choose the better homography
            if H0 is not None and H0_opencv is not None:
                valid0_dlt, err0_dlt = validate_homography(H0, pts0, world_pts_px0, "DLT H0", threshold=100)
                valid0_cv, err0_cv = validate_homography(H0_opencv, pts0, world_pts_px0, "OpenCV H0", threshold=100)
                
                if err0_cv < err0_dlt:
                    print(f"\nUsing OpenCV H0 (error: {err0_cv:.2f} vs DLT: {err0_dlt:.2f})")
                    H0 = H0_opencv
                else:
                    print(f"\nUsing DLT H0 (error: {err0_dlt:.2f} vs OpenCV: {err0_cv:.2f})")
            
            if H3 is not None and H3_opencv is not None:
                valid3_dlt, err3_dlt = validate_homography(H3, pts3, world_pts_px3, "DLT H3", threshold=100)
                valid3_cv, err3_cv = validate_homography(H3_opencv, pts3, world_pts_px3, "OpenCV H3", threshold=100)
                
                if err3_cv < err3_dlt:
                    print(f"Using OpenCV H3 (error: {err3_cv:.2f} vs DLT: {err3_dlt:.2f})")
                    H3 = H3_opencv
                else:
                    print(f"Using DLT H3 (error: {err3_dlt:.2f} vs OpenCV: {err3_cv:.2f})")
    
    if H0 is None or H3 is None:
        logging.error("Failed to calculate homographies!")
        return []
    
    # Calculate stitching homography: H3_to_0 = inv(H0) @ H3
    H3_to_0 = np.linalg.inv(H0) @ H3
    
    # Validate stitching homography
    if homography_debug:
        print("\n" + "="*50)
        print("STITCHING HOMOGRAPHY")
        print("="*50)
        
        print(f"\nH3_to_0 matrix:")
        print(H3_to_0)
        
        # Test projection
        h3, w3 = cv2.imread(sel3[0]).shape[:2] if os.path.exists(sel3[0]) else (1200, 1920)
        test_points = np.float32([
            [w3/4, h3/4],      # Top-left quadrant
            [3*w3/4, h3/4],    # Top-right quadrant
            [3*w3/4, 3*h3/4],  # Bottom-right quadrant
            [w3/4, 3*h3/4]     # Bottom-left quadrant
        ])
        
        test_points_homo = np.column_stack([test_points, np.ones(4)])
        projected_points = (H3_to_0 @ test_points_homo.T).T
        projected_points = projected_points[:, :2] / projected_points[:, 2:]
        
        print(f"\nTest projection of Dev3 points to Dev0 space:")
        for i, (src, dst) in enumerate(zip(test_points, projected_points)):
            print(f"  Point {i+1}: {src.astype(int)} -> {dst.astype(int)}")
    
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
                H0_current = refine_homography_with_features(H0_current, img0_prev, img0)
                H3_current = refine_homography_with_features(H3_current, img3_prev, img3)
                H3_to_0_current = np.linalg.inv(H0_current) @ H3_current
        
        # --- Process Frame ---
        stitched = stitch_two_images_simple(img0, img3, H3_to_0_current)
        bev0 = warp_to_bev(img0, H0_current, out_size)
        bev3 = warp_to_bev(img3, H3_current, out_size)
        
        # Optical flow and speed calculation
        translation_px = np.array([0.0, 0.0])
        speed_kmh = 0.0
        
        if i > 0 and prev_bev0_gray is not None:
            # Convert to grayscale for flow calculation
            gray = cv2.cvtColor(bev0, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            if flow_params is None:
                flow_params = {
                    'pyr_scale': 0.5,
                    'levels': 3,
                    'winsize': 15,
                    'iterations': 3,
                    'poly_n': 5,
                    'poly_sigma': 1.2,
                    'flags': 0
                }
            
            flow = dense_flow_farneback(prev_bev0_gray, gray, params=flow_params)
            
            # Calculate flow magnitude and create valid mask
            mag = np.linalg.norm(flow, axis=2)
            valid_mask = mag > 0.5  # Threshold to remove noise
            
            # Use RANSAC to find dominant translation
            if np.sum(valid_mask) > 10:
                t_px, inliers = ransac_translation_improved(flow, mask=valid_mask)
                
                if inliers is not None and np.sum(inliers) > 5:
                    translation_px = t_px
                    speed_m_per_frame = np.linalg.norm(t_px) / scale_px_per_m
                    speed_kmh = speed_m_per_frame * fps * 3.6
            
            prev_bev0_gray = gray
        else:
            # Initialize with first frame
            prev_bev0_gray = cv2.cvtColor(bev0, cv2.COLOR_BGR2GRAY)
        
        # Store results
        results.append({
            'stitched': stitched,
            'bev0': bev0,
            'bev3': bev3,
            'translation_px': translation_px,
            'speed_kmh': float(speed_kmh),
            'frame_idx': start_index + i,
            'H0': H0_current.copy(),
            'H3': H3_current.copy()
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
            
            key = cv2.waitKey(0 if i == 0 else 1)
            
            if key == ord('q'):
                break
    
    # Cleanup
    if writer:
        writer.close()
        logging.info(f'Video written to {output_video}')
    elif is_interactive_test:
        cv2.destroyAllWindows()
    
    # Save homography matrices for future use
    homography_file = os.path.join(working_dir, 'homography_matrices_corrected.npz')
    np.savez(homography_file, 
             H0=H0, H3=H3, H3_to_0=H3_to_0,
             pts0=pts0, pts3=pts3,
             world_pts_m0=world_pts_m0, world_pts_m3=world_pts_m3,
             scale_px_per_m=scale_px_per_m,
             annotation_order0=order0, annotation_order3=order3)
    logging.info(f'Homography matrices saved to {homography_file}')
    
    return results

# Keep the existing compose_visual_frame_v2 function
def compose_visual_frame_v2(stitched_img, bev0, bev3, translation_px, speed_kmh):
    """Compose visualization frame (unchanged)."""
    # [Keep the existing function exactly as it was]
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

# For backward compatibility
def run_pipeline_v3(*args, **kwargs):
    """Alias for the corrected pipeline."""
    return run_pipeline_corrected(*args, **kwargs)