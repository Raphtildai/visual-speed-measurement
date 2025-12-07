#!/usr/bin/env python3
"""
test_homography_improved.py - Improved test with resizing for large images
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory to path to import your pipeline module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Your DLT implementation (copied here for testing)
def calculate_normalization_matrix(points):
    points = np.asarray(points, dtype=np.float32)
    mean = np.mean(points, axis=0)
    std = np.std(points, axis=0)
    std_avg = np.mean(std)
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
    points = np.asarray(points, dtype=np.float32)
    normalized_points = []
    for point in points:
        point_homogeneous = np.append(point, 1)
        normalized_point = T @ point_homogeneous
        # Divide by the last coordinate to convert back to inhomogeneous (2D)
        normalized_points.append(normalized_point[:2] / normalized_point[2])
    return np.array(normalized_points)

def calculate_homography_dlt_svd(point_pairs):
    """Calculate homography using normalized DLT with SVD (CORRECT VERSION)."""
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

def calculate_homography_dlt_eig(point_pairs):
    """Calculate homography using eigenvalue decomposition (YOUR CURRENT VERSION)."""
    A = []
    for point1, point2 in point_pairs:
        x, y = point1[0], point1[1]
        xp, yp = point2[0], point2[1]
        # Your current equations
        A.append([-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x*yp, y*yp, yp])
    A = np.array(A)
    
    # Your current eigenvalue approach
    eig_val, eig_vec = np.linalg.eig(A.T @ A)
    # The solution vector is the eigenvector associated with the smallest eigenvalue
    Vt = eig_vec[:, np.argmin(eig_val)]
    H = Vt.reshape(3, 3)
    
    return H / H[2, 2]

def calculate_normalized_homography(pts_img, pts_world_px):
    """Calculate homography using normalized DLT."""
    # Normalize image and world points separately
    T_img = calculate_normalization_matrix(pts_img)
    T_world = calculate_normalization_matrix(pts_world_px)
    
    normalized_pts_img = normalize_points(pts_img, T_img)
    normalized_pts_world = normalize_points(pts_world_px, T_world)
    
    # Calculate the homography (H_norm) using normalized points
    normalized_point_pairs = list(zip(normalized_pts_img, normalized_pts_world))
    
    # Test both versions
    H_normalized_svd = calculate_homography_dlt_svd(normalized_point_pairs)
    H_normalized_eig = calculate_homography_dlt_eig(normalized_point_pairs)
    
    # Denormalize both versions
    H_svd = np.linalg.inv(T_world) @ H_normalized_svd @ T_img
    H_eig = np.linalg.inv(T_world) @ H_normalized_eig @ T_img
    
    return H_svd / H_svd[2, 2], H_eig / H_eig[2, 2]

def validate_homography(H, src_pts, dst_pts, name="Homography"):
    """Validate homography by projecting points and checking error."""
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
    
    return avg_error, max_error

class ImagePointSelector:
    """Improved point selector with zoom, pan, and scaling."""
    
    def __init__(self, image_path, display_scale=0.5, window_name="Select Points"):
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.display_scale = display_scale
        self.window_name = window_name
        self.points = []
        
        # Create display image
        self.display_image = cv2.resize(self.original_image, None, 
                                       fx=display_scale, fy=display_scale,
                                       interpolation=cv2.INTER_LINEAR)
        
        # Store scale factor for converting points back to original coordinates
        self.scale_factor = 1.0 / display_scale
        
        # Zoom and pan variables
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.dragging = False
        self.last_mouse_pos = None
        
    def click_event(self, event, x, y, flags, param):
        """Handle mouse events for point selection and panning."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click to add point
            original_x = int(x * self.scale_factor / self.zoom_level + self.pan_x)
            original_y = int(y * self.scale_factor / self.zoom_level + self.pan_y)
            
            self.points.append([original_x, original_y])
            
            # Draw point on the zoomed/panned view
            display_x = int((original_x - self.pan_x) * self.zoom_level / self.scale_factor)
            display_y = int((original_y - self.pan_y) * self.zoom_level / self.scale_factor)
            
            cv2.circle(self.current_display, (display_x, display_y), 8, (0, 255, 0), -1)
            cv2.putText(self.current_display, f"{len(self.points)}", 
                       (display_x + 12, display_y - 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(self.window_name, self.current_display)
            
            print(f"Point {len(self.points)}: ({original_x}, {original_y})")
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click to remove last point
            if self.points:
                self.points.pop()
                self.redraw_display()
                print(f"Removed point, {len(self.points)} points remaining")
                
        elif event == cv2.EVENT_MBUTTONDOWN:
            # Middle click to start panning
            self.dragging = True
            self.last_mouse_pos = (x, y)
            
        elif event == cv2.EVENT_MBUTTONUP:
            # Middle button release to stop panning
            self.dragging = False
            self.last_mouse_pos = None
            
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            # Pan the view
            if self.last_mouse_pos:
                dx = (x - self.last_mouse_pos[0]) * self.scale_factor / self.zoom_level
                dy = (y - self.last_mouse_pos[1]) * self.scale_factor / self.zoom_level
                self.pan_x -= dx
                self.pan_y -= dy
                self.last_mouse_pos = (x, y)
                self.redraw_display()
                
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Zoom in/out
            if flags > 0:  # Scroll up
                self.zoom_level *= 1.2
            else:  # Scroll down
                self.zoom_level /= 1.2
                self.zoom_level = max(0.1, self.zoom_level)  # Minimum zoom
            self.redraw_display()
            
    def redraw_display(self):
        """Redraw the display with current zoom and pan."""
        # Create zoomed and panned view
        h, w = self.original_image.shape[:2]
        
        # Calculate visible region
        visible_w = int(w / (self.zoom_level * self.scale_factor))
        visible_h = int(h / (self.zoom_level * self.scale_factor))
        
        # Clamp pan values
        self.pan_x = max(0, min(w - visible_w, self.pan_x))
        self.pan_y = max(0, min(h - visible_h, self.pan_y))
        
        # Extract region
        x1, y1 = int(self.pan_x), int(self.pan_y)
        x2, y2 = int(self.pan_x + visible_w), int(self.pan_y + visible_h)
        
        region = self.original_image[y1:y2, x1:x2]
        
        if region.size == 0:
            # If region is empty, reset
            self.pan_x = 0
            self.pan_y = 0
            region = self.original_image
        
        # Resize for display
        self.current_display = cv2.resize(region, None,
                                         fx=self.zoom_level * self.display_scale,
                                         fy=self.zoom_level * self.display_scale,
                                         interpolation=cv2.INTER_LINEAR)
        
        # Redraw points
        for i, (px, py) in enumerate(self.points, 1):
            # Convert original coordinates to display coordinates
            display_x = int((px - self.pan_x) * self.zoom_level * self.display_scale)
            display_y = int((py - self.pan_y) * self.zoom_level * self.display_scale)
            
            if 0 <= display_x < self.current_display.shape[1] and 0 <= display_y < self.current_display.shape[0]:
                cv2.circle(self.current_display, (display_x, display_y), 8, (0, 255, 0), -1)
                cv2.putText(self.current_display, f"{i}", 
                           (display_x + 12, display_y - 12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add zoom/pan info
        info_text = f"Zoom: {self.zoom_level:.1f}x | Points: {len(self.points)} | Pan: ({self.pan_x:.0f}, {self.pan_y:.0f})"
        cv2.putText(self.current_display, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add instructions
        instructions = [
            "LEFT CLICK: Add point",
            "RIGHT CLICK: Remove last point",
            "MIDDLE DRAG: Pan",
            "SCROLL: Zoom in/out",
            "'q': Finish"
        ]
        
        y_offset = 60
        for line in instructions:
            cv2.putText(self.current_display, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 100), 1)
            y_offset += 25
        
        cv2.imshow(self.window_name, self.current_display)
    
    def select_points(self, num_points=4):
        """Select points with interactive zoom/pan."""
        print(f"\n{'='*60}")
        print(f"Selecting points for: {os.path.basename(self.image_path)}")
        print(f"Original size: {self.original_image.shape[1]}x{self.original_image.shape[0]}")
        print(f"Display size: {self.display_image.shape[1]}x{self.display_image.shape[0]}")
        print(f"\nInstructions:")
        print("  - LEFT CLICK: Add point")
        print("  - RIGHT CLICK: Remove last point")
        print("  - MIDDLE DRAG: Pan")
        print("  - SCROLL: Zoom in/out")
        print("  - Press 'q' when done")
        print(f"\nSelect at least {num_points} corresponding points in the same order on both images.")
        
        # Initial display
        self.current_display = self.display_image.copy()
        cv2.imshow(self.window_name, self.current_display)
        cv2.setMouseCallback(self.window_name, self.click_event)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') and len(self.points) >= num_points:
                break
            elif key == 27:  # ESC
                self.points = []
                break
            elif key == ord('r'):  # Reset view
                self.zoom_level = 1.0
                self.pan_x = 0
                self.pan_y = 0
                self.redraw_display()
            elif key == ord('c'):  # Clear all points
                self.points = []
                self.redraw_display()
                print("Cleared all points")
        
        cv2.destroyWindow(self.window_name)
        return np.array(self.points, dtype=np.float32)

def plot_comparison(img1, img2, pts1, pts2, H_opencv, H_svd, H_eig):
    """Visual comparison of different homography calculations with resizing."""
    # Resize large images for display
    max_display_size = 800
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    scale1 = min(1.0, max_display_size / max(h1, w1))
    scale2 = min(1.0, max_display_size / max(h2, w2))
    
    img1_display = cv2.resize(img1, None, fx=scale1, fy=scale1) if scale1 < 1 else img1.copy()
    img2_display = cv2.resize(img2, None, fx=scale2, fy=scale2) if scale2 < 1 else img2.copy()
    
    # Scale points for display
    pts1_display = pts1 * scale1
    pts2_display = pts2 * scale2
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot original images with points
    axes[0, 0].imshow(cv2.cvtColor(img1_display, cv2.COLOR_BGR2RGB))
    axes[0, 0].scatter(pts1_display[:, 0], pts1_display[:, 1], c='r', s=80, label='Source', alpha=0.7)
    axes[0, 0].set_title('Source Image with Points')
    axes[0, 0].legend()
    
    axes[0, 1].imshow(cv2.cvtColor(img2_display, cv2.COLOR_BGR2RGB))
    axes[0, 1].scatter(pts2_display[:, 0], pts2_display[:, 1], c='g', s=80, label='Target', alpha=0.7)
    axes[0, 1].set_title('Target Image with Points')
    axes[0, 1].legend()
    
    # Warp images using different homographies
    h, w = img1.shape[:2]
    display_size = (int(w * scale1), int(h * scale1))
    
    img_warped_opencv = cv2.warpPerspective(img1, H_opencv, (w, h))
    img_warped_svd = cv2.warpPerspective(img1, H_svd, (w, h))
    img_warped_eig = cv2.warpPerspective(img1, H_eig, (w, h))
    
    # Resize warped images for display
    img_warped_opencv_display = cv2.resize(img_warped_opencv, display_size) if scale1 < 1 else img_warped_opencv
    img_warped_svd_display = cv2.resize(img_warped_svd, display_size) if scale1 < 1 else img_warped_svd
    img_warped_eig_display = cv2.resize(img_warped_eig, display_size) if scale1 < 1 else img_warped_eig
    
    axes[1, 0].imshow(cv2.cvtColor(img_warped_opencv_display, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('OpenCV Warped')
    
    axes[1, 1].imshow(cv2.cvtColor(img_warped_svd_display, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('SVD DLT Warped')
    
    axes[1, 2].imshow(cv2.cvtColor(img_warped_eig_display, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('Eigen DLT Warped')
    
    # Overlay warped image on target for visual comparison
    img2_resized = cv2.resize(img2, (w, h))  # Resize target to same size as source for overlay
    for ax_idx, (H, name) in enumerate([(H_opencv, 'OpenCV'), (H_svd, 'SVD'), (H_eig, 'Eig')]):
        warped = cv2.warpPerspective(img1, H, (w, h))
        overlay = cv2.addWeighted(img2_resized, 0.5, warped, 0.5, 0)
        overlay_display = cv2.resize(overlay, display_size) if scale1 < 1 else overlay
        axes[0, 2 + ax_idx].imshow(cv2.cvtColor(overlay_display, cv2.COLOR_BGR2RGB))
        axes[0, 2 + ax_idx].set_title(f'{name} Overlay')
    
    plt.tight_layout()
    plt.show()

def test_with_synthetic_points():
    """Test with perfectly known synthetic points."""
    print("=" * 60)
    print("TEST 1: Synthetic Points")
    print("=" * 60)
    
    # Create a perfect homography (rotation + translation + scale)
    H_true = np.array([[1.2, -0.3, 100],
                       [0.4, 0.9, 50],
                       [0.001, -0.002, 1]])
    
    # Generate random source points
    np.random.seed(42)
    src_pts = np.random.rand(6, 2) * 300 + 100  # Points in [100, 400] range
    
    # Transform to get destination points
    src_homo = np.column_stack([src_pts, np.ones(len(src_pts))])
    dst_homo = (H_true @ src_homo.T).T
    dst_pts = dst_homo[:, :2] / dst_homo[:, 2:]
    
    print(f"Source points:\n{src_pts}")
    print(f"\nDestination points:\n{dst_pts}")
    
    # Calculate homography using different methods
    point_pairs = list(zip(src_pts, dst_pts))
    
    # OpenCV
    H_opencv, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    
    # Your DLT implementations
    H_svd, H_eig = calculate_normalized_homography(src_pts, dst_pts)
    
    # Validate
    print(f"\nTrue H:\n{H_true}")
    print(f"\nOpenCV H:\n{H_opencv}")
    print(f"\nSVD DLT H:\n{H_svd}")
    print(f"\nEigen DLT H:\n{H_eig}")
    
    # Calculate errors
    print("\n" + "=" * 40)
    validate_homography(H_opencv, src_pts, dst_pts, "OpenCV")
    validate_homography(H_svd, src_pts, dst_pts, "SVD DLT")
    validate_homography(H_eig, src_pts, dst_pts, "Eigen DLT")
    
    # Compare differences
    print("\n" + "=" * 40)
    print("Matrix Differences (Frobenius norm):")
    print(f"OpenCV vs True: {np.linalg.norm(H_opencv - H_true):.6f}")
    print(f"SVD vs True: {np.linalg.norm(H_svd - H_true):.6f}")
    print(f"Eigen vs True: {np.linalg.norm(H_eig - H_true):.6f}")
    
    return True

def test_with_real_images(img1_path, img2_path, num_points=6):
    """Test with real images using improved point selector."""
    print("=" * 60)
    print("TEST 2: Real Images - Improved Point Selection")
    print("=" * 60)
    
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print(f"Error loading images: {img1_path}, {img2_path}")
        return False
    
    print(f"Image 1: {img1.shape}, Image 2: {img2.shape}")
    
    # Create point selectors
    selector1 = ImagePointSelector(img1_path, display_scale=0.4, window_name="Image 1 - Select Points")
    selector2 = ImagePointSelector(img2_path, display_scale=0.4, window_name="Image 2 - Select Points")
    
    try:
        # Select points from first image
        pts1 = selector1.select_points(num_points)
        
        if len(pts1) < 4:
            print("Not enough points selected for first image!")
            return False
        
        print(f"\nSelected {len(pts1)} points from first image")
        
        # Select corresponding points from second image
        print("\n" + "="*40)
        print("NOW SELECT THE SAME POINTS IN THE SAME ORDER ON THE SECOND IMAGE!")
        print("Make sure to select corresponding features.")
        print("="*40)
        
        pts2 = selector2.select_points(num_points)
        
        if len(pts2) < 4:
            print("Not enough points selected for second image!")
            return False
        
        print(f"\nSelected {len(pts2)} points from second image")
        
        if len(pts1) != len(pts2):
            print(f"Warning: Different number of points selected ({len(pts1)} vs {len(pts2)})")
            min_points = min(len(pts1), len(pts2))
            pts1 = pts1[:min_points]
            pts2 = pts2[:min_points]
        
        print(f"\nSelected {len(pts1)} corresponding points:")
        for i, (p1, p2) in enumerate(zip(pts1, pts2)):
            print(f"Point {i+1}: Image1 {p1.astype(int)} -> Image2 {p2.astype(int)}")
        
        # Calculate homographies
        H_opencv, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        H_svd, H_eig = calculate_normalized_homography(pts1, pts2)
        
        # Validate
        print("\n" + "=" * 50)
        avg_err_opencv, max_err_opencv = validate_homography(H_opencv, pts1, pts2, "OpenCV")
        avg_err_svd, max_err_svd = validate_homography(H_svd, pts1, pts2, "SVD DLT")
        avg_err_eig, max_err_eig = validate_homography(H_eig, pts1, pts2, "Eigen DLT")
        
        print("\n" + "=" * 50)
        print("SUMMARY:")
        print(f"OpenCV: Avg Error = {avg_err_opencv:.2f} px, Max Error = {max_err_opencv:.2f} px")
        print(f"SVD DLT: Avg Error = {avg_err_svd:.2f} px, Max Error = {max_err_svd:.2f} px")
        print(f"Eigen DLT: Avg Error = {avg_err_eig:.2f} px, Max Error = {max_err_eig:.2f} px")
        
        # Check if SVD DLT is close to OpenCV
        diff_svd_opencv = np.linalg.norm(H_svd - H_opencv)
        diff_eig_opencv = np.linalg.norm(H_eig - H_opencv)
        
        print(f"\nMatrix Differences (Frobenius norm):")
        print(f"SVD DLT vs OpenCV: {diff_svd_opencv:.6f}")
        print(f"Eigen DLT vs OpenCV: {diff_eig_opencv:.6f}")
        
        # Calculate percentage difference
        if np.linalg.norm(H_opencv) > 0:
            percent_diff_svd = 100 * diff_svd_opencv / np.linalg.norm(H_opencv)
            percent_diff_eig = 100 * diff_eig_opencv / np.linalg.norm(H_opencv)
            print(f"\nPercentage difference from OpenCV:")
            print(f"SVD DLT: {percent_diff_svd:.2f}%")
            print(f"Eigen DLT: {percent_diff_eig:.2f}%")
        
        # Ask user if they want to see visualization
        print("\n" + "=" * 50)
        response = input("Show visualization? (y/n): ").lower()
        if response == 'y':
            plot_comparison(img1, img2, pts1, pts2, H_opencv, H_svd, H_eig)
        
        # Save points for future use
        save_points = input("\nSave points to file for future use? (y/n): ").lower()
        if save_points == 'y':
            filename = input("Enter filename (without extension): ")
            np.savez(f"{filename}_points.npz", pts1=pts1, pts2=pts2, 
                    img1_path=img1_path, img2_path=img2_path)
            print(f"Points saved to {filename}_points.npz")
        
        return True
        
    except Exception as e:
        print(f"Error during point selection: {e}")
        return False

def quick_test_with_saved_points(points_file):
    """Quick test using previously saved points."""
    print("=" * 60)
    print("QUICK TEST: Using Saved Points")
    print("=" * 60)
    
    data = np.load(points_file)
    pts1 = data['pts1']
    pts2 = data['pts2']
    img1_path = str(data['img1_path'])
    img2_path = str(data['img2_path'])
    
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    print(f"Loaded {len(pts1)} points from {points_file}")
    print(f"Images: {img1_path}, {img2_path}")
    
    # Calculate homographies
    H_opencv, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    H_svd, H_eig = calculate_normalized_homography(pts1, pts2)
    
    # Validate
    print("\n" + "=" * 50)
    avg_err_opencv, max_err_opencv = validate_homography(H_opencv, pts1, pts2, "OpenCV")
    avg_err_svd, max_err_svd = validate_homography(H_svd, pts1, pts2, "SVD DLT")
    avg_err_eig, max_err_eig = validate_homography(H_eig, pts1, pts2, "Eigen DLT")
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"OpenCV: Avg Error = {avg_err_opencv:.2f} px, Max Error = {max_err_opencv:.2f} px")
    print(f"SVD DLT: Avg Error = {avg_err_svd:.2f} px, Max Error = {max_err_svd:.2f} px")
    print(f"Eigen DLT: Avg Error = {avg_err_eig:.2f} px, Max Error = {max_err_eig:.2f} px")
    
    # Visual comparison
    plot_comparison(img1, img2, pts1, pts2, H_opencv, H_svd, H_eig)
    
    return True

def main():
    """Main test function with improved UI."""
    print("IMPROVED HOMOGRAPHY CALCULATION TEST SUITE")
    print("=" * 70)
    
    # Test 1: Synthetic points (quick validation)
    print("\nRunning synthetic point test...")
    test_with_synthetic_points()
    
    print("\n" + "=" * 70)
    print("Press Enter to continue to real image test...")
    input()
    
    # Check command line arguments
    if len(sys.argv) > 2:
        img1_path = sys.argv[1]
        img2_path = sys.argv[2]
        
        print(f"\nTesting with images:")
        print(f"  Image 1: {img1_path}")
        print(f"  Image 2: {img2_path}")
        
        # Ask for number of points
        num_points = input(f"\nHow many points to select? (default: 6): ")
        num_points = int(num_points) if num_points.isdigit() else 6
        
        test_with_real_images(img1_path, img2_path, num_points)
        
    elif len(sys.argv) == 2 and sys.argv[1].endswith('.npz'):
        # Load saved points
        quick_test_with_saved_points(sys.argv[1])
    else:
        print("\nNo image paths provided.")
        print("\nUsage options:")
        print("  1. Test with two images:")
        print("     python test_homography_improved.py <image1_path> <image2_path>")
        print("     Example: python test_homography.py ../dataset/Dev0/Dev0_Image_w1920_h1200_fn1.jpg ../dataset/Dev3/Dev3_Image_w1920_h1200_fn1.jpg")
        print("\n  2. Test with saved points:")
        print("     python test_homography_improved.py <saved_points.npz>")
        print("\n  3. Manual entry:")
        img1_path = input("\nEnter path to first image: ")
        img2_path = input("Enter path to second image: ")
        
        if os.path.exists(img1_path) and os.path.exists(img2_path):
            test_with_real_images(img1_path, img2_path, num_points=6)
        else:
            print("One or both image paths are invalid!")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print("\nKey Findings:")
    print("1. SVD-based DLT should closely match OpenCV's findHomography")
    print("2. Eigenvalue-based DLT may have numerical issues")
    print("3. Normalized DLT improves stability")
    print("4. Use at least 4 well-distributed points for best results")
    print("\nRecommendation: Use the SVD version in pipeline.py")

if __name__ == "__main__":
    # Create a sample test if run without arguments
    if len(sys.argv) == 1:
        print("No arguments provided. Running synthetic test only.")
        test_with_synthetic_points()
    else:
        main()