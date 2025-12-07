# annotate_final.py
import cv2
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def annotate_with_numbers(image_path, n_points=6, point_names=None, window_title=None):
    """
    Annotate points with numbers displayed, making it clear which point is which.
    
    Args:
        image_path: Path to the image
        n_points: Number of points to annotate
        point_names: Optional names for each point (e.g., ['TL', 'TR', 'ML', 'MR', 'BL', 'BR'])
        window_title: Title for the annotation window
    
    Returns:
        Array of annotated points in order
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    
    # Convert to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure with instructions
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img_rgb)
    
    if point_names is None:
        point_names = [f'P{i+1}' for i in range(n_points)]
    
    # Display instructions
    title_text = window_title or f"Annotate {n_points} points (in order!)"
    instructions = f"Click points in this order:\n" + "\n".join([f"{i+1}. {point_names[i]}" for i in range(n_points)])
    
    ax.set_title(title_text, fontsize=14, fontweight='bold')
    plt.figtext(0.02, 0.02, instructions, fontsize=12, bbox=dict(facecolor='yellow', alpha=0.7))
    
    points = []
    
    def onclick(event):
        if event.inaxes != ax:
            return
        
        # Add point
        x, y = event.xdata, event.ydata
        points.append((x, y))
        
        # Draw circle and number
        circle = Circle((x, y), radius=15, color='red', fill=True, alpha=0.7)
        ax.add_patch(circle)
        
        # Add number text
        point_num = len(points)
        ax.text(x, y, str(point_num), 
                color='white', fontsize=12, fontweight='bold',
                ha='center', va='center')
        
        # Add name if available
        if point_num <= len(point_names):
            ax.text(x, y - 20, point_names[point_num-1], 
                    color='blue', fontsize=10, fontweight='bold',
                    ha='center', va='center')
        
        # Redraw
        fig.canvas.draw()
        
        print(f"Point {point_num} ({point_names[point_num-1]}): ({x:.1f}, {y:.1f})")
        
        # Check if done
        if len(points) == n_points:
            plt.close(fig)
    
    # Connect the click handler
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
    # Convert to numpy array
    if len(points) == n_points:
        points_array = np.array(points, dtype=np.float32)
        
        # Verify points make sense
        print("\n" + "="*60)
        print(f"Annotation completed for {os.path.basename(image_path)}")
        print("="*60)
        
        # Show point statistics
        x_coords = points_array[:, 0]
        y_coords = points_array[:, 1]
        
        print(f"\nPoint coordinates:")
        for i, (name, (x, y)) in enumerate(zip(point_names, points_array)):
            print(f"  {i+1}. {name}: ({x:.1f}, {y:.1f})")
        
        print(f"\nStatistics:")
        print(f"  X range: {x_coords.min():.1f} to {x_coords.max():.1f} (width: {x_coords.max()-x_coords.min():.1f})")
        print(f"  Y range: {y_coords.min():.1f} to {y_coords.max():.1f} (height: {y_coords.max()-y_coords.min():.1f})")
        
        return points_array
    else:
        print("Warning: Not all points were annotated!")
        return None

def get_annotation_strategy(strategy_name="column"):
    """
    Get point names and order based on strategy.
    
    Args:
        strategy_name: "column" or "row"
            - "column": Left column top→bottom, then right column top→bottom (2×3 grid)
            - "row": Top row left→right, then bottom row left→right (3×2 grid)
    
    Returns:
        point_names, description
    """
    if strategy_name.lower() == "column":
        point_names = ['Left-Top', 'Left-Middle', 'Left-Bottom', 
                      'Right-Top', 'Right-Middle', 'Right-Bottom']
        description = "2 columns × 3 rows: Left column (top→bottom), then Right column (top→bottom)"
    elif strategy_name.lower() == "row":
        point_names = ['Top-Left', 'Top-Middle', 'Top-Right',
                      'Bottom-Left', 'Bottom-Middle', 'Bottom-Right']
        description = "3 columns × 2 rows: Top row (left→right), then Bottom row (left→right)"
    else:
        point_names = [f'P{i+1}' for i in range(6)]
        description = "Simple sequential order"
    
    return point_names, description

def load_annotation_or_annotate_final(
    dev0_sample, 
    dev3_sample, 
    dev0_json=None, 
    dev3_json=None, 
    n_points=6,
    strategy="column",
    reannotate=False,
    visualize=True
):
    """
    Final version with multiple annotation strategies.
    
    Args:
        strategy: "column" or "row" - determines annotation order
        reannotate: If True, force re-annotation even if JSON exists
        visualize: If True, show visualization of annotations
    """
    annotations_dir = "./annotations"
    if dev0_json:
        annotations_dir = os.path.dirname(dev0_json)
    os.makedirs(annotations_dir, exist_ok=True)
    
    # Get point names based on strategy
    point_names, strategy_desc = get_annotation_strategy(strategy)
    
    print("\n" + "="*70)
    print(f"ANNOTATION STRATEGY: {strategy.upper()}")
    print("="*70)
    print(f"Pattern: {strategy_desc}")
    
    # --- Annotate Dev0 ---
    print("\n" + "="*70)
    print("DEV0 ANNOTATION")
    print("="*70)
    
    if not reannotate and dev0_json and os.path.exists(dev0_json):
        pts0 = np.array(json.load(open(dev0_json)), dtype=np.float32)
        print(f"✔ Loaded existing Dev0 points from {dev0_json}")
        
        if visualize:
            visualize_annotation(dev0_sample, pts0, point_names, 
                               os.path.join(annotations_dir, "dev0_visualization.jpg"))
    else:
        print(f"Annotating Dev0 points using {strategy} strategy...")
        print(f"STRATEGY: {strategy_desc}")
        print("\nClick points in this exact order:")
        for i, name in enumerate(point_names):
            print(f"  {i+1}. {name}")
        print("\nPoints should be on the ground plane (road surface).")
        
        pts0 = annotate_with_numbers(dev0_sample, n_points, point_names, 
                                    f"Annotate Dev0 ({strategy.upper()} strategy)")
        
        if pts0 is not None and dev0_json:
            with open(dev0_json, 'w') as f:
                json.dump(pts0.tolist(), f)
            print(f"✓ Saved Dev0 points to {dev0_json}")
            
            if visualize:
                visualize_annotation(dev0_sample, pts0, point_names,
                                   os.path.join(annotations_dir, "dev0_visualization.jpg"))
    
    # --- Annotate Dev3 ---
    print("\n" + "="*70)
    print("DEV3 ANNOTATION")
    print("="*70)
    
    if not reannotate and dev3_json and os.path.exists(dev3_json):
        pts3 = np.array(json.load(open(dev3_json)), dtype=np.float32)
        print(f"✔ Loaded existing Dev3 points from {dev3_json}")
        
        if visualize:
            visualize_annotation(dev3_sample, pts3, point_names,
                               os.path.join(annotations_dir, "dev3_visualization.jpg"))
    else:
        print(f"Annotating Dev3 points using {strategy} strategy...")
        print("CRITICAL: Click the SAME PHYSICAL POINTS in the SAME ORDER as Dev0!")
        print(f"Points should correspond to ({strategy} strategy):")
        for i, name in enumerate(point_names):
            print(f"  {i+1}. {name} (same physical location as Dev0)")
        
        pts3 = annotate_with_numbers(dev3_sample, n_points, point_names, 
                                    f"Annotate Dev3 ({strategy.upper()} strategy)")
        
        if pts3 is not None and dev3_json:
            with open(dev3_json, 'w') as f:
                json.dump(pts3.tolist(), f)
            print(f"✓ Saved Dev3 points to {dev3_json}")
            
            if visualize:
                visualize_annotation(dev3_sample, pts3, point_names,
                                   os.path.join(annotations_dir, "dev3_visualization.jpg"))
    
    # Verify annotations
    if pts0 is not None and pts3 is not None:
        print("\n" + "="*70)
        print("ANNOTATION VERIFICATION")
        print("="*70)
        
        # Add strategy-specific verification
        verify_annotation_strategy(pts0, pts3, strategy, point_names)
        
        print(f"\nNumber of points: Dev0={len(pts0)}, Dev3={len(pts3)}")
        
        # Check point distribution
        print("\nPoint distribution analysis:")
        
        # Dev0
        dev0_x_range = pts0[:, 0].max() - pts0[:, 0].min()
        dev0_y_range = pts0[:, 1].max() - pts0[:, 1].min()
        print(f"Dev0: X range = {dev0_x_range:.1f}px, Y range = {dev0_y_range:.1f}px")
        
        # Dev3  
        dev3_x_range = pts3[:, 0].max() - pts3[:, 0].min()
        dev3_y_range = pts3[:, 1].max() - pts3[:, 1].min()
        print(f"Dev3: X range = {dev3_x_range:.1f}px, Y range = {dev3_y_range:.1f}px")
        
        # Create a side-by-side visualization
        if visualize:
            create_comparison_visualization(dev0_sample, dev3_sample, pts0, pts3, 
                                          point_names, annotations_dir)
    
    return pts0, pts3, point_names

def verify_annotation_strategy(pts0, pts3, strategy, point_names):
    """Verify annotations match the chosen strategy."""
    print(f"\nVerifying {strategy.upper()} strategy pattern...")
    
    if strategy == "column":
        # For column strategy: First 3 points should be left column, last 3 right column
        # Points 0-2: Left column (should have similar X coordinates)
        # Points 3-5: Right column (should have similar X coordinates)
        
        left_col_pts0 = pts0[:3]
        right_col_pts0 = pts0[3:]
        
        left_col_pts3 = pts3[:3]
        right_col_pts3 = pts3[3:]
        
        # Check X consistency within columns
        left_x_std0 = np.std(left_col_pts0[:, 0])
        right_x_std0 = np.std(right_col_pts0[:, 0])
        left_x_std3 = np.std(left_col_pts3[:, 0])
        right_x_std3 = np.std(right_col_pts3[:, 0])
        
        print(f"  Left column X std: Dev0={left_x_std0:.1f}, Dev3={left_x_std3:.1f}")
        print(f"  Right column X std: Dev0={right_x_std0:.1f}, Dev3={right_x_std3:.1f}")
        
        if left_x_std0 < 30 and right_x_std0 < 30 and left_x_std3 < 30 and right_x_std3 < 30:
            print("  ✓ Columns are well-aligned!")
        else:
            print("  ⚠ Columns may not be properly aligned")
        
        # Check Y ordering (should be top to bottom within each column)
        for i in range(3):
            if left_col_pts0[i, 1] > left_col_pts0[min(i+1, 2), 1]:
                print(f"  ⚠ Left column point {i+1} may not be above point {i+2}")
        
    elif strategy == "row":
        # For row strategy: First 3 points top row, last 3 bottom row
        top_row_pts0 = pts0[:3]
        bottom_row_pts0 = pts0[3:]
        
        # Check Y consistency within rows
        top_y_std0 = np.std(top_row_pts0[:, 1])
        bottom_y_std0 = np.std(bottom_row_pts0[:, 1])
        
        print(f"  Top row Y std: Dev0={top_y_std0:.1f}")
        print(f"  Bottom row Y std: Dev0={bottom_y_std0:.1f}")
        
        if top_y_std0 < 30 and bottom_y_std0 < 30:
            print("  ✓ Rows are well-aligned!")
        else:
            print("  ⚠ Rows may not be properly aligned")

def create_comparison_visualization(img0_path, img3_path, pts0, pts3, point_names, save_dir):
    """Create side-by-side comparison visualization."""
    img0 = cv2.imread(img0_path)
    img3 = cv2.imread(img3_path)
    
    # Resize to same height for comparison
    h0, w0 = img0.shape[:2]
    h3, w3 = img3.shape[:2]
    height = min(h0, h3, 600)
    
    scale0 = height / h0
    scale3 = height / h3
    
    img0_small = cv2.resize(img0, (int(w0*scale0), int(h0*scale0)))
    img3_small = cv2.resize(img3, (int(w3*scale3), int(h3*scale3)))
    
    # Scale points
    pts0_small = pts0 * scale0
    pts3_small = pts3 * scale3
    
    # Draw points
    for i, (x, y) in enumerate(pts0_small):
        cv2.circle(img0_small, (int(x), int(y)), 6, (0, 255, 0), -1)
        cv2.putText(img0_small, f"{i+1}:{point_names[i]}", (int(x)+10, int(y)-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    for i, (x, y) in enumerate(pts3_small):
        cv2.circle(img3_small, (int(x), int(y)), 6, (0, 255, 0), -1)
        cv2.putText(img3_small, f"{i+1}:{point_names[i]}", (int(x)+10, int(y)-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    # Combine images
    combined = np.hstack([img0_small, img3_small])
    
    # Add labels
    cv2.putText(combined, "Dev0 (Left Camera)", (20, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(combined, "Dev3 (Right Camera)", (img0_small.shape[1] + 20, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Add strategy info
    cv2.putText(combined, f"Strategy: Column (Left→Right, Top→Bottom)", (20, height - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    comparison_path = os.path.join(save_dir, "annotation_comparison_final.jpg")
    cv2.imwrite(comparison_path, combined)
    print(f"✓ Comparison visualization saved to {comparison_path}")
    
    cv2.imshow("Annotation Comparison", combined)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

def visualize_annotation(image_path, points, point_names=None, save_path=None):
    """Visualize annotated points on the image."""
    img = cv2.imread(image_path)
    img_display = img.copy()
    
    if point_names is None:
        point_names = [f'P{i+1}' for i in range(len(points))]
    
    for i, (x, y) in enumerate(points):
        # Draw circle
        cv2.circle(img_display, (int(x), int(y)), 8, (0, 255, 0), -1)
        cv2.circle(img_display, (int(x), int(y)), 10, (0, 0, 255), 2)
        
        # Draw number and name
        text = f"{i+1}:{point_names[i]}"
        cv2.putText(img_display, text, 
                   (int(x) + 15, int(y) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Resize for display if too large
    h, w = img_display.shape[:2]
    if w > 1280:
        scale = 1280 / w
        img_display = cv2.resize(img_display, (int(w*scale), int(h*scale)))
    
    if save_path:
        cv2.imwrite(save_path, img_display)
        print(f"Visualization saved to {save_path}")
    
    cv2.imshow("Annotation Visualization", img_display)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

def test_homography(pts0, pts3):
    """Test if homography can be estimated from points."""
    try:
        H, mask = cv2.findHomography(pts0, pts3, cv2.RANSAC, 5.0)
        inliers = np.sum(mask)
        print(f"\nHomography test: {inliers}/{len(pts0)} inliers")
        
        if inliers < 4:
            print("⚠ WARNING: Not enough inliers for reliable homography!")
        else:
            print("✓ Good: Homography can be reliably estimated")
            
        # Show which points are outliers
        if inliers < len(pts0):
            outliers = [i+1 for i, m in enumerate(mask.flatten()) if m == 0]
            print(f"  Outliers: Points {outliers}")
            
        return H, mask
    except Exception as e:
        print(f"✗ Homography failed: {e}")
        return None, None

# Main function for testing
if __name__ == "__main__":
    dev0_img = "../dataset/Dev0/Dev0_Image_w1920_h1200_fn1.jpg"
    dev3_img = "../dataset/Dev3/Dev3_Image_w1920_h1200_fn1.jpg"
    dev0_json = "../annotations/dev0_pts.json"
    dev3_json = "../annotations/dev3_pts.json"
    
    # CHOOSE YOUR STRATEGY HERE:
    # "column": Left column top→bottom, then right column top→bottom (2×3 grid)
    # "row": Top row left→right, then bottom row left→right (3×2 grid)
    annotation_strategy = "column"  # Change this to "row" if needed
    
    # Set to True to force re-annotation
    reannotate = True
    
    print("\n" + "="*70)
    print("FINAL ANNOTATION TOOL")
    print("="*70)
    print(f"Selected strategy: {annotation_strategy.upper()}")
    print("="*70)
    
    pts0, pts3, point_names = load_annotation_or_annotate_final(
        dev0_img, dev3_img, dev0_json, dev3_json, 
        n_points=6, 
        strategy=annotation_strategy,
        reannotate=reannotate,
        visualize=True
    )
    
    if pts0 is not None and pts3 is not None:
        print("\n" + "="*70)
        print("FINAL VERIFICATION")
        print("="*70)
        
        print("\nDev0 Points:")
        for i, (x, y) in enumerate(pts0):
            print(f"  {i+1}. {point_names[i]}: ({x:.1f}, {y:.1f})")
        
        print("\nDev3 Points:")
        for i, (x, y) in enumerate(pts3):
            print(f"  {i+1}. {point_names[i]}: ({x:.1f}, {y:.1f})")
        
        # Test homography
        H, mask = test_homography(pts0, pts3)
        
        # Generate world coordinates based on strategy
        print("\n" + "="*70)
        print("WORLD COORDINATES SUGGESTION")
        print("="*70)
        
        if annotation_strategy == "column":
            # For column strategy (2 columns × 3 rows)
            print("\nSuggested world coordinates (meters):")
            print("Format: (X, Y) where X is across road, Y is along road")
            print("\nPoint 1 (Left-Top):    (0, 2)    # Top-left")
            print("Point 2 (Left-Middle): (0, 1)    # Middle-left")
            print("Point 3 (Left-Bottom): (0, 0)    # Bottom-left")
            print("Point 4 (Right-Top):   (1, 2)    # Top-right")
            print("Point 5 (Right-Middle):(1, 1)    # Middle-right")
            print("Point 6 (Right-Bottom):(1, 0)    # Bottom-right")
            print("\nNote: Adjust scale based on actual road measurements.")
            
        elif annotation_strategy == "row":
            # For row strategy (3 columns × 2 rows)
            print("\nSuggested world coordinates (meters):")
            print("Format: (X, Y) where X is across road, Y is along road")
            print("\nPoint 1 (Top-Left):     (0, 1)    # Top-left")
            print("Point 2 (Top-Middle):   (1, 1)    # Top-middle")
            print("Point 3 (Top-Right):    (2, 1)    # Top-right")
            print("Point 4 (Bottom-Left):  (0, 0)    # Bottom-left")
            print("Point 5 (Bottom-Middle):(1, 0)    # Bottom-middle")
            print("Point 6 (Bottom-Right): (2, 0)    # Bottom-right")