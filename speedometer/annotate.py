# annotate_improved.py
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
        # Default names based on common annotation patterns
        if n_points == 6:
            point_names = ['TL', 'TR', 'ML', 'MR', 'BL', 'BR']  # Top-Left, Top-Right, etc.
        else:
            point_names = [f'P{i+1}' for i in range(n_points)]
    
    # Display instructions
    title_text = window_title or f"Annotate {n_points} points (in order!)"
    instructions = f"Click points in this order:\n" + "\n".join([f"{i+1}. {point_names[i]}" for i in range(n_points)])
    
    ax.set_title(title_text, fontsize=14, fontweight='bold')
    plt.figtext(0.02, 0.02, instructions, fontsize=12, bbox=dict(facecolor='yellow', alpha=0.7))
    
    points = []
    annotated_points = []
    
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
        
        # Check for common patterns
        print(f"\nDetected pattern:")
        if abs(x_coords.max() - x_coords.min()) > abs(y_coords.max() - y_coords.min()):
            print("  Horizontal distribution (likely left/right cameras)")
        else:
            print("  Vertical distribution (likely forward-facing camera)")
        
        return points_array
    else:
        print("Warning: Not all points were annotated!")
        return None

def visualize_annotation(image_path, points, point_names=None, save_path=None):
    """Visualize annotated points on the image."""
    img = cv2.imread(image_path)
    img_display = img.copy()
    
    if point_names is None and len(points) == 6:
        point_names = ['TL', 'TR', 'ML', 'MR', 'BL', 'BR']
    
    for i, (x, y) in enumerate(points):
        # Draw circle
        cv2.circle(img_display, (int(x), int(y)), 8, (0, 255, 0), -1)
        cv2.circle(img_display, (int(x), int(y)), 10, (0, 0, 255), 2)
        
        # Draw number and name
        text = f"{i+1}"
        if point_names and i < len(point_names):
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
    cv2.waitKey(3000)  # Show for 3 seconds
    cv2.destroyAllWindows()

def load_annotation_or_annotate_improved(
    dev0_sample, 
    dev3_sample, 
    dev0_json=None, 
    dev3_json=None, 
    n_points=6,
    reannotate=False,
    visualize=True
):
    """
    Improved version with better visualization and verification.
    
    Args:
        reannotate: If True, force re-annotation even if JSON exists
        visualize: If True, show visualization of annotations
    """
    annotations_dir = "./annotations"
    if dev0_json:
        annotations_dir = os.path.dirname(dev0_json)
    os.makedirs(annotations_dir, exist_ok=True)
    
    # Define point names for clarity
    point_names = ['Top-Left', 'Top-Right', 'Middle-Left', 'Middle-Right', 'Bottom-Left', 'Bottom-Right']
    
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
        print("Annotating Dev0 points...")
        print("IMPORTANT: Click points in this exact order:")
        for i, name in enumerate(point_names):
            print(f"  {i+1}. {name}")
        print("\nPoints should be on the ground plane (road surface).")
        
        pts0 = annotate_with_numbers(dev0_sample, n_points, point_names, "Annotate Dev0 (Left Camera)")
        
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
        print("Annotating Dev3 points...")
        print("CRITICAL: Click the SAME PHYSICAL POINTS in the SAME ORDER as Dev0!")
        print("Points should correspond to:")
        for i, name in enumerate(point_names):
            print(f"  {i+1}. {name} (same physical location as Dev0)")
        
        pts3 = annotate_with_numbers(dev3_sample, n_points, point_names, "Annotate Dev3 (Right Camera)")
        
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
        
        # Check if patterns match
        if abs(dev0_x_range - dev3_x_range) / max(dev0_x_range, dev3_x_range) > 0.3:
            print("⚠ WARNING: X ranges differ significantly between cameras!")
        
        if abs(dev0_y_range - dev3_y_range) / max(dev0_y_range, dev3_y_range) > 0.3:
            print("⚠ WARNING: Y ranges differ significantly between cameras!")
        
        # Create a side-by-side visualization
        if visualize:
            img0 = cv2.imread(dev0_sample)
            img3 = cv2.imread(dev3_sample)
            
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
                cv2.putText(img0_small, str(i+1), (int(x)+10, int(y)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            for i, (x, y) in enumerate(pts3_small):
                cv2.circle(img3_small, (int(x), int(y)), 6, (0, 255, 0), -1)
                cv2.putText(img3_small, str(i+1), (int(x)+10, int(y)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Combine images
            combined = np.hstack([img0_small, img3_small])
            
            # Add labels
            cv2.putText(combined, "Dev0 (Left Camera)", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(combined, "Dev3 (Right Camera)", (img0_small.shape[1] + 20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            cv2.imshow("Annotation Comparison (Check that numbers match same physical points!)", combined)
            cv2.waitKey(5000)  # Show for 5 seconds
            cv2.destroyAllWindows()
            
            # Save comparison
            comparison_path = os.path.join(annotations_dir, "annotation_comparison.jpg")
            cv2.imwrite(comparison_path, combined)
            print(f"✓ Comparison visualization saved to {comparison_path}")
    
    return pts0, pts3

def verify_annotation_order(pts0, pts3, point_names=None):
    """Verify that points are in the correct order by analyzing patterns."""
    if point_names is None:
        point_names = ['TL', 'TR', 'ML', 'MR', 'BL', 'BR']
    
    print("\n" + "="*70)
    print("ORDER VERIFICATION")
    print("="*70)
    
    # Analyze point patterns
    for name, points in [("Dev0", pts0), ("Dev3", pts3)]:
        print(f"\n{name} point analysis:")
        
        # Sort by X coordinate
        x_sorted = points[np.argsort(points[:, 0])]
        print(f"  Sorted by X (left to right):")
        for i, (x, y) in enumerate(x_sorted):
            print(f"    {i+1}. ({x:.1f}, {y:.1f})")
        
        # Sort by Y coordinate
        y_sorted = points[np.argsort(points[:, 1])]
        print(f"  Sorted by Y (top to bottom):")
        for i, (x, y) in enumerate(y_sorted):
            print(f"    {i+1}. ({x:.1f}, {y:.1f})")
        
        # Check for grid pattern
        x_values = points[:, 0]
        y_values = points[:, 1]
        
        # Find unique X and Y values (for grid detection)
        unique_x = np.unique(np.round(x_values, -1))  # Round to nearest 10px
        unique_y = np.unique(np.round(y_values, -1))
        
        print(f"  Unique X values: {len(unique_x)} (suggests {len(unique_x)} columns)")
        print(f"  Unique Y values: {len(unique_y)} (suggests {len(unique_y)} rows)")
        
        if len(unique_x) == 2 and len(unique_y) == 3:
            print("  ✓ Pattern detected: 2 columns × 3 rows grid")
            print("  Expected order: Left column top→bottom, then right column top→bottom")
        elif len(unique_x) == 3 and len(unique_y) == 2:
            print("  ✓ Pattern detected: 3 columns × 2 rows grid")
            print("  Expected order: Top row left→right, then bottom row left→right")
        else:
            print("  ⚠ Unclear grid pattern")
    
    return True

# Main function for testing
if __name__ == "__main__":
    dev0_img = "../dataset/Dev0/Dev0_Image_w1920_h1200_fn1.jpg"
    dev3_img = "../dataset/Dev3/Dev3_Image_w1920_h1200_fn1.jpg"
    dev0_json = "../annotations/dev0_pts.json"
    dev3_json = "../annotations/dev3_pts.json"
    
    # Use this to re-annotate if needed
    reannotate = False  # Set to True to force re-annotation
    
    pts0, pts3 = load_annotation_or_annotate_improved(
        dev0_img, dev3_img, dev0_json, dev3_json, 
        n_points=6, 
        reannotate=reannotate,
        visualize=True
    )
    
    if pts0 is not None and pts3 is not None:
        verify_annotation_order(pts0, pts3)
        
        print("\n" + "="*70)
        print("ANNOTATION COMPLETE")
        print("="*70)
        print("\nDev0 Points:")
        for i, (x, y) in enumerate(pts0):
            print(f"  {i+1}. ({x:.1f}, {y:.1f})")
        
        print("\nDev3 Points:")
        for i, (x, y) in enumerate(pts3):
            print(f"  {i+1}. ({x:.1f}, {y:.1f})")