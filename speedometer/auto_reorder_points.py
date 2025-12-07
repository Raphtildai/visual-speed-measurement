# auto_reorder_points.py
import json
import numpy as np
import cv2

def visualize_points(image_path, points, title="Points", save_path=None):
    """Visualize points on image."""
    img = cv2.imread(image_path)
    if img is None:
        return
    
    for i, (x, y) in enumerate(points):
        cv2.circle(img, (int(x), int(y)), 8, (0, 255, 0), -1)
        cv2.putText(img, f"{i+1}", (int(x)+12, int(y)-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Resize if too large
    h, w = img.shape[:2]
    if w > 1000:
        scale = 1000 / w
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
    
    cv2.imshow(title, img)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
    
    if save_path:
        cv2.imwrite(save_path, img)

def auto_detect_grid_order(points, camera_name="Camera"):
    """
    Automatically detect the correct grid order from points.
    Assumes a 2x3 grid (2 columns, 3 rows).
    """
    print(f"\n{'='*60}")
    print(f"AUTO-DETECTING GRID ORDER FOR {camera_name}")
    print('='*60)
    
    # Step 1: Cluster by X to find columns
    x_coords = points[:, 0]
    
    # Simple clustering: sort by X and split into 2 groups
    x_sorted_indices = np.argsort(x_coords)
    left_indices = x_sorted_indices[:3]  # 3 leftmost points
    right_indices = x_sorted_indices[3:]  # 3 rightmost points
    
    left_points = points[left_indices]
    right_points = points[right_indices]
    
    print(f"\nLeft column points (by X):")
    for i, idx in enumerate(left_indices):
        x, y = points[idx]
        print(f"  Point {idx+1}: ({x:.1f}, {y:.1f})")
    
    print(f"\nRight column points (by X):")
    for i, idx in enumerate(right_indices):
        x, y = points[idx]
        print(f"  Point {idx+1}: ({x:.1f}, {y:.1f})")
    
    # Step 2: Sort each column by Y (top to bottom)
    left_points = left_points[np.argsort(left_points[:, 1])]  # Sort by Y
    right_points = right_points[np.argsort(right_points[:, 1])]  # Sort by Y
    
    print(f"\nLeft column sorted top→bottom:")
    for i, (x, y) in enumerate(left_points):
        print(f"  Row {i+1}: ({x:.1f}, {y:.1f})")
    
    print(f"\nRight column sorted top→bottom:")
    for i, (x, y) in enumerate(right_points):
        print(f"  Row {i+1}: ({x:.1f}, {y:.1f})")
    
    # Step 3: Combine in column-first order
    reordered = np.vstack([
        left_points[0],   # Left column, top
        left_points[1],   # Left column, middle
        left_points[2],   # Left column, bottom
        right_points[0],  # Right column, top
        right_points[1],  # Right column, middle
        right_points[2]   # Right column, bottom
    ])
    
    # Map original indices to new order
    index_mapping = []
    for pt in reordered:
        # Find which original point this corresponds to
        for orig_idx, orig_pt in enumerate(points):
            if np.allclose(pt, orig_pt, atol=1.0):
                index_mapping.append(orig_idx + 1)  # 1-based indexing
                break
    
    print(f"\nOriginal point order → New order mapping:")
    for new_idx, orig_idx in enumerate(index_mapping):
        print(f"  New Point {new_idx+1} = Original Point {orig_idx}")
    
    return reordered, index_mapping

def main():
    # Load your points
    dev0_json = "../annotations/dev0_pts.json"
    dev3_json = "../annotations/dev3_pts.json"
    
    dev0_img = "../dataset/Dev0/Dev0_Image_w1920_h1200_fn1.jpg"
    dev3_img = "../dataset/Dev3/Dev3_Image_w1920_h1200_fn1.jpg"
    
    with open(dev0_json, 'r') as f:
        pts0 = np.array(json.load(f), dtype=np.float32)
    
    with open(dev3_json, 'r') as f:
        pts3 = np.array(json.load(f), dtype=np.float32)
    
    print("ORIGINAL POINTS (as annotated):")
    print("\nDev0:")
    for i, (x, y) in enumerate(pts0):
        print(f"  {i+1}. ({x:.1f}, {y:.1f})")
    
    print("\nDev3:")
    for i, (x, y) in enumerate(pts3):
        print(f"  {i+1}. ({x:.1f}, {y:.1f})")
    
    # Auto-detect and reorder
    pts0_reordered, mapping0 = auto_detect_grid_order(pts0, "Dev0")
    pts3_reordered, mapping3 = auto_detect_grid_order(pts3, "Dev3")
    
    print("\n" + "="*60)
    print("REORDERED POINTS (Correct Column-First Order)")
    print("="*60)
    
    print("\nDev0 (Column-First Order):")
    labels = ["Top-Left", "Middle-Left", "Bottom-Left", "Top-Right", "Middle-Right", "Bottom-Right"]
    for i, (label, (x, y)) in enumerate(zip(labels, pts0_reordered)):
        print(f"  {i+1}. {label}: ({x:.1f}, {y:.1f})")
    
    print("\nDev3 (Column-First Order):")
    for i, (label, (x, y)) in enumerate(zip(labels, pts3_reordered)):
        print(f"  {i+1}. {label}: ({x:.1f}, {y:.1f})")
    
    # Save reordered points
    dev0_reordered_json = "../annotations/dev0_pts_corrected.json"
    dev3_reordered_json = "../annotations/dev3_pts_corrected.json"
    
    with open(dev0_reordered_json, 'w') as f:
        json.dump(pts0_reordered.tolist(), f)
    
    with open(dev3_reordered_json, 'w') as f:
        json.dump(pts3_reordered.tolist(), f)
    
    print(f"\n✓ Saved corrected points to:")
    print(f"  {dev0_reordered_json}")
    print(f"  {dev3_reordered_json}")
    
    # Visualize
    print("\nVisualizing corrected points...")
    visualize_points(dev0_img, pts0_reordered, "Dev0 Corrected Points", 
                    "../annotations/dev0_corrected_visualization.jpg")
    visualize_points(dev3_img, pts3_reordered, "Dev3 Corrected Points",
                    "../annotations/dev3_corrected_visualization.jpg")
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Update your pipeline to use the corrected points:")
    print("   annotations/dev0_pts_corrected.json")
    print("   annotations/dev3_pts_corrected.json")
    print("\n2. OR modify load_annotation_or_annotate to auto-reorder:")
    print("""
def load_annotation_or_annotate(dev0_sample, dev3_sample, dev0_json=None, dev3_json=None, n_points=6):
    # [Load points as before...]
    
    # Auto-reorder points to column-first
    pts0 = auto_detect_grid_order(pts0)[0]
    pts3 = auto_detect_grid_order(pts3)[0]
    
    return pts0, pts3
""")
    
    # Verify the reordering makes sense
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    # Check that corresponding points have similar patterns
    dev0_left_col = pts0_reordered[:3]
    dev0_right_col = pts0_reordered[3:]
    
    dev3_left_col = pts3_reordered[:3]
    dev3_right_col = pts3_reordered[3:]
    
    print("\nLeft Column Y positions (should be similar pattern):")
    print(f"  Dev0: {dev0_left_col[:, 1]}")
    print(f"  Dev3: {dev3_left_col[:, 1]}")
    
    print("\nRight Column Y positions (should be similar pattern):")
    print(f"  Dev0: {dev0_right_col[:, 1]}")
    print(f"  Dev3: {dev3_right_col[:, 1]}")

if __name__ == "__main__":
    main()