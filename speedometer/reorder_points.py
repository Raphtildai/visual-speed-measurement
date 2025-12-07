# reorder_points.py
import json
import numpy as np

def reorder_points_z_to_column_first(points):
    """Reorder Z-pattern points to column-first order."""
    # Sort by Y (top to bottom), then by X (left to right)
    sorted_by_y = points[np.argsort(points[:, 1])]
    
    # Split into top, middle, bottom rows
    top_row = sorted_by_y[:2]  # Two top points
    middle_row = sorted_by_y[2:4]  # Two middle points
    bottom_row = sorted_by_y[4:]  # Two bottom points
    
    # Sort each row by X (left to right)
    top_row = top_row[np.argsort(top_row[:, 0])]
    middle_row = middle_row[np.argsort(middle_row[:, 0])]
    bottom_row = bottom_row[np.argsort(bottom_row[:, 0])]
    
    # Create column-first order: left column top→bottom, then right column
    reordered = np.vstack([
        top_row[0],     # Top-Left
        middle_row[0],  # Middle-Left
        bottom_row[0],  # Bottom-Left
        top_row[1],     # Top-Right
        middle_row[1],  # Middle-Right
        bottom_row[1]   # Bottom-Right
    ])
    
    return reordered

def reorder_points_z_to_row_first(points):
    """Reorder Z-pattern points to row-first order."""
    # Sort by Y (top to bottom), then by X (left to right)
    sorted_by_y = points[np.argsort(points[:, 1])]
    
    # Split into top, middle, bottom rows
    top_row = sorted_by_y[:2]  # Two top points
    middle_row = sorted_by_y[2:4]  # Two middle points
    bottom_row = sorted_by_y[4:]  # Two bottom points
    
    # Sort each row by X (left to right)
    top_row = top_row[np.argsort(top_row[:, 0])]
    middle_row = middle_row[np.argsort(middle_row[:, 0])]
    bottom_row = bottom_row[np.argsort(bottom_row[:, 0])]
    
    # Create row-first order: top row left→right, middle row, bottom row
    reordered = np.vstack([
        top_row[0],     # Top-Left
        top_row[1],     # Top-Right
        middle_row[0],  # Middle-Left
        middle_row[1],  # Middle-Right
        bottom_row[0],  # Bottom-Left
        bottom_row[1]   # Bottom-Right
    ])
    
    return reordered

def main():
    # Load your current points
    dev0_json = "../annotations/dev0_pts.json"
    dev3_json = "../annotations/dev3_pts.json"
    
    with open(dev0_json, 'r') as f:
        pts0 = np.array(json.load(f), dtype=np.float32)
    
    with open(dev3_json, 'r') as f:
        pts3 = np.array(json.load(f), dtype=np.float32)
    
    print("Original Dev0 points:")
    for i, (x, y) in enumerate(pts0):
        print(f"  {i+1}. ({x:.1f}, {y:.1f})")
    
    print("\nOriginal Dev3 points:")
    for i, (x, y) in enumerate(pts3):
        print(f"  {i+1}. ({x:.1f}, {y:.1f})")
    
    # Reorder to column-first (recommended for your case)
    print("\n" + "="*60)
    print("REORDERING TO COLUMN-FIRST (Recommended)")
    print("="*60)
    
    pts0_reordered = reorder_points_z_to_column_first(pts0)
    pts3_reordered = reorder_points_z_to_column_first(pts3)
    
    print("\nReordered Dev0 points (column-first):")
    print("  1. Top-Left")
    print("  2. Middle-Left")
    print("  3. Bottom-Left")
    print("  4. Top-Right")
    print("  5. Middle-Right")
    print("  6. Bottom-Right")
    
    for i, (x, y) in enumerate(pts0_reordered):
        print(f"  {i+1}. ({x:.1f}, {y:.1f})")
    
    print("\nReordered Dev3 points (column-first):")
    for i, (x, y) in enumerate(pts3_reordered):
        print(f"  {i+1}. ({x:.1f}, {y:.1f})")
    
    # Save reordered points
    dev0_reordered_json = "../annotations/dev0_pts_column_first.json"
    dev3_reordered_json = "../annotations/dev3_pts_column_first.json"
    
    with open(dev0_reordered_json, 'w') as f:
        json.dump(pts0_reordered.tolist(), f)
    
    with open(dev3_reordered_json, 'w') as f:
        json.dump(pts3_reordered.tolist(), f)
    
    print(f"\n✓ Saved reordered points to:")
    print(f"  {dev0_reordered_json}")
    print(f"  {dev3_reordered_json}")
    
    # Also reorder to row-first for comparison
    print("\n" + "="*60)
    print("REORDERING TO ROW-FIRST (Alternative)")
    print("="*60)
    
    pts0_row_first = reorder_points_z_to_row_first(pts0)
    pts3_row_first = reorder_points_z_to_row_first(pts3)
    
    print("\nReordered Dev0 points (row-first):")
    print("  1. Top-Left")
    print("  2. Top-Right")
    print("  3. Middle-Left")
    print("  4. Middle-Right")
    print("  5. Bottom-Left")
    print("  6. Bottom-Right")
    
    for i, (x, y) in enumerate(pts0_row_first):
        print(f"  {i+1}. ({x:.1f}, {y:.1f})")
    
    dev0_row_first_json = "../annotations/dev0_pts_row_first.json"
    dev3_row_first_json = "../annotations/dev3_pts_row_first.json"
    
    with open(dev0_row_first_json, 'w') as f:
        json.dump(pts0_row_first.tolist(), f)
    
    with open(dev3_row_first_json, 'w') as f:
        json.dump(pts3_row_first.tolist(), f)
    
    print(f"\n✓ Saved row-first points to:")
    print(f"  {dev0_row_first_json}")
    print(f"  {dev3_row_first_json}")
    
    print("\n" + "="*60)
    print("INSTRUCTIONS:")
    print("="*60)
    print("1. Use column-first ordering (recommended for your setup)")
    print("2. Update your pipeline to use the reordered points:")
    print("   - Use dev0_pts_column_first.json")
    print("   - Use dev3_pts_column_first.json")
    print("3. OR re-annotate with the correct order using the improved tool")

if __name__ == "__main__":
    main()