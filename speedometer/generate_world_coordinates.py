# generate_world_coordinates.py
import json
import numpy as np

def generate_world_coordinates_column_strategy():
    """Generate world coordinates for column strategy (2×3 grid, 3m spacing)."""
    world_coords = []
    
    # Column strategy: Left column top→bottom, then Right column top→bottom
    # With 3m spacing, total height = 6m (3 rows × 3m), width = 3m (2 cols × 3m)
    
    # Left column (x=0)
    world_coords.append([0.0, 6.0, 0.0])  # Left-Top
    world_coords.append([0.0, 3.0, 0.0])  # Left-Middle  
    world_coords.append([0.0, 0.0, 0.0])  # Left-Bottom
    
    # Right column (x=3m)
    world_coords.append([3.0, 6.0, 0.0])  # Right-Top
    world_coords.append([3.0, 3.0, 0.0])  # Right-Middle
    world_coords.append([3.0, 0.0, 0.0])  # Right-Bottom
    
    return np.array(world_coords, dtype=np.float32)

def generate_world_coordinates_row_strategy():
    """Generate world coordinates for row strategy (3×2 grid, 3m spacing)."""
    world_coords = []
    
    # Row strategy: Top row left→right, then Bottom row left→right
    # With 3m spacing, total width = 6m (3 cols × 3m), height = 3m (2 rows × 3m)
    
    # Top row (y=3m)
    world_coords.append([0.0, 3.0, 0.0])  # Top-Left
    world_coords.append([3.0, 3.0, 0.0])  # Top-Middle
    world_coords.append([6.0, 3.0, 0.0])  # Top-Right
    
    # Bottom row (y=0m)
    world_coords.append([0.0, 0.0, 0.0])  # Bottom-Left
    world_coords.append([3.0, 0.0, 0.0])  # Bottom-Middle
    world_coords.append([6.0, 0.0, 0.0])  # Bottom-Right
    
    return np.array(world_coords, dtype=np.float32)

def create_complete_json(strategy="column"):
    """Create complete JSON with all metadata."""
    
    # Your pixel coordinates (from your annotation output)
    pixel_coords_dev0 = np.array([
        [293.0, 642.6],
        [257.2, 667.4],
        [190.9, 706.1],
        [559.4, 637.1],
        [571.8, 666.1],
        [605.0, 700.6]
    ], dtype=np.float32)
    
    pixel_coords_dev3 = np.array([
        [1251.9, 689.0],
        [1223.8, 714.5],
        [1177.9, 745.1],
        [1505.5, 694.1],
        [1525.9, 720.9],
        [1564.1, 763.0]
    ], dtype=np.float32)
    
    point_names_column = ["Left-Top", "Left-Middle", "Left-Bottom", 
                         "Right-Top", "Right-Middle", "Right-Bottom"]
    
    point_names_row = ["Top-Left", "Top-Middle", "Top-Right",
                      "Bottom-Left", "Bottom-Middle", "Bottom-Right"]
    
    if strategy == "column":
        world_coords = generate_world_coordinates_column_strategy()
        point_names = point_names_column
        grid_dimensions = {"width_meters": 3.0, "height_meters": 6.0, "columns": 2, "rows": 3}
        description = "2 columns × 3 rows grid with 3m spacing"
    else:  # row strategy
        world_coords = generate_world_coordinates_row_strategy()
        point_names = point_names_row
        grid_dimensions = {"width_meters": 6.0, "height_meters": 3.0, "columns": 3, "rows": 2}
        description = "3 columns × 2 rows grid with 3m spacing"
    
    # Build the JSON structure
    data = {
        "strategy": strategy,
        "description": description,
        "units": "meters",
        "spacing": 3.0,
        "points": [],
        "grid_dimensions": grid_dimensions,
        "coordinate_system": {
            "x_axis": "Across road (left to right)",
            "y_axis": "Along road (bottom to top, away from cameras)",
            "z_axis": "Upwards from ground",
            "origin": "Bottom-left corner at ground level"
        },
        "pixel_coordinates_source": "From your latest annotation output"
    }
    
    # Add each point with all its data
    for i in range(6):
        point_data = {
            "name": point_names[i],
            "world_coordinates": world_coords[i].tolist(),
            "pixel_coordinates_dev0": pixel_coords_dev0[i].tolist(),
            "pixel_coordinates_dev3": pixel_coords_dev3[i].tolist(),
            "annotation_order": i + 1
        }
        
        # Add descriptions based on position
        if strategy == "column":
            if i < 3:
                point_data["description"] = f"Left column, position {['top', 'middle', 'bottom'][i]}"
            else:
                point_data["description"] = f"Right column, position {['top', 'middle', 'bottom'][i-3]}"
        else:  # row strategy
            if i < 3:
                point_data["description"] = f"Top row, position {['left', 'middle', 'right'][i]}"
            else:
                point_data["description"] = f"Bottom row, position {['left', 'middle', 'right'][i-3]}"
        
        data["points"].append(point_data)
    
    return data

def save_world_coordinates():
    """Save both strategies as JSON files."""
    
    # Create directory if it doesn't exist
    import os
    os.makedirs("../world_coordinates", exist_ok=True)
    
    # Save column strategy
    column_data = create_complete_json("column")
    with open("../world_coordinates/world_coordinates_column.json", "w") as f:
        json.dump(column_data, f, indent=2)
    print("✓ Saved column strategy world coordinates to ../world_coordinates/world_coordinates_column.json")
    
    # Save row strategy  
    row_data = create_complete_json("row")
    with open("../world_coordinates/world_coordinates_row.json", "w") as f:
        json.dump(row_data, f, indent=2)
    print("✓ Saved row strategy world coordinates to ../world_coordinates/world_coordinates_row.json")
    
    # Also save simplified versions for OpenCV use
    save_simplified_versions()

def save_simplified_versions():
    """Save simplified numpy arrays for direct use with OpenCV."""
    
    # Column strategy
    world_coords_column = generate_world_coordinates_column_strategy()
    np.save("../world_coordinates/world_coords_column.npy", world_coords_column)
    np.savetxt("../world_coordinates/world_coords_column.txt", world_coords_column)
    
    # Row strategy
    world_coords_row = generate_world_coordinates_row_strategy()
    np.save("../world_coordinates/world_coords_row.npy", world_coords_row)
    np.savetxt("../world_coordinates/world_coords_row.txt", world_coords_row)
    
    print("✓ Saved simplified numpy arrays for both strategies")
    print(f"  Column shape: {world_coords_column.shape}")
    print(f"  Row shape: {world_coords_row.shape}")

def test_homography_with_world_coords():
    """Test homography with your pixel coordinates and generated world coordinates."""
    import cv2
    
    # Load your pixel coordinates
    pts0 = np.array([
        [293.0, 642.6],
        [257.2, 667.4],
        [190.9, 706.1],
        [559.4, 637.1],
        [571.8, 666.1],
        [605.0, 700.6]
    ], dtype=np.float32)
    
    pts3 = np.array([
        [1251.9, 689.0],
        [1223.8, 714.5],
        [1177.9, 745.1],
        [1505.5, 694.1],
        [1525.9, 720.9],
        [1564.1, 763.0]
    ], dtype=np.float32)
    
    # Generate world coordinates
    world_column = generate_world_coordinates_column_strategy()
    world_row = generate_world_coordinates_row_strategy()
    
    print("\n" + "="*70)
    print("HOMOGRAPHY TESTS WITH WORLD COORDINATES")
    print("="*70)
    
    # Test Dev0 to world (column strategy)
    H0_column, mask0_column = cv2.findHomography(pts0, world_column[:, :2], cv2.RANSAC, 5.0)
    print(f"\nDev0 → World (Column strategy): {np.sum(mask0_column)}/6 inliers")
    
    # Test Dev3 to world (column strategy)
    H3_column, mask3_column = cv2.findHomography(pts3, world_column[:, :2], cv2.RANSAC, 5.0)
    print(f"Dev3 → World (Column strategy): {np.sum(mask3_column)}/6 inliers")
    
    # Test Dev0 to world (row strategy)
    H0_row, mask0_row = cv2.findHomography(pts0, world_row[:, :2], cv2.RANSAC, 5.0)
    print(f"\nDev0 → World (Row strategy): {np.sum(mask0_row)}/6 inliers")
    
    # Test Dev3 to world (row strategy)
    H3_row, mask3_row = cv2.findHomography(pts3, world_row[:, :2], cv2.RANSAC, 5.0)
    print(f"Dev3 → World (Row strategy): {np.sum(mask3_row)}/6 inliers")
    
    return H0_column, H3_column, H0_row, H3_row

if __name__ == "__main__":
    print("\n" + "="*70)
    print("GENERATING WORLD COORDINATES")
    print("="*70)
    print("\nBased on your annotation with 3m spacing between markers")
    
    # Save all coordinate files
    save_world_coordinates()
    
    # Test homography
    test_homography_with_world_coords()
    
    print("\n" + "="*70)
    print("FILES GENERATED:")
    print("="*70)
    print("1. ../world_coordinates/world_coordinates_column.json")
    print("   - Complete metadata for column strategy (2×3 grid)")
    print("2. ../world_coordinates/world_coordinates_row.json")
    print("   - Complete metadata for row strategy (3×2 grid)")
    print("3. ../world_coordinates/world_coords_column.npy/.txt")
    print("   - Simple numpy array for column strategy")
    print("4. ../world_coordinates/world_coords_row.npy/.txt")
    print("   - Simple numpy array for row strategy")