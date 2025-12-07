# check_annotations.py
import cv2
import numpy as np
import json
import os

def load_points(json_file):
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            return np.array(json.load(f), dtype=np.float32)
    return None

def check_correspondences():
    """Check if annotations make sense for stitching."""
    
    # Load points
    pts0 = load_points('annotations/dev0_pts.json')
    pts3 = load_points('annotations/dev3_pts.json')
    
    if pts0 is None or pts3 is None:
        print("Annotations not found. Please annotate first.")
        return
    
    print("="*60)
    print("ANNOTATION CORRESPONDENCE CHECK")
    print("="*60)
    
    print(f"\nDev0 (Left Camera) - should be on LEFT side of image:")
    print(f"  X range: {pts0[:, 0].min():.1f} to {pts0[:, 0].max():.1f}")
    print(f"  Y range: {pts0[:, 1].min():.1f} to {pts0[:, 1].max():.1f}")
    
    print(f"\nDev3 (Right Camera) - should be on RIGHT side of image:")
    print(f"  X range: {pts3[:, 0].min():.1f} to {pts3[:, 0].max():.1f}")
    print(f"  Y range: {pts3[:, 1].min():.1f} to {pts3[:, 1].max():.1f}")
    
    # Check if points make sense for stitching
    print(f"\nANALYSIS:")
    
    # Dev0 should have points on the LEFT (X < 960 for 1920x1200 image)
    dev0_left_ratio = np.sum(pts0[:, 0] < 960) / len(pts0)
    print(f"  Dev0 points on left side: {dev0_left_ratio*100:.1f}%")
    
    # Dev3 should have points on the RIGHT (X > 960 for 1920x1200 image)
    dev3_right_ratio = np.sum(pts3[:, 0] > 960) / len(pts3)
    print(f"  Dev3 points on right side: {dev3_right_ratio*100:.1f}%")
    
    # Try to compute homography
    if len(pts0) >= 4 and len(pts3) >= 4:
        H, mask = cv2.findHomography(pts3, pts0, cv2.RANSAC, 5.0)
        
        if H is not None:
            # Check what this homography does to corners
            corners = np.array([[0, 0], [1920, 0], [1920, 1200], [0, 1200]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H).reshape(-1, 2)
            
            print(f"\nHomography transforms corners to:")
            for i, (orig, trans) in enumerate(zip(corners, transformed)):
                print(f"  Corner {i}: {orig} -> {trans.astype(int)}")
            
            # Check if transformation is reasonable
            avg_x = np.mean(transformed[:, 0])
            print(f"\nAverage X after transform: {avg_x:.1f}")
            
            if 500 < avg_x < 2500:  # Reasonable range
                print(f"  ✓ Homography looks reasonable!")
            elif avg_x < 0:
                print(f"  ✗ Homography moves image too far LEFT")
            else:
                print(f"  ✗ Homography moves image too far RIGHT ({avg_x:.1f} is extreme)")
        else:
            print(f"\n✗ Failed to compute homography")
    else:
        print(f"\n✗ Need at least 4 points in each image")

if __name__ == "__main__":
    check_correspondences()