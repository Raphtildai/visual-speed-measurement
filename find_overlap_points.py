# find_overlap_points.py
import cv2
import numpy as np
import json
import os

def select_overlap_points_manually(img0_path, img3_path, n_points=6):
    """Manually select overlapping points in both images."""
    
    img0 = cv2.imread(img0_path)
    img3 = cv2.imread(img3_path)
    
    # Display images side by side
    h0, w0 = img0.shape[:2]
    h3, w3 = img3.shape[:2]
    
    # Resize for display
    scale = 0.5
    display_h = int(min(h0, h3) * scale)
    img0_display = cv2.resize(img0, (int(w0 * scale), display_h))
    img3_display = cv2.resize(img3, (int(w3 * scale), display_h))
    
    combined = np.hstack([img0_display, img3_display])
    
    print("="*60)
    print("MANUAL OVERLAP POINT SELECTION")
    print("="*60)
    print("\nINSTRUCTIONS:")
    print("1. Look for points in the DISTANCE/FAR FIELD")
    print("2. Choose points that are VISIBLE IN BOTH IMAGES")
    print("3. Good candidates:")
    print("   - Lane markings in the distance")
    print("   - Horizon features")
    print("   - Road signs far away")
    print("   - Distinct pavement patterns")
    print("\nClick points in this order in BOTH images.")
    print("First click in LEFT image (Dev0), then RIGHT image (Dev3).")
    
    points0 = []
    points3 = []
    
    def click_event(event, x, y, flags, param):
        nonlocal points0, points3, combined
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Determine which image was clicked
            if x < img0_display.shape[1]:
                # Clicked in Dev0 image
                actual_x = int(x / scale)
                actual_y = int(y / scale)
                points0.append([actual_x, actual_y])
                print(f"Dev0 Point {len(points0)}: ({actual_x}, {actual_y})")
                
                # Draw on display
                cv2.circle(combined, (x, y), 8, (0, 255, 0), -1)
                cv2.putText(combined, str(len(points0)), (x+10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                # Clicked in Dev3 image
                adjusted_x = x - img0_display.shape[1]
                actual_x = int(adjusted_x / scale)
                actual_y = int(y / scale)
                points3.append([actual_x, actual_y])
                print(f"Dev3 Point {len(points3)}: ({actual_x}, {actual_y})")
                
                # Draw on display
                cv2.circle(combined, (x, y), 8, (0, 255, 0), -1)
                cv2.putText(combined, str(len(points3)), (x+10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow("Select Overlap Points", combined)
    
    cv2.namedWindow("Select Overlap Points", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Overlap Points", 1600, 600)
    cv2.setMouseCallback("Select Overlap Points", click_event)
    
    # Add instructional text to image
    cv2.putText(combined, "LEFT: Dev0 Camera", (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(combined, "RIGHT: Dev3 Camera", (img0_display.shape[1] + 50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(combined, "Click corresponding points in BOTH images", (50, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Select Overlap Points", combined)
    
    print("\nClick points. Press 'q' when done, 'r' to reset.")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            points0 = []
            points3 = []
            combined = np.hstack([img0_display, img3_display])
            cv2.putText(combined, "LEFT: Dev0 Camera", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(combined, "RIGHT: Dev3 Camera", (img0_display.shape[1] + 50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(combined, "Click corresponding points in BOTH images", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Select Overlap Points", combined)
            print("\nReset. Click points again.")
    
    cv2.destroyAllWindows()
    
    # Convert to numpy arrays
    if len(points0) == len(points3) and len(points0) >= 4:
        pts0_array = np.array(points0, dtype=np.float32)
        pts3_array = np.array(points3, dtype=np.float32)
        
        print(f"\nCollected {len(points0)} point pairs")
        return pts0_array, pts3_array
    
    return None, None

if __name__ == "__main__":
    img0_path = "dataset/Dev0/Dev0_Image_w1920_h1200_fn1.jpg"
    img3_path = "dataset/Dev3/Dev3_Image_w1920_h1200_fn1.jpg"
    
    pts0, pts3 = select_overlap_points_manually(img0_path, img3_path, n_points=6)
    
    if pts0 is not None and pts3 is not None:
        # Save points
        with open('annotations/dev0_pts_overlap_manual.json', 'w') as f:
            json.dump(pts0.tolist(), f)
        with open('annotations/dev3_pts_overlap_manual.json', 'w') as f:
            json.dump(pts3.tolist(), f)
        
        print(f"\n✓ Points saved!")
        print(f"  Dev0: annotations/dev0_pts_overlap_manual.json")
        print(f"  Dev3: annotations/dev3_pts_overlap_manual.json")
        
        # Test homography
        H, mask = cv2.findHomography(pts3, pts0, cv2.RANSAC, 5.0)
        if H is not None:
            corners = np.array([[0, 0], [1920, 0], [1920, 1200], [0, 1200]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H).reshape(-1, 2)
            
            print(f"\nHomography test:")
            for i, (orig, trans) in enumerate(zip(corners, transformed)):
                print(f"  Corner {i}: {orig} -> {trans.astype(int)}")
            
            avg_x = np.mean(transformed[:, 0])
            print(f"\nAverage X: {avg_x:.1f}")
            
            if 500 < avg_x < 2500:
                print("✓ Homography looks good for stitching!")
            else:
                print("⚠ Homography might be extreme")
                
            # Show the homography matrix
            print(f"\nHomography matrix:")
            print(H)
    else:
        print("\n✗ Need at least 4 corresponding points")