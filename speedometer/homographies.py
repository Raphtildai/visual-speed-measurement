# homographies.py
import cv2
import numpy as np

def load_points(json_path):
    import json
    pts = json.load(open(json_path))
    return np.array(pts, dtype=np.float32)


def build_homography(img_pts, world_pts_m, scale_px_per_m=120.0):
    dst_pts = np.array(world_pts_m, dtype=np.float32) * scale_px_per_m
    H, mask = cv2.findHomography(img_pts, dst_pts, cv2.RANSAC)
    return H, mask


def stitch_two_images(img0, img3, H3_to_0, pts0=None, pts3=None):
    """Improved stitching that can use direct point correspondences."""
    h0, w0 = img0.shape[:2]
    h3, w3 = img3.shape[:2]
    
    # If we have direct point correspondences, compute stitching homography
    if pts0 is not None and pts3 is not None and len(pts0) >= 4:
        # Compute direct homography from dev3 to dev0
        H_direct, mask = cv2.findHomography(pts3, pts0, cv2.RANSAC, 5.0)
        
        if H_direct is not None:
            print(f"Using direct homography for stitching")
            H_stitch = H_direct
        else:
            print("Direct homography failed, using provided H3_to_0")
            H_stitch = H3_to_0
    else:
        H_stitch = H3_to_0
    
    if H_stitch is None:
        print("WARNING: No stitching homography available, using side-by-side")
        return np.hstack([img0, img3])
    
    # Check if homography is reasonable
    H_identity = np.eye(3)
    diff_from_identity = np.linalg.norm(H_stitch - H_identity)
    
    if diff_from_identity < 1.0:
        print(f"WARNING: Stitching homography is close to identity (diff={diff_from_identity:.4f})")
        print("This suggests cameras aren't properly aligned for stitching")
    
    try:
        # Warp img3 to img0's perspective
        warped_img3 = cv2.warpPerspective(img3, H_stitch, (w0 * 2, h0))
        
        # Create output canvas (wider to accommodate both views)
        result = np.zeros((h0, w0 * 2, 3), dtype=np.uint8)
        result[:, :w0] = img0  # Place img0 on left
        
        # Blend warped img3
        mask_warped = (warped_img3 > 0).any(axis=2)
        
        # For overlapping regions, blend
        overlap_mask = mask_warped[:, :w0]
        if np.any(overlap_mask):
            result[overlap_mask] = cv2.addWeighted(
                result[overlap_mask], 0.5, 
                warped_img3[overlap_mask], 0.5, 0
            )
        
        # For non-overlapping regions of warped image, just copy
        if w0 * 2 > w0:
            result[:, w0:] = warped_img3[:, w0:]
        
        return result
        
    except Exception as e:
        print(f"Error in stitching: {e}")
        return np.hstack([img0, img3])


def warp_to_bev(img, H_img_to_bev, out_size):
    return cv2.warpPerspective(img, H_img_to_bev, out_size, flags=cv2.INTER_LINEAR)
