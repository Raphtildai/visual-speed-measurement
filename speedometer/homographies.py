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
    h0, w0 = img0.shape[:2]
    h3, w3 = img3.shape[:2]

    # Prefer direct homography if points available
    if pts0 is not None and pts3 is not None and len(pts0) >= 4:
        H_stitch, _ = cv2.findHomography(pts3, pts0, cv2.RANSAC, 1.0)
        if H_stitch is not None:
            H_stitch = H_stitch
        else:
            H_stitch = H3_to_0
    else:
        H_stitch = H3_to_0

    # Compute corners to find proper output size (no black triangles)
    corners3 = np.float32([[0,0], [w3,0], [w3,h3], [0,h3]]).reshape(-1,1,2)
    corners3_warped = cv2.perspectiveTransform(corners3, H_stitch)

    all_corners = np.concatenate((np.float32([[0,0],[w0,0],[w0,h0],[0,h0]]).reshape(-1,1,2), corners3_warped))
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translate = np.array([[1,0,-x_min], [0,1,-y_min], [0,0,1]])
    H_trans = translate @ H_stitch

    out_w, out_h = x_max - x_min, y_max - y_min
    warped_img3 = cv2.warpPerspective(img3, H_trans, (out_w, out_h))

    # Translate img0
    H_identity_trans = translate @ np.eye(3)
    warped_img0 = cv2.warpPerspective(img0, H_identity_trans, (out_w, out_h))

    # Simple alpha blend in overlap
    mask3 = (warped_img3 > 0).any(axis=2)
    result = warped_img0.copy()
    overlap = mask3 & ((warped_img0 > 0).any(axis=2))
    if overlap.any():
        result[overlap] = (warped_img0[overlap] * 0.6 + warped_img3[overlap] * 0.4).astype(np.uint8)
    result[mask3 & ~((warped_img0 > 0).any(axis=2))] = warped_img3[mask3 & ~((warped_img0 > 0).any(axis=2))]

    return result


def warp_to_bev(img, H_img_to_bev, out_size):
    return cv2.warpPerspective(img, H_img_to_bev, out_size, flags=cv2.INTER_LINEAR)
