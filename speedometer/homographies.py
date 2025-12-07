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


def stitch_two_images(img0, img3, H3_to_0):
    h0,w0 = img0.shape[:2]
    h3,w3 = img3.shape[:2]

    corners3 = np.array([[0,0],[w3,0],[w3,h3],[0,h3]], dtype=np.float32).reshape(-1,1,2)
    warped = cv2.perspectiveTransform(corners3, H3_to_0).reshape(-1,2)
    corners0 = np.array([[0,0],[w0,0],[w0,h0],[0,h0]], dtype=np.float32)

    all_pts = np.vstack([warped, corners0])
    x_min, y_min = np.floor(all_pts.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_pts.max(axis=0)).astype(int)
    tx = -x_min; ty = -y_min
    H_trans = np.array([[1,0,tx],[0,1,ty],[0,0,1]], dtype=np.float64)
    out_w, out_h = x_max-x_min, y_max-y_min

    canvas = cv2.warpPerspective(img3, H_trans @ H3_to_0, (out_w, out_h))
    canvas[ty:ty+h0, tx:tx+w0] = img0
    return canvas


def warp_to_bev(img, H_img_to_bev, out_size):
    return cv2.warpPerspective(img, H_img_to_bev, out_size, flags=cv2.INTER_LINEAR)
