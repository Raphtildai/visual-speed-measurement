# bev_and_flow.py
import cv2
import numpy as np

def dense_flow_farneback(prev_gray, gray, params=None):
    if params is None:
        params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **params)
    return flow
