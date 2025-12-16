# ransac_speed.py
import numpy as np
import logging

def ransac_translation_improved(flow, mask=None, n_iter=1000, sample_size=10, tol_px=2.0, min_inliers_frac=0.10):
    """
    RANSAC for translation, improved for robustness.
    - Uses mean of inliers for refined translation.
    - Requires a minimum fraction of inliers to return a non-zero translation.
    """
    H,W = flow.shape[:2]
    vecs = flow.reshape(-1,2)
    
    if mask is not None:
        mask_flat = mask.reshape(-1)
        inds = np.where(mask_flat)[0]
        if inds.size == 0:
            return np.array([0.0,0.0]), np.zeros((H,W), dtype=bool)
        sample_space = vecs[inds]
        global_inds = inds
    else:
        sample_space = vecs
        global_inds = np.arange(vecs.shape[0])

    N = sample_space.shape[0]
    if N == 0:
        return np.array([0.0,0.0]), np.zeros((H,W), dtype=bool)

    best_inliers_global_inds = None
    best_count = 0
    best_t = np.array([0.0,0.0])
    rng = np.random.default_rng()
    
    min_required_inliers = max(10, int(min_inliers_frac * N))

    for _ in range(n_iter):
        # Choose samples. Use all if N is small.
        if N <= sample_size:
            sample_idx = rng.choice(N, N, replace=False)
        else:
            sample_idx = rng.choice(N, sample_size, replace=False)
            
        sample = sample_space[sample_idx]
        
        # Estimate candidate translation using the median for robustness against extreme outliers
        t_cand = np.median(sample, axis=0) 
        
        # Calculate distance of all flow vectors to the candidate translation
        dists = np.linalg.norm(sample_space - t_cand, axis=1)
        inliers_local = dists < tol_px
        cnt = int(inliers_local.sum())
        
        if cnt > best_count:
            best_count = cnt
            best_t = t_cand
            best_inliers_global_inds = global_inds[inliers_local]

    # Check if the best model is reliable
    if best_count < min_required_inliers:
        logging.debug(f"RANSAC failed: Only {best_count} inliers found (min required: {min_required_inliers})")
        return np.array([0.0,0.0]), np.zeros((H,W), dtype=bool)
        
    # Refined model: Calculate the mean of all found inliers
    refined = vecs[best_inliers_global_inds].mean(axis=0)

    # Create the final inlier mask
    inlier_mask = np.zeros(H*W, dtype=bool)
    inlier_mask[best_inliers_global_inds] = True
    inlier_mask = inlier_mask.reshape(H,W)

    return refined, inlier_mask