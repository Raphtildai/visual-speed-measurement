This is a comprehensive README file detailing the setup, configuration, execution, and fine-tuning of the camera-based speedometer project.

# 🚗 Camera-Based Speedometer Project README

This project implements a camera-based speedometer by calculating the ego-motion of a vehicle using homographies, Bird's-Eye View (BEV) transformation, dense optical flow, and RANSAC-based robust estimation.

-----

## 1\. Project Setup and Dependencies

### 1.1 Virtual Environment

Ensure you are running the project within a virtual environment.

```bash
# Create a virtual environment (if you haven't already)
python -m venv venv

# Activate the virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate
```

### 1.2 Required Libraries

This project relies on **OpenCV** (`cv2`), **NumPy**, **Tqdm**, and **ImageIO**.

```bash
# Install required libraries
pip install opencv-python numpy tqdm imageio
```

### 1.3 Data Structure

Ensure your image data is structured correctly under a `dataset` folder:

```
visual-speed-measurement/
├── dataset/
│   ├── Dev0/
│   │   ├── Dev0_Image_w1920_h1200_fn1.jpg
│   │   ├── Dev0_Image_w1920_h1200_fn2.jpg
│   │   └── ...
│   └── Dev3/
│       ├── Dev3_Image_w1920_h1200_fn1.jpg
│       └── ...
├── annotations/
│   ├── dev0_pts.json  # Annotated marker points
│   └── dev3_pts.json
└── main.py
└── speedometer/
    ├── pipeline.py
    └── ...
```

-----

## 2\. Configuration and Execution Modes (`main.py`)

The primary execution script, `main.py`, contains a `RUN_MODE` switch to easily control the processing workload.

### 2.1 Parameter Tuning

The following parameters are fine-tuned for optimal performance and clarity:

| Parameter | Value | Purpose |
| :--- | :--- | :--- |
| `TOTAL_FRAMES` | `3405` | The full count of frames in the dataset. |
| `SCALE_PX_PER_M` | `150.0` | **BEV Clarity Enhancement:** Increased pixel density from `120.0` to improve BEV image detail. |
| `flow_params` | `{...}` | Default Farneback optical flow parameters for robust motion estimation. |

### 2.2 Selecting the `RUN_MODE`

Modify the `RUN_MODE` variable in `main.py` before running the script.

| `RUN_MODE` | Frame Range | Purpose | Output File |
| :--- | :--- | :--- | :--- |
| **`"FULL"`** | 0 to 3404 | **Final Submission Run.** Processes the entire dataset. | `result_speedometer_full.mp4` |
| **`"HALF1"`** | 0 to 1701 | First half of the frames (for batch processing). | `result_speedometer_part1.mp4` |
| **`"HALF2"`** | 1702 to 3404 | Second half of the frames (for batch processing). | `result_speedometer_part2.mp4` |
| **`"CHUNK"`** | 0 to 99 | **Fast Testing/Debugging.** Runs only the first 100 frames. | `result_speedometer_test_chunk.mp4` |

### 2.3 Running the Script

Execute the pipeline using the activated environment:

```bash
python main.py
```

-----

## 3\. Core Code Functionality (pipeline.py)

The `pipeline.py` has been updated with critical fixes to ensure stability and accuracy.

### 3.1 Chronological Sorting Fix

The file naming convention (e.g., `fn1`, `fn10`) requires **natural sorting**. This is implemented by adding the `re` module and the `natural_sort_key` function, ensuring frames are processed in the correct order (`1, 2, 3, ... 9, 10, 11...`).

### 3.2 Optical Flow Stability

The pipeline now uses a **low-resolution copy of the BEV image** (`FLOW_DOWNSCALE_FACTOR = 0.5`) for the `cv2.calcOpticalFlowFarneback` step. This prevents memory errors on large BEV images and maintains the size consistency required by the optical flow function. The resulting flow vector is correctly scaled back up for speed calculation.

### 3.3 Enhanced Visualization (`compose_visual_frame_v2`)

The visualization function has been completely redesigned to meet the project's requirements for clarity:

  * **Layout:** Three-panel horizontal layout: **[BEV LEFT (Dev0) | STITCHED MAIN VIEW | BEV RIGHT (Dev3)]**.
  * **Dominant View:** The **Stitched Main View** is dominant (`1080px` wide) while the two BEV panels are smaller (`540px` wide), focusing attention on the road.
  * **Speedometer:** The calculated **speed in km/h** is placed directly under the central Stitched Main View.
  * **Motion Vector:** The dominant RANSAC-derived motion vector is visualized in the scoreboard area below the BEV LEFT panel.

-----

## 4\. Merging Videos (Batch Processing)

If you run the pipeline using the **`"HALF1"`** and **`"HALF2"`** modes, you must use **FFmpeg** to concatenate the two output video files.

1.  **Run Part 1** and **Run Part 2** (as described in Section 2).

2.  **Create a list file** (`results/file_list.txt`):

    ```
    file 'result_speedometer_part1.mp4'
    file 'result_speedometer_part2.mp4'
    ```

3.  **Execute the FFmpeg command** in your terminal to quickly merge the files without re-encoding:

    ```bash
    ffmpeg -f concat -i results/file_list.txt -c copy results/final_speedometer_full.mp4
    ```