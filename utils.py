# utils.py
import os, shutil, glob

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def sort_images(input_dir, output_dir):
    ensure_dir(output_dir)
    dev_lists = {f"Dev{i}": [] for i in range(4)}
    for fname in sorted(os.listdir(input_dir)):
        for i in range(4):
            if f"dev{i}_image" in fname.lower():
                dst_dir = os.path.join(output_dir, f"Dev{i}")
                ensure_dir(dst_dir)
                shutil.copy2(os.path.join(input_dir,fname), dst_dir)
                dev_lists[f"Dev{i}"].append(os.path.join(dst_dir,fname))
    return dev_lists

def load_dev_folders(base_dir):
    dev0 = glob.glob(os.path.join(base_dir,"**/Dev0"), recursive=True)
    dev3 = glob.glob(os.path.join(base_dir,"**/Dev3"), recursive=True)
    if not dev0 or not dev3:
        raise ValueError("Dev0 or Dev3 folder not found")
    dev_lists = {
        "Dev0": sorted(glob.glob(os.path.join(dev0[0],"*.jpg"))+glob.glob(os.path.join(dev0[0],"*.png"))),
        "Dev3": sorted(glob.glob(os.path.join(dev3[0],"*.jpg"))+glob.glob(os.path.join(dev3[0],"*.png")))
    }
    return dev_lists
