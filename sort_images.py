import os
import shutil
import argparse
import re

def sort_images(input_dir, output_dir):
    # Create output folders
    for dev in ["Dev0", "Dev1", "Dev2", "Dev3"]:
        os.makedirs(os.path.join(output_dir, dev), exist_ok=True)

    # Regular expression to extract Dev number from filename
    pattern = re.compile(r"(Dev[0-9])_Image.*\.jpg$", re.IGNORECASE)

    files = os.listdir(input_dir)
    total = 0
    sorted = 0

    for filename in files:
        total += 1
        match = pattern.search(filename)
        if not match:
            print(f"Skipping non-matching file: {filename}")
            continue

        dev_folder = match.group(1)  # e.g. "Dev0", "Dev1", etc.
        src = os.path.join(input_dir, filename)
        dst = os.path.join(output_dir, dev_folder, filename)

        shutil.copy2(src, dst)
        sorted += 1

    print(f"\nDone! {sorted}/{total} files sorted.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sort device images into Dev0-Dev3 folders.")
    parser.add_argument("--input", required=True, help="Folder containing all images mixed together.")
    parser.add_argument("--output", required=True, help="Output folder for sorted dataset.")
    args = parser.parse_args()

    sort_images(args.input, args.output)

# Usage:
# python sort_images.py --input path/to/mixed_images --output path/to/sorted_dataset
