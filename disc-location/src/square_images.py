import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

import cv2
import numpy as np


def square_image(img, threshold=10, background_color=(0, 0, 0), final_size=1920):
    # Convert the input image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold
    _, mask = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)

    # Find connected components in the binary mask.
    # 'stats' is a matrix where each row gives informations about each component
    num_components, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_components <= 1:
        cropped = img  # there is only background (no foreground regions found)
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]  # Skip label 0 (background)
        largest = 1 + int(np.argmax(areas))  # Add 1 because we skipped label 0

        # Get bounding box coordinates of the largest component.
        x = stats[largest, cv2.CC_STAT_LEFT]
        y = stats[largest, cv2.CC_STAT_TOP]
        w = stats[largest, cv2.CC_STAT_WIDTH]
        h = stats[largest, cv2.CC_STAT_HEIGHT]

        # Crop the image to this bounding box.
        cropped = img[y : y + h, x : x + w]

    # Compute how much padding is needed on each side to make the image square.
    h, w = cropped.shape[:2]
    size = max(h, w)
    px = (size - w) // 2
    py = (size - h) // 2

    # Add constant padding (background color) to make the image square
    squared = cv2.copyMakeBorder(
        cropped,
        py,
        py,
        px,
        px,
        borderType=cv2.BORDER_CONSTANT,
        value=background_color,
    )

    # Resize the final square image to a fixed size
    return cv2.resize(squared, (final_size, final_size), interpolation=cv2.INTER_AREA)


def process_single_image(src, output_dir):
    """Process a single image and return processing info."""
    try:
        img = cv2.imread(src, cv2.IMREAD_COLOR)
        if img is None:
            return {
                "file": os.path.basename(src),
                "status": "error",
                "message": "Could not read image",
            }

        dst = os.path.join(output_dir, os.path.basename(src))

        t0 = time.perf_counter()

        # Apply square image processing
        squared_img = square_image(img)
        cv2.imwrite(dst, squared_img)

        t1 = time.perf_counter()

        return {
            "file": os.path.basename(dst),
            "status": "success",
            "time_ms": (t1 - t0) * 1000,
        }

    except Exception as e:
        return {"file": os.path.basename(src), "status": "error", "message": str(e)}


def main(input_dir, output_dir, max_workers=None):
    """Process images in input_dir and save squared versions to output_dir."""
    files = sorted(glob(os.path.join(input_dir, "*")))
    if not files:
        print("No images found in", input_dir)
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing {len(files)} images using {max_workers or 'auto'} workers...")

    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_single_image, src, output_dir): src for src in files
        }

        completed = 0
        for future in as_completed(future_to_file):
            result = future.result()
            completed += 1

            if result["status"] == "success":
                print(
                    f"[{completed}/{len(files)}] ✅ Processed {result['file']} in {result['time_ms']:.1f} ms"
                )
            else:
                print(
                    f"[{completed}/{len(files)}] ⚠️ Failed to process {result['file']}: {result['message']}"
                )

    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"\nCompleted processing {len(files)} images in {total_time:.2f} seconds")
    print(f"Average time per image: {total_time / len(files):.2f} seconds")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=(
            "Process fundus photographs by converting them to square images "
            "with consistent dimensions"
        )
    )
    p.add_argument(
        "--input_dir",
        "-i",
        required=True,
        help="Path to the directory containing input fundus images.",
    )
    p.add_argument(
        "--output_dir",
        "-o",
        required=True,
        help="Path to the directory where squared images will be saved. Will be created if it doesn't exist.",
    )
    p.add_argument(
        "--max_workers",
        "-w",
        type=int,
        default=os.cpu_count(),
        help="Maximum number of worker threads. If not specified, uses the number of CPUs.",
    )

    args = p.parse_args()

    main(
        args.input_dir,
        args.output_dir,
        args.max_workers,
    )
