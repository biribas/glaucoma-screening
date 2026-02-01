import argparse
import os
import time
from glob import glob

import cv2

from utils import (
    create_onnx_model,
    crop_disc,
    get_binary_mask,
    make_predict_fn,
    run_inference,
)


def main(input_dir, output_dir, model_path):
    files = sorted(glob(os.path.join(input_dir, "*")))
    if not files:
        print("No images found in", input_dir)
        return

    _, ext = os.path.splitext(model_path)
    if ext != ".onnx":
        print(f"{ext} is not a valid extension for model weights")
        return

    model = create_onnx_model(model_path)
    predict = make_predict_fn(model)
    input_shape = model.get_inputs()[0].shape[1:3]

    found = 0
    not_found = 0
    os.makedirs(output_dir, exist_ok=True)
    for src in files:
        src_basename = os.path.basename(src)
        img = cv2.imread(src, cv2.IMREAD_COLOR)
        if img is None:
            print(f"❌ Could not read {src}")
            continue

        dst = os.path.join(output_dir, src_basename)

        t0 = time.perf_counter()

        prob_map = run_inference(img, predict, input_shape)
        mask = get_binary_mask(prob_map)
        if mask is None:
            print(f"❌ No optic disc found in {src_basename}")
            not_found += 1
            continue
        cropped = crop_disc(img, mask, input_shape)
        # clahe = apply_CLAHE(cropped)

        cv2.imwrite(dst, cropped)

        t1 = time.perf_counter()
        found += 1
        print(f"✅ Processed {src_basename} in {(t1 - t0) * 1000:.1f} ms")

    print("-----------------------------------------------------------")
    print(f"Processed {found + not_found} images in total:")
    print(f"✅ {found} optic discs detected")
    print(f"❌ {not_found} images without detectable optic discs")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=(
            "Batch-process fundus photographs by detecting the optic nerve head and "
            "cropping that region from each image"
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
        help="Path to the directory where cropped images will be saved. Will be created if it doesn’t exist.",
    )
    p.add_argument(
        "--model",
        "-m",
        required=True,
        help="Path to the pre-trained ONNX model file (.onnx) for optic disc segmentation. The model should output probability maps for disc detection.",
    )

    args = p.parse_args()

    main(
        args.input_dir,
        args.output_dir,
        args.model,
    )
