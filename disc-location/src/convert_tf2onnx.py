import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse

import tensorflow as tf
import tf2onnx

from disc_seg_model import DiscSegModel


def main(path: str, size: int):
    model = DiscSegModel(size)
    model.load_weights(path)

    base_dir = os.path.dirname(path)
    output_path = os.path.join(base_dir, "model.onnx")

    spec = (tf.TensorSpec(shape=(None, size, size, 3), dtype=tf.float32, name="input"),)

    tf2onnx.convert.from_keras(
        model, input_signature=spec, opset=18, output_path=output_path
    )

    print(f"\n✅ ONNX model saved to: {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Convert a TensorFlow model to ONNX format and save in the same folder."
    )
    p.add_argument(
        "--size",
        "-s",
        type=int,
        default=640,
        help="Size of the input image (default: 640).",
    )
    p.add_argument(
        "--model",
        "-m",
        required=True,
        help="Path to the weights of the pre-trained disc segmentation model (.h5).",
    )

    args = p.parse_args()
    main(args.model, args.size)
