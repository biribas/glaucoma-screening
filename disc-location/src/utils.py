import cv2
import numpy as np
import onnxruntime as ort
from cv2.typing import MatLike
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_holes


def make_predict_fn(model):
    input_name = model.get_inputs()[0].name
    output_names = [o.name for o in model.get_outputs()]

    def predict(input_img):
        return model.run(output_names, {input_name: input_img})

    return predict


def create_onnx_model(model_path: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_mem_pattern = True
    so.enable_cpu_mem_arena = True

    sess = ort.InferenceSession(
        model_path,
        providers=[
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
        sess_options=so,
    )
    return sess


def run_inference(img: MatLike, predict, input_shape):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Prepare input image
    input = cv2.resize(img, input_shape, interpolation=cv2.INTER_LINEAR)
    input = input.astype("float32")[np.newaxis, ...]  # shape: (1, H, W, 3)

    *_, prob_map = predict(input)  # Run inference
    prob_map = np.asarray(prob_map).reshape(input_shape)  # reshape to 2D map

    # zero out border artifacts
    border = input_shape[0] // 5
    prob_map[:border, :] = 0
    prob_map[-border:, :] = 0

    return prob_map


def get_binary_mask(prob_map: np.ndarray, base_threshold: float = 0.5):
    """
    Convert a probability map into a clean binary mask:
    - Apply adaptive thresholding
    - Keep only the largest connected component
    - Fill small holes inside the main region
    """
    # Adaptative thresholding
    max_value = prob_map.max()
    if max_value < 0.01:
        return None

    threshold = base_threshold if max_value > base_threshold else (max_value / 2.0)
    mask = prob_map > threshold

    labeled = label(mask)
    regions = regionprops(labeled)
    if not regions:
        return None

    region_areas = [region.area for region in regions]
    largest_label = regions[np.argmax(region_areas)].label
    mask = labeled == largest_label

    filled_mask = remove_small_holes(mask, area_threshold=1000)
    return filled_mask.astype(np.uint8)


def crop_disc(
    img: np.ndarray,
    mask: np.ndarray,
    input_shape,
    output_size=512,
) -> np.ndarray:
    h_img, w_img, _ = img.shape
    scale_y = h_img / input_shape[0]
    scale_x = w_img / input_shape[1]

    # An optic disc has a radius of approximately 150 pixels in an image with a height of 1920 pixels.
    disc_radius = 150 * h_img / 1920

    regions = regionprops(label(mask))
    cy, cx = regions[0].centroid
    cy *= scale_y
    cx *= scale_x

    side = int(5 * disc_radius)
    half_side = side // 2

    y1 = max(int(cy - half_side), 0)
    y2 = min(int(cy + half_side), h_img)
    x1 = max(int(cx - half_side), 0)
    x2 = min(int(cx + half_side), w_img)

    side = min(y2 - y1, x2 - x1)
    crop = np.zeros((side, side, 3), dtype=img.dtype)
    crop = img[y1:y2, x1:x2]

    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)


def apply_CLAHE(img):
    # Converting image to LAB Color so CLAHE can be applied to the luminance channel
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Splitting the LAB image to L, A and B channels, respectively
    L, A, B = cv2.split(lab_img)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(L)

    # Combine the CLAHE enhanced L-channel back with A and B channels
    updated_lab_img2 = cv2.merge((clahe_img, A, B))

    # Convert LAB image back to color (RGB)
    return cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)
