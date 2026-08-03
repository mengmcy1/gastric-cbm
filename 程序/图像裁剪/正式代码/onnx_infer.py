#!/usr/bin/env python3
"""使用shiye_V1分割内镜有效视野，并生成gastric-cbm候选裁剪。"""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ModuleNotFoundError:
    ort = None


from config import MODEL_PATH, INPUT_ROOT, OUTPUT_ROOT

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
INPUT_SIZE = 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_MODEL = MODEL_PATH


class _TensorMetadata:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class OpenCVDnnSession:
    """提供本流程所需的最小ONNX Runtime兼容接口。"""

    backend_name = "opencv_dnn_cpu"

    def __init__(self, model_path):
        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        self._inputs = [_TensorMetadata("input", [1, 3, INPUT_SIZE, INPUT_SIZE])]
        self._outputs = [_TensorMetadata("output", None)]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, _, feeds):
        model_input = next(iter(feeds.values()))
        self.net.setInput(model_input)
        return [self.net.forward()]


def create_inference_session(model_path):
    if ort is not None:
        return ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
    return OpenCVDnnSession(model_path)


def inference_backend_name(session):
    return getattr(session, "backend_name", "onnxruntime_cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="递归分割内镜有效视野，并输出掩膜、裁剪图和原图映射表。"
    )
    parser.add_argument("--onnx", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--input", type=Path, default=INPUT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_ROOT / "图像裁剪候选" / "shiye_V1_safe_rectangle",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--crop-strategy",
        choices=("safe_rectangle", "bbox", "none"),
        default="safe_rectangle",
        help="safe_rectangle 保证矩形位于预测视野内；bbox 仅取外接框；none 只输出掩膜。",
    )
    parser.add_argument(
        "--erode-pixels",
        type=int,
        default=1,
        help="在 256×256 模型掩膜上向内收缩的像素数，默认 1。",
    )
    parser.add_argument(
        "--black-threshold",
        type=int,
        default=12,
        help="剔除与图像边缘连通、各通道均不高于该值的近黑区域；0 表示关闭。",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def validate_args(args):
    if not args.onnx.is_file():
        raise FileNotFoundError(f"ONNX 模型不存在：{args.onnx}")
    if not args.input.exists():
        raise FileNotFoundError(f"输入不存在：{args.input}")
    if not 0 < args.threshold < 1:
        raise ValueError("--threshold 必须位于 (0, 1)")
    if args.erode_pixels < 0:
        raise ValueError("--erode-pixels 不能为负数")
    if not 0 <= args.black_threshold <= 255:
        raise ValueError("--black-threshold 必须位于 [0, 255]")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit 必须为正整数")

    input_resolved = args.input.resolve()
    output_resolved = args.output.resolve()
    if args.input.is_dir() and (
        output_resolved == input_resolved or input_resolved in output_resolved.parents
    ):
        raise ValueError("输出目录不能位于原始输入目录内部")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"输出目录不是空目录，拒绝覆盖：{args.output}")


def list_images(input_path):
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"不支持的图像格式：{input_path}")
        return [input_path]
    return sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def relative_image_path(image_path, input_path):
    if input_path.is_file():
        return Path(image_path.name)
    return image_path.relative_to(input_path)


def read_image(image_path):
    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图像：{image_path}")
    return image


def preprocess(image_bgr):
    """模型已通过工程 A/B 确认为 BGR + 双次除255。"""
    resized = cv2.resize(
        image_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)
    normalized = (resized / 255.0 - MEAN) / STD
    normalized = normalized / 255.0
    return normalized.transpose(2, 0, 1)[None].astype(np.float32)


def sigmoid(values):
    values = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def extract_probability(model_output):
    logits = np.squeeze(np.asarray(model_output))
    if logits.ndim != 2:
        raise ValueError(f"预期二维单通道 logits，实际 shape={model_output.shape}")
    return sigmoid(logits)


def largest_component(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        raise ValueError("模型未检出有效视野")
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = (labels == index).astype(np.uint8)
    return component, count - 1


def remove_border_connected_black(mask, image_bgr, black_threshold):
    if black_threshold == 0:
        return mask
    resized = cv2.resize(
        image_bgr, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA
    )
    near_black = (resized.max(axis=2) <= black_threshold).astype(np.uint8)
    count, labels = cv2.connectedComponents(near_black, connectivity=8)
    if count <= 1:
        return mask
    border_labels = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    border_labels = border_labels[border_labels != 0]
    if not len(border_labels):
        return mask
    exterior_black = np.isin(labels, border_labels).astype(np.uint8)
    exterior_black = cv2.dilate(exterior_black, np.ones((3, 3), np.uint8))
    return mask & (1 - exterior_black)


def clean_component(mask, image_bgr, erode_pixels, black_threshold):
    kernel = np.ones((3, 3), dtype=np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cleaned = remove_border_connected_black(cleaned, image_bgr, black_threshold)
    if erode_pixels:
        size = 2 * erode_pixels + 1
        cleaned = cv2.erode(cleaned, np.ones((size, size), np.uint8), iterations=1)
    if not cleaned.any():
        raise ValueError("掩膜内缩后为空")
    return cleaned


def largest_inner_rectangle(mask):
    """返回二值掩膜内面积最大的轴对齐矩形 (x, y, width, height)。"""
    heights = np.zeros(mask.shape[1], dtype=np.int32)
    best_area = 0
    best = None

    for y, row in enumerate(mask):
        heights = np.where(row > 0, heights + 1, 0)
        stack = []
        for x in range(mask.shape[1] + 1):
            height = int(heights[x]) if x < mask.shape[1] else 0
            start = x
            while stack and stack[-1][1] > height:
                start_index, previous_height = stack.pop()
                area = previous_height * (x - start_index)
                if area > best_area:
                    best_area = area
                    best = (
                        start_index,
                        y - previous_height + 1,
                        x - start_index,
                        previous_height,
                    )
                start = start_index
            if not stack or stack[-1][1] < height:
                stack.append((start, height))

    if best is None or best_area == 0:
        raise ValueError("无法从掩膜计算安全内接矩形")
    return best


def component_bbox(mask):
    points = cv2.findNonZero(mask)
    if points is None:
        raise ValueError("有效视野掩膜为空")
    return cv2.boundingRect(points)


def scale_box(box, source_width, source_height, target_width, target_height):
    x, y, width, height = box
    left = max(0, math.floor(x * target_width / source_width))
    top = max(0, math.floor(y * target_height / source_height))
    right = min(target_width, math.ceil((x + width) * target_width / source_width))
    bottom = min(target_height, math.ceil((y + height) * target_height / source_height))
    if right <= left or bottom <= top:
        raise ValueError(f"缩放后的裁剪框无效：{(left, top, right, bottom)}")
    return left, top, right, bottom


def encode_and_save(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有文件：{path}")
    suffix = path.suffix.lower()
    parameters = [cv2.IMWRITE_JPEG_QUALITY, 95] if suffix in {".jpg", ".jpeg"} else []
    ok, encoded = cv2.imencode(suffix, image, parameters)
    if not ok:
        raise ValueError(f"无法编码图像：{path}")
    encoded.tofile(str(path))


def verify_session(session):
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or inputs[0].shape != [1, 3, INPUT_SIZE, INPUT_SIZE]:
        raise ValueError(f"不支持的模型输入：{[(item.name, item.shape) for item in inputs]}")
    if len(outputs) < 1:
        raise ValueError("模型没有输出")
    return inputs[0].name


def process_one(session, input_name, image_path, relative_path, args):
    image = read_image(image_path)
    model_input = preprocess(image)
    probability = extract_probability(session.run(None, {input_name: model_input})[0])
    raw_mask = (probability > args.threshold).astype(np.uint8)
    component, component_count = largest_component(raw_mask)
    cleaned = clean_component(
        component,
        image,
        args.erode_pixels,
        args.black_threshold,
    )

    original_height, original_width = image.shape[:2]
    display_mask = cv2.resize(
        cleaned,
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST,
    )
    mask_path = args.output / "masks" / relative_path.with_suffix(".png")
    encode_and_save(mask_path, display_mask * 255)

    box = None
    crop_path = None
    if args.crop_strategy != "none":
        if args.crop_strategy == "safe_rectangle":
            model_box = largest_inner_rectangle(cleaned)
        else:
            model_box = component_bbox(cleaned)
        box = scale_box(
            model_box,
            cleaned.shape[1],
            cleaned.shape[0],
            original_width,
            original_height,
        )
        left, top, right, bottom = box
        crop = image[top:bottom, left:right]
        crop_path = args.output / "crops" / relative_path
        encode_and_save(crop_path, crop)

    return {
        "source": str(image_path),
        "relative_path": str(relative_path),
        "mask": str(mask_path),
        "crop": str(crop_path) if crop_path else "",
        "source_width": original_width,
        "source_height": original_height,
        "mask_coverage": float(cleaned.mean()),
        "component_count_before_largest": component_count,
        "black_threshold": args.black_threshold,
        "erode_pixels": args.erode_pixels,
        "crop_strategy": args.crop_strategy,
        "left": box[0] if box else "",
        "top": box[1] if box else "",
        "right": box[2] if box else "",
        "bottom": box[3] if box else "",
        "crop_width": box[2] - box[0] if box else "",
        "crop_height": box[3] - box[1] if box else "",
    }


def main():
    args = parse_args()
    validate_args(args)
    image_paths = list_images(args.input)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise FileNotFoundError(f"未找到图像：{args.input}")

    args.output.mkdir(parents=True, exist_ok=True)
    session = create_inference_session(args.onnx)
    print(f"推理后端：{inference_backend_name(session)}")
    input_name = verify_session(session)

    rows = []
    for index, image_path in enumerate(image_paths, start=1):
        relative_path = relative_image_path(image_path, args.input)
        row = process_one(session, input_name, image_path, relative_path, args)
        rows.append(row)
        print(f"[{index}/{len(image_paths)}] {image_path} -> {row['crop'] or row['mask']}")

    mapping_path = args.output / "mapping.csv"
    with mapping_path.open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"完成：{len(rows)} 张；映射表：{mapping_path}")


if __name__ == "__main__":
    main()
