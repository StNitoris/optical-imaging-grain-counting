"""固定视野下的小麦与玉米计数。直接运行处理自带样图，也可传入照片路径。"""

import argparse
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
LONG_EDGE = 1020
MIN_AREA = 100  # 统一缩放后，排除鼠标垫上的碎屑，单位为像素
# 纯样图中最大小麦为 1087 像素，最小玉米为 3041 像素，取两者中点。
CORN_AREA = (1087 + 3041) // 2


def count_image(path, destination):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图像 {path}")
    height, width = image.shape[:2]
    scale = LONG_EDGE / max(height, width)
    image = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                       interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    annotated = image.copy()
    keep = np.zeros(len(stats), dtype=np.uint8)
    counts = {"Wheat": 0, "Corn": 0}
    colors = {"Wheat": (240, 210, 40), "Corn": (170, 60, 240)}
    for label, (x, y, w, h, area) in enumerate(stats[1:], start=1):
        if area < MIN_AREA:
            continue
        keep[label] = 255
        kind = "Corn" if area >= CORN_AREA else "Wheat"
        counts[kind] += 1
        cv2.rectangle(annotated, (x, y), (x + w - 1, y + h - 1), colors[kind], 2)

    # 将统计写入图上方的独立留白，避免遮挡边缘颗粒。
    result = cv2.copyMakeBorder(annotated, 46, 0, 0, 0, cv2.BORDER_CONSTANT)
    for x, kind in [(12, "Wheat"), (185, "Corn")]:
        cv2.putText(result, f"{kind}: {counts[kind]}", (x, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, colors[kind], 2, cv2.LINE_AA)
    cv2.putText(result, f"Total: {sum(counts.values())}", (350, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in [("01_灰度", gray), ("02_滤波", smooth), ("03_二值", binary),
                        ("04_面积筛选", keep[labels]), ("05_计数", result)]:
        ok, data = cv2.imencode(".png", frame)
        if not ok:
            raise OSError(f"无法编码 {name}")
        data.tofile(destination / f"{name}.png")
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", type=Path, help="待处理照片路径")
    parser.add_argument("--output", type=Path, default=ROOT / "结果", help="结果目录")
    args = parser.parse_args()
    for path in args.images or sorted((ROOT / "样图").glob("*.jpg")):
        counts = count_image(path, args.output / path.stem)
        print(f"{path.name}  小麦 {counts['Wheat']}  玉米 {counts['Corn']}")
