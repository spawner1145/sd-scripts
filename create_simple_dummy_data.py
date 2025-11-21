import math
import os
import random
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from PIL import Image, ImageDraw

CANVAS_SIZE = 1024
DATASET_ROOT = "dummy_data_shapes"
OUTPUT_FOLDER = "10_imgs"
NUM_IMAGES = 40


def create_background(color: Tuple[int, int, int]) -> Image.Image:
    """Create a lightly textured background so every sample stays unique."""

    img = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), color)
    noise_strength = random.randint(1, 8)
    arr = np.array(img).astype(np.int16)
    noise = np.random.randint(-noise_strength, noise_strength + 1, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def random_bbox(square: bool = True, min_ratio: float = 0.25, max_ratio: float = 0.75) -> Tuple[int, int, int, int]:
    size_ratio = random.uniform(min_ratio, max_ratio)
    width = int(CANVAS_SIZE * size_ratio)
    if square:
        height = width
    else:
        height = int(width * random.uniform(0.5, 1.4))
        height = max(int(CANVAS_SIZE * 0.15), min(height, CANVAS_SIZE - 80))

    width = min(width, CANVAS_SIZE - 80)
    x0 = random.randint(40, CANVAS_SIZE - width - 40)
    y0 = random.randint(40, CANVAS_SIZE - height - 40)
    return x0, y0, x0 + width, y0 + height


def draw_circle(draw: ImageDraw.ImageDraw, bbox, fill, outline):
    draw.ellipse(bbox, fill=fill, outline=outline, width=10)


def draw_square(draw: ImageDraw.ImageDraw, bbox, fill, outline):
    draw.rectangle(bbox, fill=fill, outline=outline, width=10)


def draw_rectangle(draw: ImageDraw.ImageDraw, bbox, fill, outline):
    draw.rectangle(bbox, fill=fill, outline=outline, width=10)


def draw_triangle(draw: ImageDraw.ImageDraw, bbox, fill, outline):
    x0, y0, x1, y1 = bbox
    points = [(0.5 * (x0 + x1), y0), (x0, y1), (x1, y1)]
    draw.polygon(points, fill=fill, outline=outline)


def draw_diamond(draw: ImageDraw.ImageDraw, bbox, fill, outline):
    x0, y0, x1, y1 = bbox
    points = [
        (0.5 * (x0 + x1), y0),
        (x1, 0.5 * (y0 + y1)),
        (0.5 * (x0 + x1), y1),
        (x0, 0.5 * (y0 + y1)),
    ]
    draw.polygon(points, fill=fill, outline=outline)


def draw_pentagon(draw: ImageDraw.ImageDraw, bbox, fill, outline):
    x0, y0, x1, y1 = bbox
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    rx = 0.5 * (x1 - x0)
    ry = 0.5 * (y1 - y0)
    points = []
    for i in range(5):
        angle = math.radians(90 + i * 72)
        points.append((cx + rx * math.cos(angle), cy + ry * math.sin(angle)))
    draw.polygon(points, fill=fill, outline=outline)


def apply_light_shadow(draw: ImageDraw.ImageDraw):
    shadow_bbox = random_bbox(square=False, min_ratio=0.1, max_ratio=0.3)
    shade = random.randint(210, 240)
    color = (shade, shade, shade)
    draw.ellipse(shadow_bbox, fill=color)


@dataclass
class ShapeSpec:
    name: str
    drawer: Callable[[ImageDraw.ImageDraw, Tuple[int, int, int, int], Tuple[int, int, int], Tuple[int, int, int]], None]
    square_bbox: bool


@dataclass
class ColorSpec:
    name: str
    fill: Tuple[int, int, int]
    outline: Tuple[int, int, int]


SHAPES = [
    ShapeSpec("circle", draw_circle, True),
    ShapeSpec("square", draw_square, True),
    ShapeSpec("rectangle", draw_rectangle, False),
    ShapeSpec("triangle", draw_triangle, True),
    ShapeSpec("diamond", draw_diamond, True),
    ShapeSpec("pentagon", draw_pentagon, True),
]


COLORS = [
    ColorSpec("crimson", (220, 60, 80), (120, 25, 40)),
    ColorSpec("amber", (245, 185, 60), (155, 110, 15)),
    ColorSpec("emerald", (60, 185, 120), (25, 90, 55)),
    ColorSpec("sapphire", (70, 115, 230), (25, 45, 120)),
    ColorSpec("violet", (160, 95, 225), (85, 35, 150)),
    ColorSpec("teal", (55, 170, 185), (30, 90, 105)),
]

BACKGROUND_DESCRIPTIONS = [
    "a softly textured white canvas",
    "a grainy watercolor paper",
    "a pale icy backdrop",
    "a muted parchment surface",
    "a light silver fabric",
]

POSITION_DESCRIPTIONS = {
    (-1, -1): "near the upper left",
    (0, -1): "along the top center",
    (1, -1): "near the upper right",
    (-1, 0): "toward the center left",
    (0, 0): "right in the middle",
    (1, 0): "toward the center right",
    (-1, 1): "near the lower left",
    (0, 1): "along the bottom center",
    (1, 1): "near the lower right",
}


SIZE_BUCKETS = [
    (0.25, "a small"),
    (0.38, "a modest"),
    (0.52, "a medium"),
    (0.68, "a large"),
    (1.00, "an oversized"),
]


def describe_size(width: int) -> str:
    ratio = width / CANVAS_SIZE
    for threshold, label in SIZE_BUCKETS:
        if ratio <= threshold:
            return label
    return SIZE_BUCKETS[-1][1]


def describe_position(bbox: Tuple[int, int, int, int]) -> str:
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    def bucket(value):
        if value < CANVAS_SIZE / 3:
            return -1
        if value > 2 * CANVAS_SIZE / 3:
            return 1
        return 0

    key = (bucket(cx), bucket(cy))
    return POSITION_DESCRIPTIONS.get(key, "somewhere on the canvas")


def build_caption(index: int, background_desc: str, size_desc: str, color_name: str, shape_name: str, position_desc: str) -> str:
    return (
        f"Sample {index + 1:02d}: on {background_desc}, {size_desc} {color_name} {shape_name} rests {position_desc}."
    )


def save_sample(output_dir: str, index: int):
    base_color = random.randint(232, 250)
    background = create_background((base_color, base_color, base_color))
    draw = ImageDraw.Draw(background)

    if random.random() < 0.35:
        apply_light_shadow(draw)

    shape = random.choice(SHAPES)
    color = random.choice(COLORS)
    bbox = random_bbox(square=shape.square_bbox)
    shape.drawer(draw, bbox, color.fill, color.outline)

    if random.random() < 0.2:
        outline_color = tuple(max(0, c - 40) for c in color.outline)
        draw.rectangle([bbox[0] - 6, bbox[1] - 6, bbox[2] + 6, bbox[3] + 6], outline=outline_color, width=3)

    background_desc = random.choice(BACKGROUND_DESCRIPTIONS)
    size_desc = describe_size(bbox[2] - bbox[0])
    position_desc = describe_position(bbox)
    caption = build_caption(index, background_desc, size_desc, color.name, shape.name, position_desc)

    image_path = os.path.join(output_dir, f"shape_{index:03d}.png")
    caption_path = image_path.replace(".png", ".txt")
    background.save(image_path)
    with open(caption_path, "w", encoding="utf-8") as caption_file:
        caption_file.write(caption)


def create_dummy_data(num_images: int = NUM_IMAGES, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    output_dir = os.path.join(DATASET_ROOT, "img", OUTPUT_FOLDER)
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_images):
        save_sample(output_dir, i)

    print(f"Generated {num_images} samples under {DATASET_ROOT}/img/{OUTPUT_FOLDER}")


if __name__ == "__main__":
    create_dummy_data()
