"""Synthetic forensic benchmark dataset generator for SafeHER Image Intelligence evaluation.

Provides deterministic, reproducible image generation for authentic images, spliced composites,
localized filter retouches, inpainting artifacts, and robustness stress-test perturbations.
Avoids multi-gigabyte external downloads while ensuring zero data leakage across train, val,
and test splits.
"""

from dataclasses import dataclass
import io
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


@dataclass(frozen=True)
class ImageBenchmarkSample:
    """Benchmark test sample with ground truth manipulation labels.

    Attributes:
        sample_id: Unique deterministic sample identifier.
        image_bytes: Raw encoded JPEG or PNG image bytes.
        is_manipulated: Ground truth binary label (True = manipulated/spliced, False = authentic).
        manipulation_type: Fine-grained category (e.g., 'spliced_compression', 'localized_retouch', 'authentic_gradient').
        generator_source: Synthetic generator or provenance description.
        split: Dataset split ('train', 'val', 'test').
        width: Image width in pixels.
        height: Image height in pixels.
        metadata: Additional provenance metadata (e.g., quality factors, splice coordinates).
    """

    sample_id: str
    image_bytes: bytes
    is_manipulated: bool
    manipulation_type: str
    generator_source: str
    split: str
    width: int
    height: int
    metadata: Dict[str, Any]


def generate_authentic_image(
    seed: int,
    width: int = 256,
    height: int = 256,
    scene_type: str = "gradient",
    jpeg_quality: int = 85,
) -> Tuple[bytes, Dict[str, Any]]:
    """Generate a deterministic authentic image with uniform sensor noise and consistent compression.

    Args:
        seed: Random seed for reproducibility.
        width: Image width.
        height: Image height.
        scene_type: Type of scene ('gradient', 'texture', 'portrait', 'high_contrast', 'recompressed').
        jpeg_quality: Primary JPEG compression quality.

    Returns:
        Tuple of (raw JPEG bytes, metadata dictionary).
    """
    rng = np.random.RandomState(seed)
    arr = np.zeros((height, width, 3), dtype=np.float32)

    if scene_type == "gradient":
        # Smooth sky/studio backdrop gradient with subtle camera sensor noise
        for y in range(height):
            arr[y, :, 0] = np.linspace(80, 180, width) + (y / height) * 40
            arr[y, :, 1] = np.linspace(100, 200, width) - (y / height) * 20
            arr[y, :, 2] = np.linspace(140, 240, width) + (y / height) * 15
        arr += rng.normal(0.0, 2.5, (height, width, 3))

    elif scene_type == "texture":
        # Multi-frequency procedural landscape / fabric texture
        x_coords = np.linspace(0, 4 * math.pi, width)
        y_coords = np.linspace(0, 4 * math.pi, height)
        xx, yy = np.meshgrid(x_coords, y_coords)
        base = np.sin(xx) * np.cos(yy) * 40.0 + 128.0
        arr[:, :, 0] = base + np.sin(xx * 2) * 20.0
        arr[:, :, 1] = base + np.cos(yy * 2) * 15.0
        arr[:, :, 2] = base + 10.0
        arr += rng.normal(0.0, 3.5, (height, width, 3))

    elif scene_type == "portrait":
        # Synthetic portrait simulation with uniform face gradient and camera noise
        for y in range(height):
            arr[y, :, 0] = 180 + rng.normal(0, 2, width)
            arr[y, :, 1] = 140 + rng.normal(0, 2, width)
            arr[y, :, 2] = 120 + rng.normal(0, 2, width)
        # Add soft central facial oval
        y_center, x_center = height // 2, width // 2
        yy, xx = np.ogrid[:height, :width]
        dist_sq = ((xx - x_center) / (width * 0.25)) ** 2 + ((yy - y_center) / (height * 0.35)) ** 2
        mask = dist_sq <= 1.0
        arr[mask, 0] = np.clip(arr[mask, 0] + 30.0, 0, 255)
        arr[mask, 1] = np.clip(arr[mask, 1] + 20.0, 0, 255)

    elif scene_type == "high_contrast":
        # Architectural lines / sharp geometric edges (authentic false-positive stress test)
        arr[:] = 200.0
        for i in range(20, width - 20, 30):
            arr[:, i : i + 4, :] = 40.0
        for j in range(20, height - 20, 30):
            arr[j : j + 4, :, :] = 40.0
        arr += rng.normal(0.0, 1.5, (height, width, 3))

    elif scene_type == "recompressed":
        # Authentic image passed through uniform dual-pass recompression (social-media simulation)
        for y in range(height):
            arr[y, :, 0] = (y / height) * 200.0
            arr[y, :, 1] = 120.0
            arr[y, :, 2] = 200.0 - (y / height) * 150.0
        arr += rng.normal(0.0, 3.0, (height, width, 3))

    else:
        arr[:] = 128.0 + rng.normal(0.0, 5.0, (height, width, 3))

    arr_uint8 = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr_uint8, mode="RGB")

    buf = io.BytesIO()
    if scene_type == "recompressed":
        # First pass Q=75, second pass Q=jpeg_quality
        temp_buf = io.BytesIO()
        img.save(temp_buf, format="JPEG", quality=75)
        temp_buf.seek(0)
        re_img = Image.open(temp_buf).convert("RGB")
        re_img.save(buf, format="JPEG", quality=jpeg_quality)
    else:
        img.save(buf, format="JPEG", quality=jpeg_quality)

    metadata = {
        "scene_type": scene_type,
        "jpeg_quality": jpeg_quality,
        "seed": seed,
        "is_manipulated": False,
    }
    return buf.getvalue(), metadata


def generate_manipulated_image(
    seed: int,
    width: int = 256,
    height: int = 256,
    manipulation_type: str = "spliced_compression",
    jpeg_quality: int = 85,
) -> Tuple[bytes, Dict[str, Any]]:
    """Generate a deterministic manipulated image exhibiting digital forensics signatures.

    Args:
        seed: Random seed for reproducibility.
        width: Image width.
        height: Image height.
        manipulation_type: Type of manipulation ('spliced_compression', 'spliced_noise', 'localized_retouch', 'boundary_inpainting', 'copy_move').
        jpeg_quality: Output JPEG quality.

    Returns:
        Tuple of (raw JPEG bytes, metadata dictionary).
    """
    rng = np.random.RandomState(seed)

    # 1. Base image background
    bg_arr = np.zeros((height, width, 3), dtype=np.float32)
    for y in range(height):
        bg_arr[y, :, 0] = np.linspace(70, 170, width)
        bg_arr[y, :, 1] = 110.0 + rng.normal(0, 2, width)
        bg_arr[y, :, 2] = np.linspace(150, 230, width)
    bg_arr_uint8 = np.clip(bg_arr, 0, 255).astype(np.uint8)
    base_img = Image.fromarray(bg_arr_uint8, mode="RGB")

    patch_size = 70
    patch_x = width // 4
    patch_y = height // 4

    if manipulation_type == "spliced_compression":
        # Background pre-compressed at low quality (Q=60)
        buf_bg = io.BytesIO()
        base_img.save(buf_bg, format="JPEG", quality=60)
        buf_bg.seek(0)
        compressed_base = Image.open(buf_bg).convert("RGB")

        # Foreground patch rendered with sharp high-frequency texture, saved at Q=98
        fg_arr = rng.randint(40, 220, (patch_size, patch_size, 3), dtype=np.uint8)
        fg_img = Image.fromarray(fg_arr, mode="RGB")
        buf_fg = io.BytesIO()
        fg_img.save(buf_fg, format="JPEG", quality=98)
        buf_fg.seek(0)
        compressed_fg = Image.open(buf_fg).convert("RGB")

        # Composite together and resave at target quality
        composite = compressed_base.copy()
        composite.paste(compressed_fg, (patch_x, patch_y))
        buf_out = io.BytesIO()
        composite.save(buf_out, format="JPEG", quality=jpeg_quality)

    elif manipulation_type == "spliced_noise":
        # Background with low noise (sigma=1.0), foreground with high noise (sigma=15.0)
        base_arr = np.array(base_img, dtype=np.float32)
        base_arr += rng.normal(0.0, 1.0, base_arr.shape)

        patch = base_arr[patch_y : patch_y + patch_size, patch_x : patch_x + patch_size].copy()
        patch += rng.normal(0.0, 18.0, patch.shape)
        base_arr[patch_y : patch_y + patch_size, patch_x : patch_x + patch_size] = patch

        composite = Image.fromarray(np.clip(base_arr, 0, 255).astype(np.uint8), mode="RGB")
        buf_out = io.BytesIO()
        composite.save(buf_out, format="JPEG", quality=jpeg_quality)

    elif manipulation_type == "localized_retouch":
        # Portrait smoothing: selective Gaussian blur on central quadrant simulating beauty retouching
        composite = base_img.copy()
        blurred_region = composite.crop((patch_x, patch_y, patch_x + patch_size, patch_y + patch_size))
        blurred_region = blurred_region.filter(ImageFilter.GaussianBlur(radius=3.5))
        composite.paste(blurred_region, (patch_x, patch_y))

        buf_out = io.BytesIO()
        composite.save(buf_out, format="JPEG", quality=jpeg_quality)

    elif manipulation_type == "boundary_inpainting":
        # Central patch filled with interpolated flat color/gradient (loss of sensor noise)
        composite = base_img.copy()
        draw = ImageDraw.Draw(composite)
        draw.rectangle(
            [patch_x, patch_y, patch_x + patch_size, patch_y + patch_size],
            fill=(140, 140, 140),
            outline=(160, 160, 160),
        )
        buf_out = io.BytesIO()
        composite.save(buf_out, format="JPEG", quality=jpeg_quality)

    elif manipulation_type == "copy_move":
        # Duplicate region cloned from (20, 20) to (patch_x, patch_y) with edge artifact
        composite = base_img.copy()
        clone_src = composite.crop((20, 20, 20 + patch_size, 20 + patch_size))
        composite.paste(clone_src, (patch_x, patch_y))
        buf_out = io.BytesIO()
        composite.save(buf_out, format="JPEG", quality=jpeg_quality)

    else:
        # Default splice
        composite = base_img.copy()
        draw = ImageDraw.Draw(composite)
        draw.ellipse([patch_x, patch_y, patch_x + patch_size, patch_y + patch_size], fill=(220, 50, 50))
        buf_out = io.BytesIO()
        composite.save(buf_out, format="JPEG", quality=jpeg_quality)

    metadata = {
        "manipulation_type": manipulation_type,
        "patch_location": (patch_x, patch_y, patch_size, patch_size),
        "jpeg_quality": jpeg_quality,
        "seed": seed,
        "is_manipulated": True,
    }
    return buf_out.getvalue(), metadata


def apply_perturbation(image_bytes: bytes, perturbation_type: str, **kwargs: Any) -> bytes:
    """Apply a robustness transformation / perturbation to an encoded image.

    Supported perturbations:
        - 'jpeg_q50': Heavy JPEG recompression (Q=50)
        - 'jpeg_q75': Moderate JPEG recompression (Q=75)
        - 'jpeg_q90': Light JPEG recompression (Q=90)
        - 'resize_down_up': 0.5x downscaling followed by upscaling (preview simulation)
        - 'center_crop': 0.7x center crop resized back to original resolution
        - 'gaussian_blur': Slight Gaussian blur (radius=1.5)
        - 'brightness_shift': Intensity shift (+30 pixel values)
        - 'contrast_shift': 0.7x contrast reduction
        - 'screenshot_composite': White border and status bar framing simulation
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = img.size
    out_buf = io.BytesIO()

    if perturbation_type in ("jpeg_q50", "jpeg_q75", "jpeg_q90"):
        quality = int(perturbation_type.split("_q")[1])
        img.save(out_buf, format="JPEG", quality=quality)

    elif perturbation_type == "resize_down_up":
        down = img.resize((max(16, orig_w // 2), max(16, orig_h // 2)), Image.Resampling.BILINEAR)
        up = down.resize((orig_w, orig_h), Image.Resampling.BILINEAR)
        up.save(out_buf, format="JPEG", quality=85)

    elif perturbation_type == "center_crop":
        crop_w = int(orig_w * 0.7)
        crop_h = int(orig_h * 0.7)
        x0 = (orig_w - crop_w) // 2
        y0 = (orig_h - crop_h) // 2
        cropped = img.crop((x0, y0, x0 + crop_w, y0 + crop_h))
        restored = cropped.resize((orig_w, orig_h), Image.Resampling.BILINEAR)
        restored.save(out_buf, format="JPEG", quality=85)

    elif perturbation_type == "gaussian_blur":
        blurred = img.filter(ImageFilter.GaussianBlur(radius=1.5))
        blurred.save(out_buf, format="JPEG", quality=85)

    elif perturbation_type == "brightness_shift":
        enhancer = ImageEnhance.Brightness(img)
        enhanced = enhancer.enhance(1.25)
        enhanced.save(out_buf, format="JPEG", quality=85)

    elif perturbation_type == "contrast_shift":
        enhancer = ImageEnhance.Contrast(img)
        enhanced = enhancer.enhance(0.70)
        enhanced.save(out_buf, format="JPEG", quality=85)

    elif perturbation_type == "screenshot_composite":
        # Framing simulation: place image on top of a neutral messenger window frame
        framed = Image.new("RGB", (orig_w, orig_h), (240, 242, 245))
        inner_w = int(orig_w * 0.85)
        inner_h = int(orig_h * 0.85)
        inner = img.resize((inner_w, inner_h), Image.Resampling.BILINEAR)
        framed.paste(inner, ((orig_w - inner_w) // 2, (orig_h - inner_h) // 2))
        framed.save(out_buf, format="JPEG", quality=85)

    else:
        # Pass-through
        img.save(out_buf, format="JPEG", quality=85)

    return out_buf.getvalue()


def build_forensic_benchmark_dataset(
    num_train: int = 40,
    num_val: int = 30,
    num_test: int = 50,
    base_seed: int = 1000,
) -> List[ImageBenchmarkSample]:
    """Construct a partitioned synthetic benchmark dataset for regression and safety analysis.

    Splits (40 train, 30 val, 50 test by default) use separate deterministic seed offsets
    while reusing the same five authentic scene families and five manipulation families:
    - Train: synthetic exploration and threshold regression checks.
    - Val: synthetic boundary validation checks.
    - Test: held-out synthetic seed partition (not an independent real-world benchmark).
    """
    samples: List[ImageBenchmarkSample] = []

    authentic_types = ["gradient", "texture", "portrait", "high_contrast", "recompressed"]
    manip_types = ["spliced_compression", "spliced_noise", "localized_retouch", "boundary_inpainting", "copy_move"]

    splits = [
        ("train", num_train, base_seed),
        ("val", num_val, base_seed + 10000),
        ("test", num_test, base_seed + 20000),
    ]

    for split_name, count, seed_offset in splits:
        half = count // 2
        # Authentic samples
        for i in range(half):
            sample_seed = seed_offset + i
            scene = authentic_types[i % len(authentic_types)]
            data, meta = generate_authentic_image(seed=sample_seed, scene_type=scene)
            sample_id = f"{split_name.upper()}_AUTH_{i+1:03d}"
            samples.append(
                ImageBenchmarkSample(
                    sample_id=sample_id,
                    image_bytes=data,
                    is_manipulated=False,
                    manipulation_type=f"authentic_{scene}",
                    generator_source=f"synthetic_camera_{scene}",
                    split=split_name,
                    width=256,
                    height=256,
                    metadata=meta,
                )
            )

        # Manipulated samples
        for j in range(count - half):
            sample_seed = seed_offset + half + j
            manip = manip_types[j % len(manip_types)]
            data, meta = generate_manipulated_image(seed=sample_seed, manipulation_type=manip)
            sample_id = f"{split_name.upper()}_MANIP_{j+1:03d}"
            samples.append(
                ImageBenchmarkSample(
                    sample_id=sample_id,
                    image_bytes=data,
                    is_manipulated=True,
                    manipulation_type=manip,
                    generator_source=f"synthetic_splice_{manip}",
                    split=split_name,
                    width=256,
                    height=256,
                    metadata=meta,
                )
            )

    return samples
