#!/usr/bin/env python3
"""
GVK Private AI - ASCII Portrait Generator (tools/ascii_portrait.py)

Converts standard photos (.jpg, .png, .heic, .webp, .bmp) into high-contrast
ASCII and Unicode Braille sprites for the GVK Private AI terminal interface (TUI).

Key Responsibilities:
  - Load and center-crop portrait images via Pillow or macOS native `sips` (zero-dependency fallback).
  - Enhance facial clarity (eyes, brows, smile, glasses) using percentile normalization,
    unsharp-mask edge boosting, and contrast/gamma tuning.
  - Render terminal-aspect-corrected art in four styles (`clean`, `cyber`, `blocks`, `braille`).
  - Batch-generate five persona emotion sprites (`idle`, `sleeping`, `smiling`, `talking`, `talking2`).

Usage:
  python3 tools/ascii_portrait.py photo.jpg -o portraits/idle.txt --color cyan
  python3 tools/ascii_portrait.py photo.jpg --style braille -w 44 --color cyan
  python3 tools/ascii_portrait.py --init-pack ./my_photos/ --out-dir ./portraits/friend_1/
  python3 tools/ascii_portrait.py self-test
"""

import argparse
import math
import os
import struct
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional

# =====================================================================
# CONSTANTS & PALETTES
# =====================================================================
# Character palettes ordered from darkest to brightest (for dark terminal backgrounds)
PALETTES = {
    "cyber": " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$",
    "clean": " .:-=+*#%@",
    "blocks": " ░▒▓█",
}

ANSI_COLORS = {
    "cyan": "\033[96m",
    "amber": "\033[93m",
    "violet": "\033[95m",
    "green": "\033[92m",
    "red": "\033[91m",
    "white": "\033[97m",
    "reset": "\033[0m",
}


# =====================================================================
# IMAGE LOADING & PREPROCESSING
# =====================================================================
def load_grayscale_grid(image_path: str, target_w: int, target_h: int):
    """Load an image, center-crop to portrait proportions, and resize to a grayscale grid.

    Uses Pillow when installed, falling back to macOS native `sips` converted to BMP
    so the tool works without external dependencies on macOS.

    Args:
        image_path: Path to the source image file.
        target_w: Desired output grid width in pixels.
        target_h: Desired output grid height in pixels.

    Returns:
        2D list of normalized luminance floats (`[target_h][target_w]`) in `[0.0, 1.0]`.

    Raises:
        FileNotFoundError: If `image_path` does not exist.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Attempt 1: Load via Pillow (if installed)
    try:
        from PIL import Image, ImageOps

        img = Image.open(image_path).convert("L")
        img = ImageOps.exif_transpose(img)
        w, h = img.size
        # Center portrait crop (~4:5 aspect ratio)
        desired_ratio = 0.82
        cur_ratio = w / float(h)
        if cur_ratio > desired_ratio:
            new_w = int(h * desired_ratio)
            left = (w - new_w) // 2
            img = img.crop((left, 0, left + new_w, h))
        img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        pixels = list(img.getdata())
        grid = []
        for r in range(target_h):
            row = [pixels[r * target_w + c] / 255.0 for c in range(target_w)]
            grid.append(row)
        return grid
    except ImportError:
        pass

    # Attempt 2: Zero-dependency fallback via macOS `sips` -> temporary BMP
    with tempfile.NamedTemporaryFile(suffix=".bmp", delete=False) as tmp:
        tmp_bmp = tmp.name

    try:
        # Convert any supported format (JPG/PNG/HEIC) to a resized BMP along the longest edge
        subprocess.run(
            ["sips", "-s", "format", "bmp", "-Z", "600", image_path, "--out", tmp_bmp],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        raw_grid = _read_bmp_grayscale(tmp_bmp)
        return _crop_and_resize_grid(raw_grid, target_w, target_h)
    finally:
        if os.path.exists(tmp_bmp):
            os.remove(tmp_bmp)


def _read_bmp_grayscale(bmp_path: str):
    """Read an uncompressed 24-bit or 32-bit BMP file into a 2D luminance grid.

    Args:
        bmp_path: Path to the BMP file.

    Returns:
        2D list of luminance values `[height][width]` normalized to `[0.0, 1.0]`.

    Raises:
        ValueError: If the file does not start with a valid `BM` header.
    """
    with open(bmp_path, "rb") as f:
        data = f.read()

    if data[:2] != b"BM":
        raise ValueError("Unsupported BMP format")

    pixel_offset = struct.unpack_from("<I", data, 10)[0]
    width, height = struct.unpack_from("<ii", data, 18)
    bpp = struct.unpack_from("<H", data, 28)[0]

    top_down = height < 0
    height = abs(height)
    bytes_per_pixel = bpp // 8
    row_stride = ((width * bpp + 31) // 32) * 4

    grid = []
    for y in range(height):
        src_y = y if top_down else (height - 1 - y)
        row_start = pixel_offset + src_y * row_stride
        row = []
        for x in range(width):
            px_off = row_start + x * bytes_per_pixel
            b = data[px_off]
            g = data[px_off + 1]
            r = data[px_off + 2]
            # Perceptual grayscale conversion (ITU-R BT.601)
            lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
            row.append(lum)
        grid.append(row)
    return grid


def _crop_and_resize_grid(grid, target_w: int, target_h: int):
    """Center-crop a luminance grid to portrait ratio and resize via bilinear interpolation.

    Args:
        grid: Input 2D luminance grid.
        target_w: Desired output width.
        target_h: Desired output height.

    Returns:
        Resized 2D luminance grid of shape `[target_h][target_w]`.
    """
    h = len(grid)
    w = len(grid[0]) if h else 0

    # Center portrait crop (trim wide horizontal background)
    desired_ratio = 0.82
    if w / float(h) > desired_ratio:
        new_w = int(h * desired_ratio)
        x0 = (w - new_w) // 2
        grid = [row[x0 : x0 + new_w] for row in grid]
        w = new_w

    out = []
    for ty in range(target_h):
        sy = (ty + 0.5) * h / target_h - 0.5
        y0 = max(0, min(h - 1, int(math.floor(sy))))
        y1 = max(0, min(h - 1, y0 + 1))
        fy = sy - y0
        row = []
        for tx in range(target_w):
            sx = (tx + 0.5) * w / target_w - 0.5
            x0 = max(0, min(w - 1, int(math.floor(sx))))
            x1 = max(0, min(w - 1, x0 + 1))
            fx = sx - x0
            v00 = grid[y0][x0]
            v01 = grid[y0][x1]
            v10 = grid[y1][x0]
            v11 = grid[y1][x1]
            val = (1 - fy) * ((1 - fx) * v00 + fx * v01) + fy * ((1 - fx) * v10 + fx * v11)
            row.append(val)
        out.append(row)
    return out


def enhance_portrait(grid, contrast: float = 1.35, edge_boost: float = 0.45, gamma: float = 0.9):
    """Sharpen facial features for small terminal portraits.

    Applies a three-stage enhancement pipeline:
    1. Percentile histogram stretch (2nd to 98th percentile).
    2. Local edge boost via a 3x3 unsharp mask.
    3. Gamma correction and contrast scaling around midpoint `0.5`.

    Args:
        grid: Input 2D luminance grid in `[0.0, 1.0]`.
        contrast: Linear contrast multiplier applied around `0.5`.
        edge_boost: Strength of local unsharp-mask edge enhancement.
        gamma: Gamma exponent applied before contrast adjustment.

    Returns:
        Enhanced 2D luminance grid clamped to `[0.0, 1.0]`.
    """
    h = len(grid)
    w = len(grid[0])
    flat = sorted(val for row in grid for val in row)
    lo = flat[int(len(flat) * 0.02)]
    hi = flat[int(len(flat) * 0.98)]
    span = max(hi - lo, 1e-5)

    # 1. Stretch dynamic range
    norm = [[max(0.0, min(1.0, (v - lo) / span)) for v in row] for row in grid]

    # 2. Boost local edges (difference against 3x3 neighborhood mean)
    sharpened = []
    for y in range(h):
        row = []
        for x in range(w):
            c = norm[y][x]
            neighbors = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny = max(0, min(h - 1, y + dy))
                    nx = max(0, min(w - 1, x + dx))
                    neighbors.append(norm[ny][nx])
            avg = sum(neighbors) / 9.0
            val = c + edge_boost * (c - avg)
            # Apply gamma and contrast around 0.5
            val = max(0.0, min(1.0, val)) ** gamma
            val = 0.5 + (val - 0.5) * contrast
            row.append(max(0.0, min(1.0, val)))
        sharpened.append(row)
    return sharpened


# =====================================================================
# ASCII & BRAILLE RENDERING
# =====================================================================
def render_ascii(
    image_path: str,
    width: int = 38,
    style: str = "clean",
    invert: bool = False,
    contrast: float = 1.35,
    edge_boost: float = 0.45,
) -> str:
    """Render a multi-line ASCII or Unicode Braille portrait from an image file.

    Args:
        image_path: Path to the source photo.
        width: Output width in terminal columns.
        style: Rendering style (`"clean"`, `"cyber"`, `"blocks"`, or `"braille"`).
        invert: Whether to invert light and shadow tones (for light backgrounds).
        contrast: Contrast boost multiplier passed to `enhance_portrait`.
        edge_boost: Edge sharpening strength passed to `enhance_portrait`.

    Returns:
        Multi-line string containing the rendered portrait.

    Example:
        >>> art = render_ascii("photo.jpg", width=38, style="braille")
    """
    # Terminal character cells are roughly twice as tall as they are wide
    char_aspect = 0.46

    if style == "braille":
        # Each Braille character encodes a 2x4 pixel matrix
        px_w = width * 2
        px_h = int(width * char_aspect * 4)
        grid = load_grayscale_grid(image_path, px_w, px_h)
        grid = enhance_portrait(grid, contrast=contrast, edge_boost=edge_boost)
        return _grid_to_braille(grid, invert=invert)

    target_h = max(8, int(round(width * char_aspect * 1.15)))
    grid = load_grayscale_grid(image_path, width, target_h)
    grid = enhance_portrait(grid, contrast=contrast, edge_boost=edge_boost)

    palette = PALETTES.get(style, PALETTES["clean"])
    if invert:
        palette = palette[::-1]

    n_chars = len(palette)
    lines = []
    for row in grid:
        line_chars = []
        for val in row:
            idx = int(round(val * (n_chars - 1)))
            idx = max(0, min(n_chars - 1, idx))
            line_chars.append(palette[idx])
        lines.append("".join(line_chars))
    return "\n".join(lines)


def _grid_to_braille(grid, invert: bool = False) -> str:
    """Convert a pixel luminance grid into Unicode Braille characters (U+2800..U+28FF).

    Args:
        grid: 2D luminance grid with dimensions divisible by 2x4 blocks.
        invert: Whether to invert brightness before thresholding.

    Returns:
        Multi-line Braille string.
    """
    h = len(grid)
    w = len(grid[0])
    # Braille dot bitmask mapping (2 columns x 4 rows)
    dot_map = [
        (0, 0, 0x01),
        (0, 1, 0x02),
        (0, 2, 0x04),
        (1, 0, 0x08),
        (1, 1, 0x10),
        (1, 2, 0x20),
        (0, 3, 0x40),
        (1, 3, 0x80),
    ]
    lines = []
    for y in range(0, h - 3, 4):
        row_chars = []
        for x in range(0, w - 1, 2):
            mask = 0
            for dx, dy, bit in dot_map:
                val = grid[y + dy][x + dx]
                if invert:
                    val = 1.0 - val
                # Checkerboard-dithered threshold for smoother midtone depth
                threshold = 0.42 if ((x + dx + y + dy) % 2 == 0) else 0.52
                if val >= threshold:
                    mask |= bit
            row_chars.append(chr(0x2800 + mask))
        lines.append("".join(row_chars))
    return "\n".join(lines)


# =====================================================================
# SPRITE PACK GENERATION
# =====================================================================
REQUIRED_SPRITE_EMOTIONS = ("idle", "sleeping", "smiling", "talking", "talking2")
_SPRITE_ALIASES: Dict[str, tuple] = {
    "idle": ("idle", "мовчить", "спок", "neutral"),
    "sleeping": ("sleep", "спит", "спить", "afk"),
    "smiling": ("smil", "сміх", "смех", "посміш", "улыб", "laugh"),
    "talking2": ("talking2", "talk2", "говор2", "say2"),
    "talking": ("talk", "говор", "speak", "подив"),
}
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".heic", ".bmp")


def _resolve_pack_sources(input_path: str) -> Dict[str, str]:
    """Map each of the five required emotions to a source image from a folder or single file.

    Args:
        input_path: Directory of emotion photos or a single fallback image file.

    Returns:
        Dictionary mapping each emotion name in `REQUIRED_SPRITE_EMOTIONS` to an image path.

    Raises:
        FileNotFoundError: If `input_path` does not exist or contains no supported images.
    """
    if os.path.isfile(input_path):
        return {emo: input_path for emo in REQUIRED_SPRITE_EMOTIONS}

    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"--init-pack path not found: {input_path}")

    candidates = sorted(
        os.path.join(input_path, fname)
        for fname in os.listdir(input_path)
        if os.path.isfile(os.path.join(input_path, fname)) and fname.lower().endswith(_IMAGE_EXTS)
    )
    if not candidates:
        raise FileNotFoundError(f"No supported images found in --init-pack directory: {input_path}")

    resolved: Dict[str, str] = {}
    for emo in REQUIRED_SPRITE_EMOTIONS:
        aliases = _SPRITE_ALIASES[emo]
        matched = next(
            (p for p in candidates if any(al in os.path.splitext(os.path.basename(p))[0].lower() for al in aliases)),
            None,
        )
        if matched is None and emo == "talking2" and "talking" in resolved:
            matched = resolved["talking"]
        resolved[emo] = matched or candidates[0]
    return resolved


def generate_init_pack(
    input_path: str,
    out_dir: str,
    width: int = 38,
    style: str = "braille",
    contrast: float = 1.35,
    invert: bool = False,
) -> List[str]:
    """Generate all five emotion sprites (`idle`, `sleeping`, `smiling`, `talking`, `talking2`).

    Args:
        input_path: Directory of source photos or a single image file.
        out_dir: Destination directory for the generated `<emotion>.txt` files.
        width: Output portrait width in terminal columns.
        style: Rendering style (`"braille"`, `"clean"`, `"cyber"`, or `"blocks"`).
        contrast: Contrast multiplier for image enhancement.
        invert: Whether to invert light and shadow tones.

    Returns:
        List of generated sprite file paths.
    """
    sources = _resolve_pack_sources(input_path)
    os.makedirs(out_dir, exist_ok=True)
    generated_files: List[str] = []
    for emo in REQUIRED_SPRITE_EMOTIONS:
        src_img = sources[emo]
        art = render_ascii(
            src_img,
            width=width,
            style=style,
            invert=invert,
            contrast=contrast,
        )
        dst_path = os.path.join(out_dir, f"{emo}.txt")
        with open(dst_path, "w", encoding="utf-8") as f:
            f.write(art + "\n")
        generated_files.append(dst_path)
    return generated_files


# =====================================================================
# SELF-TEST & CLI
# =====================================================================
def _write_synthetic_bmp(path: str, width: int = 40, height: int = 48) -> None:
    """Create a synthetic 24-bit BMP face image for offline self-testing.

    Args:
        path: Output file path for the synthetic BMP.
        width: Image width in pixels.
        height: Image height in pixels.
    """
    row_stride = (width * 3 + 3) & ~3
    pixel_size = row_stride * height
    file_size = 54 + pixel_size
    header = struct.pack("<2sIHHI", b"BM", file_size, 0, 0, 54)
    dib = struct.pack("<IIIHHIIIIII", 40, width, height, 1, 24, 0, pixel_size, 2835, 2835, 0, 0)
    rows = bytearray()
    for y in range(height):
        row = bytearray(row_stride)
        for x in range(width):
            dx = (x - width / 2.0) / (width * 0.35)
            dy = (y - height / 2.0) / (height * 0.38)
            val = 230 if (dx * dx + dy * dy) < 1.0 else 30
            if 0.36 * height <= y <= 0.44 * height and (abs(x - 0.36 * width) < 2 or abs(x - 0.64 * width) < 2):
                val = 20
            row[x * 3 : x * 3 + 3] = bytes((val, val, val))
        rows.extend(row)
    with open(path, "wb") as f:
        f.write(header + dib + rows)


def run_self_test() -> None:
    """Run an offline self-test verifying single rendering and 5-emotion pack generation."""
    print("=====================================================================")
    print(" [*] GVK Private AI - Self-Test (tools/ascii_portrait.py)")
    print("=====================================================================")

    with tempfile.TemporaryDirectory(prefix="gvk_portrait_test_") as tmp_dir:
        sample_bmp = os.path.join(tmp_dir, "idle_face.bmp")
        _write_synthetic_bmp(sample_bmp)

        single = render_ascii(sample_bmp, width=32, style="braille")
        lines = single.splitlines()
        assert lines and len(lines[0]) == 32, f"Expected width of 32 chars, got: {len(lines[0]) if lines else 0}"

        out_pack_dir = os.path.join(tmp_dir, "portraits")
        written = generate_init_pack(tmp_dir, out_pack_dir, width=32, style="braille")
        assert len(written) == 5, f"Expected 5 sprite files, got {len(written)}"
        for emo in REQUIRED_SPRITE_EMOTIONS:
            fp = os.path.join(out_pack_dir, f"{emo}.txt")
            assert os.path.isfile(fp) and os.path.getsize(fp) > 10, f"Invalid sprite file: {fp}"
        print(" [OK] Single render + --init-pack (5 emotions: idle, sleeping, smiling, talking, talking2) passed.")

    print("=====================================================================")
    print(" [OK] ASCII PORTRAIT SELF-TEST PASSED!")
    print("=====================================================================")


def main():
    """Parse CLI arguments and render a portrait, generate an emotion pack, or run self-test."""
    parser = argparse.ArgumentParser(
        description="GVK Private AI - Convert photos into ASCII/Braille portraits for TUI"
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="Path to a source photo (.jpg, .png, .heic) or 'self-test'",
    )
    parser.add_argument(
        "-w", "--width", type=int, default=38, help="Portrait width in characters (default: 38)"
    )
    parser.add_argument(
        "-s",
        "--style",
        choices=["clean", "cyber", "blocks", "braille"],
        default="clean",
        help="Rendering style: clean (high-contrast ASCII), cyber (detailed), blocks (░▒▓█), braille (2x4 Braille dots)",
    )
    parser.add_argument("-o", "--output", help="Save rendered output to a .txt file (e.g., portraits/idle.txt)")
    parser.add_argument(
        "--init-pack",
        default=None,
        help="Directory of emotion photos or a single photo for batch-generating 5 sprites",
    )
    parser.add_argument(
        "--out-dir",
        default="./portraits/friend_1",
        help="Target output directory for --init-pack sprites (default: ./portraits/friend_1)",
    )
    parser.add_argument(
        "-c",
        "--color",
        choices=["cyan", "amber", "violet", "green", "red", "white"],
        default="cyan",
        help="Terminal preview ANSI color",
    )
    parser.add_argument("--contrast", type=float, default=1.35, help="Contrast multiplier (default: 1.35)")
    parser.add_argument("--invert", action="store_true", help="Invert light/shadow tones (for light terminal backgrounds)")

    args = parser.parse_args()

    if args.image == "self-test":
        run_self_test()
        return

    if args.init_pack:
        pack_style = "braille" if args.style == "clean" and "--style" not in sys.argv and "-s" not in sys.argv else args.style
        generated = generate_init_pack(
            input_path=args.init_pack,
            out_dir=args.out_dir,
            width=args.width,
            style=pack_style,
            contrast=args.contrast,
            invert=args.invert,
        )
        print(f"[OK] Generated {len(generated)} sprites in {os.path.abspath(args.out_dir)}:")
        for p in generated:
            print(f"    - {os.path.basename(p)}")
        return

    if not args.image:
        parser.print_help()
        sys.exit(1)

    art = render_ascii(
        args.image,
        width=args.width,
        style=args.style,
        invert=args.invert,
        contrast=args.contrast,
    )

    color_code = ANSI_COLORS.get(args.color, "")
    reset_code = ANSI_COLORS["reset"] if color_code else ""

    border = "─" * args.width
    print(f"\n{color_code}┌─ ASCII PORTRAIT PREVIEW ({args.style.upper()}) {'─' * max(0, args.width - 28)}┐")
    for line in art.splitlines():
        print(f"│{line.ljust(args.width)}│")
    print(f"└{border}┘{reset_code}\n")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(art + "\n")
        print(f"[OK] Portrait saved to: {args.output}")


if __name__ == "__main__":
    main()
