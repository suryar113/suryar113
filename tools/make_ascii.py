"""Make the ASCII art for the left side of the card.

    python tools/make_ascii.py --image me.png --mask mask.png --crop 300,70,1060,830
    python tools/make_ascii.py --text SR

Writes ascii.txt (dark card: bright pixels get dense characters) and, for photos,
ascii_light.txt (light card: dark pixels get dense characters, so it isn't a negative).

--mask is a white-on-black cutout of you; anything outside it becomes blank space, which
keeps a busy background out of the art. On a Mac, make one with
    swift tools/mask.swift me.png mask.png
--crop is the left,top,right,bottom box to use; it is widened to the card's art aspect.

Needs Pillow (pip install pillow); photos also need OpenCV (pip install opencv-python).
Any grid size works: the card scales the art to fit, growing taller if it has many rows.
"""
import argparse
import glob
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
RAMP = " .:-=+*#%@"
CELL_W, CELL_H = 9.64, 20  # one character cell on the card, in px


def find_font():
    for pattern in (
        "/System/Library/Fonts/Supplemental/Arial Black.ttf",
        "/usr/share/fonts/**/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/ariblk.ttf",
    ):
        hits = glob.glob(pattern, recursive=True)
        if hits:
            return hits[0]
    raise SystemExit("No bold font found; pass --font /path/to/font.ttf")


def monogram(letters, cols, rows, font_path):
    w, h = round(cols * CELL_W * 2.5), rows * CELL_H * 5 // 2
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    slot = h / len(letters)

    def fitted(ch):
        size = 10
        while True:
            font = ImageFont.truetype(font_path, size + 5)
            l, t, r, b = draw.textbbox((0, 0), ch, font=font)
            if r - l > w * 0.9 or b - t > slot * 0.86:
                return ImageFont.truetype(font_path, size)
            size += 5

    for i, ch in enumerate(letters):
        font = fitted(ch)
        l, t, r, b = draw.textbbox((0, 0), ch, font=font)
        draw.text((w / 2 - (r - l) / 2 - l, slot * (i + 0.5) - (b - t) / 2 - t), ch, fill=255, font=font)
    return img, img


def portrait(path, mask_path, crop, cols, rows, gamma=0.85, clip=3.0, edge=1.0):
    """A float image in 0..1 (brighter = denser on the dark card) and a subject mask."""
    import cv2
    import numpy as np

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, 0) if mask_path else np.full(img.shape[:2], 255, np.uint8)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    left, top, right, bottom = crop or (0, 0, img.shape[1], img.shape[0])
    # Widen (or heighten) the box around its centre to match the art's aspect ratio.
    aspect = cols * CELL_W / (rows * CELL_H)
    h = bottom - top
    w = h * aspect
    if w < right - left:
        w = right - left
        h = w / aspect
    cx, cy = (left + right) / 2, (top + bottom) / 2
    x0, y0, x1, y1 = (round(v) for v in (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    img, mask = img[y0:y1, x0:x1], mask[y0:y1, x0:x1]

    lum = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0]
    subject = mask > 128
    # Local contrast (CLAHE) pulls detail out of dark skin and under a cap brim. The background
    # is flattened first so the cutout's edge isn't mistaken for contrast.
    flat = lum.copy()
    flat[~subject] = int(np.median(lum[subject]))
    tone = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(flat).astype(np.float32) / 255
    tone = tone ** gamma
    # A difference of Gaussians darkens fine outlines (eyes, nostrils, lips) and lightens flats.
    g = lum.astype(np.float32) / 255
    dog = cv2.GaussianBlur(g, (0, 0), 3) - cv2.GaussianBlur(g, (0, 0), 9)
    tone = np.clip(tone - edge * 6 * dog, 0, 1)
    return Image.fromarray((tone * 255).astype("uint8")), Image.fromarray(mask)


def to_ascii(img, mask, cols, rows, invert=False):
    px = img.resize((cols, rows), Image.BOX).load()
    mk = mask.resize((cols, rows), Image.BOX).load()
    lines = []
    for y in range(rows):
        line = ""
        for x in range(cols):
            if mk[x, y] < 110:
                line += " "
                continue
            v = px[x, y] / 255
            if invert:
                v = 1 - v
            line += RAMP[min(len(RAMP) - 1, 1 + int(v * (len(RAMP) - 1)))]
        lines.append(line.rstrip())
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="letters to stack vertically, e.g. SR")
    src.add_argument("--image", help="photo to convert")
    ap.add_argument("--mask", help="white-on-black cutout of the subject (see tools/mask.swift)")
    ap.add_argument("--crop", help="left,top,right,bottom box in photo pixels")
    ap.add_argument("--cols", type=int, default=100)
    ap.add_argument("--rows", type=int, default=56)
    ap.add_argument("--font")
    args = ap.parse_args()

    if args.text:
        img, mask = monogram(args.text, args.cols, args.rows, args.font or find_font())
        (ROOT / "ascii.txt").write_text(to_ascii(img, mask, args.cols, args.rows))
        (ROOT / "ascii_light.txt").unlink(missing_ok=True)  # a monogram reads the same on both
    else:
        crop = tuple(int(v) for v in args.crop.split(",")) if args.crop else None
        img, mask = portrait(args.image, args.mask, crop, args.cols, args.rows)
        (ROOT / "ascii.txt").write_text(to_ascii(img, mask, args.cols, args.rows))
        (ROOT / "ascii_light.txt").write_text(to_ascii(img, mask, args.cols, args.rows, invert=True))
    print((ROOT / "ascii.txt").read_text())
