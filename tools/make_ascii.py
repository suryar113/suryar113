"""Make the ASCII art for the left side of the card.

    python tools/make_ascii.py --image me.png --mask mask.png --crop 450,150,920,700 --rotate 9.5
    python tools/make_ascii.py --text SR

Writes ascii.txt (dark card: bright pixels get dense characters) and, for photos,
ascii_light.txt (light card: dark pixels get dense characters, so it isn't a negative).

--mask is a white-on-black cutout of you; anything outside it becomes blank space, which
keeps a busy background out of the art. On a Mac, make one with
    swift tools/mask.swift me.png mask.png
--crop is the left,top,right,bottom box to use; it is widened to the card's art aspect.
--rotate turns the photo counter-clockwise by that many degrees around the crop's centre,
to straighten a tilted head (negative turns it clockwise).
--crisp is a left,top,right,bottom box (in the original photo) holding printed lettering, such
as a logo on a cap. Inside it the photo is split into ink and background, and characters are
picked by shape instead of shading, so the letters stay sharp.
--label "TEXT@x,y" writes TEXT into the art, centred on that point of the original photo, for
lettering too small to survive as shapes (repeatable).

Needs Pillow (pip install pillow); photos also need OpenCV and NumPy (pip install opencv-python).
Any grid size works: the card scales the art to fit, growing taller if it has many rows.
"""
import argparse
import glob
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageStat

ROOT = Path(__file__).resolve().parent.parent
RAMP = " .:-=+*#%@"
CELL_W, CELL_H = 9.64, 20  # one character cell on the card, in px


def find_font(patterns):
    for pattern in patterns:
        hits = glob.glob(pattern, recursive=True)
        if hits:
            return hits[0]
    raise SystemExit("No suitable font found; pass --font /path/to/font.ttf")


BOLD_FONTS = ("/System/Library/Fonts/Supplemental/Arial Black.ttf",
              "/usr/share/fonts/**/DejaVuSans-Bold.ttf", "C:/Windows/Fonts/ariblk.ttf")
MONO_FONTS = ("/System/Library/Fonts/Menlo.ttc", "/usr/share/fonts/**/DejaVuSansMono.ttf",
              "C:/Windows/Fonts/consola.ttf")


def measured_ramp(font_path, steps=24):
    """Characters whose ink coverage, measured in the card's monospace font, is evenly spaced
    from blank to densest. A long, even ramp keeps skin tones from collapsing into one symbol.
    Returns (chars, coverage in 0..1)."""
    font = ImageFont.truetype(font_path, 20)
    cov = {}
    for code in range(32, 127):
        im = Image.new("L", (12, 25), 0)
        ImageDraw.Draw(im).text((0, 19), chr(code), fill=255, font=font, anchor="ls")
        cov[chr(code)] = ImageStat.Stat(im).mean[0] / 255
    top = max(cov.values())
    chars, levels = [], []
    for i in range(steps):
        target = i / (steps - 1)
        ch = min(cov, key=lambda c: abs(cov[c] / top - target))
        if ch not in chars:
            chars.append(ch)
            levels.append(cov[ch] / top)
    return chars, levels


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


def portrait(path, mask_path, crop, cols, rows, detail=1.5, rotate=0.0, crisp=None):
    """A greyscale image (brighter = denser on the dark card), a subject mask, a mask of the
    --crisp box, and a function mapping a point of the original photo to (column, row)."""
    import cv2
    import numpy as np

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, 0) if mask_path else np.full(img.shape[:2], 255, np.uint8)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    region = np.zeros(img.shape[:2], np.uint8)
    if crisp:
        region[crisp[1]:crisp[3], crisp[0]:crisp[2]] = 255
    left, top, right, bottom = crop or (0, 0, img.shape[1], img.shape[0])
    turn = np.float32([[1, 0, 0], [0, 1, 0]])
    if rotate:
        turn = cv2.getRotationMatrix2D(((left + right) / 2, (top + bottom) / 2), rotate, 1.0)
        size = (img.shape[1], img.shape[0])
        img = cv2.warpAffine(img, turn, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        mask = cv2.warpAffine(mask, turn, size, flags=cv2.INTER_LINEAR, borderValue=0)
        region = cv2.warpAffine(region, turn, size, flags=cv2.INTER_NEAREST, borderValue=0)
    # Widen (or heighten) the box around its centre to match the art's aspect ratio.
    aspect = cols * CELL_W / (rows * CELL_H)
    h = bottom - top
    w = h * aspect
    if w < right - left:
        w = right - left
        h = w / aspect
    cx, cy = (left + right) / 2, (top + bottom) / 2
    x0, y0, x1, y1 = (round(v) for v in (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    img, mask, region = img[y0:y1, x0:x1], mask[y0:y1, x0:x1], region[y0:y1, x0:x1]

    lum = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32) / 255
    subject = mask > 128
    lum[~subject] = np.median(lum[subject])  # so the cutout's edge isn't mistaken for detail
    # Split the photo into broad shading and fine detail. The broad shading is replaced by
    # its rank within the subject (histogram equalisation), so a bright cap can't use up the
    # whole range and leave the face in a narrow band. The fine detail (nose, lips, jaw,
    # the sunglasses' frame) is then added back on top, amplified.
    s = 0.012 * img.shape[1]
    base = cv2.GaussianBlur(lum, (0, 0), s)
    fine = lum - cv2.GaussianBlur(lum, (0, 0), 4 * s)
    ranked = np.sort(base[subject])
    tone = np.searchsorted(ranked, base).astype(np.float32) / len(ranked) + detail * fine
    tone = np.clip(tone, 0, 1)
    inside = region > 0
    if inside.any():
        # Lettering: a clean two-level split (Otsu's threshold) of the original brightness.
        raw = (lum * 255).astype(np.uint8)
        cut, _ = cv2.threshold(raw[inside], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ink = cv2.GaussianBlur((raw > cut).astype(np.float32), (0, 0), 0.7)
        tone[inside] = ink[inside]
    as_image = lambda a: Image.fromarray(a.astype("uint8"))

    def to_cell(px, py):
        px, py = turn @ np.float32([px, py, 1])
        return (px - x0) / (x1 - x0) * cols, (py - y0) / (y1 - y0) * rows

    return as_image(tone * 255), as_image(mask), as_image(region), to_cell


def shape_match(img, region, cols, rows, font_path, invert=False):
    """{(x, y): char} for the cells inside `region`, each picked by comparing a 3x5 grid of the
    cell with the same grid of every glyph, so strokes land where the ink is (_ low, - middle,
    / and \\ on diagonals) and not just at the right density."""
    import numpy as np

    sx, sy, cw, ch = 3, 5, 12, 25
    font = ImageFont.truetype(font_path, 20)
    # Punctuation only: letters and digits here would read as stray words on the logo.
    chars = list(" _-=^~`'\".,:;!|/\\()[]{}<>@#%*+")
    glyphs = []
    for c in chars:
        im = Image.new("L", (cw, ch), 0)
        ImageDraw.Draw(im).text((0, 19), c, fill=255, font=font, anchor="ls")
        glyphs.append(np.asarray(im.resize((sx, sy), Image.BOX), np.float32).ravel() / 255)
    glyphs = np.stack(glyphs)
    gmean = glyphs.mean(1)
    target = np.asarray(img.resize((cols * sx, rows * sy), Image.BOX), np.float32) / 255
    if invert:
        target = 1 - target
    cells = target.reshape(rows, sy, cols, sx).transpose(0, 2, 1, 3).reshape(rows, cols, sx * sy)
    cells = cells * gmean.max()  # full brightness = the densest glyph
    inside = np.asarray(region.resize((cols, rows), Image.BOX)) > 127
    picked = {}
    for y, x in zip(*np.nonzero(inside)):
        p = cells[y, x]
        shape = (((p - p.mean()) - (glyphs - gmean[:, None])) ** 2).sum(1)
        tone = (p.mean() - gmean) ** 2 * sx * sy
        picked[(x, y)] = chars[int((shape + 4 * tone).argmin())]
    return picked


def to_ascii(img, mask, cols, rows, chars, levels, invert=False, dither=0.0, fixed=None):
    """One character per cell, picked by tone. With dither > 0, each cell's rounding error is
    passed on to its neighbours (Floyd-Steinberg), which keeps soft gradients like skin.
    Cells in `fixed` ({(x, y): char}, from shape_match) are used as given."""
    px = img.resize((cols, rows), Image.BOX).load()
    mk = mask.resize((cols, rows), Image.BOX).load()
    v = [[(1 - px[x, y] / 255) if invert else px[x, y] / 255 for x in range(cols)] for y in range(rows)]
    lines = []
    for y in range(rows):
        line = ""
        for x in range(cols):
            if mk[x, y] < 110:
                line += " "
                continue
            if fixed and (x, y) in fixed:
                line += fixed[(x, y)]
                continue
            i = min(range(len(levels)), key=lambda k: abs(levels[k] - v[y][x]))
            line += chars[i]
            err = (v[y][x] - levels[i]) * dither
            for dx, dy, wt in ((1, 0, 7), (-1, 1, 3), (0, 1, 5), (1, 1, 1)):
                if 0 <= x + dx < cols and y + dy < rows:
                    v[y + dy][x + dx] += err * wt / 16
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
    ap.add_argument("--detail", type=float, default=1.5, help="how strongly to sharpen features")
    ap.add_argument("--rotate", type=float, default=0.0, help="degrees counter-clockwise, to level the head")
    ap.add_argument("--crisp", help="left,top,right,bottom box of printed lettering, e.g. a cap logo")
    ap.add_argument("--label", action="append", default=[], help='"TEXT@x,y" written into the art')
    ap.add_argument("--font", help="a bold font (--text) or a monospace font (--image)")
    args = ap.parse_args()

    if args.text:
        img, mask = monogram(args.text, args.cols, args.rows, args.font or find_font(BOLD_FONTS))
        levels = [i / (len(RAMP) - 1) for i in range(len(RAMP))]
        (ROOT / "ascii.txt").write_text(to_ascii(img, mask, args.cols, args.rows, RAMP, levels))
        (ROOT / "ascii_light.txt").unlink(missing_ok=True)  # a monogram reads the same on both
    else:
        box = lambda v: tuple(int(n) for n in v.split(",")) if v else None
        img, mask, region, to_cell = portrait(args.image, args.mask, box(args.crop), args.cols, args.rows,
                                     args.detail, args.rotate, box(args.crisp))
        font = args.font or find_font(MONO_FONTS)
        chars, levels = measured_ramp(font)
        for name, invert in (("ascii.txt", False), ("ascii_light.txt", True)):
            fixed = shape_match(img, region, args.cols, args.rows, font, invert)
            for label in args.label:
                text, at = label.rsplit("@", 1)
                cx, cy = to_cell(*box(at))
                start = round(cx - len(text) / 2)
                for i, c in enumerate(f" {text} "):  # a blank on each side sets the word apart
                    fixed[(start - 1 + i, int(cy))] = c
            art = to_ascii(img, mask, args.cols, args.rows, chars, levels, invert, dither=0.8, fixed=fixed)
            (ROOT / name).write_text(art)
    print((ROOT / "ascii.txt").read_text())
