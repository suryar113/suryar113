"""Make the ASCII art for the left side of the card.

    python tools/make_ascii.py --image me.png --mask mask.png --crop 450,150,920,700 --rotate 9.5
    python tools/make_ascii.py --text SR

--text draws the letters as a slanted, extruded 3D monogram. It writes ascii.txt and
ascii_colors.txt, which tells today.py what to colour: the letter faces in the card's key
colour (orange in dark mode) and the 3D sides in its dim grey. Both themes use the same art.

--image writes ascii.txt (dark card: bright pixels get dense characters) and
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

Needs Pillow and NumPy (pip install pillow numpy); photos also need OpenCV (pip install opencv-python).
Any grid size works: the card scales the art to fit, growing taller if it has many rows.
"""
import argparse
import glob
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageStat

ROOT = Path(__file__).resolve().parent.parent
CELL_W, CELL_H = 9.64, 20  # one character cell on the card, in px


def find_font(patterns):
    for pattern in patterns:
        pattern, index = pattern if isinstance(pattern, tuple) else (pattern, None)
        hits = glob.glob(pattern, recursive=True)
        if hits:
            return hits[0] if index is None else (hits[0], index)
    raise SystemExit("No suitable font found; pass --font /path/to/font.ttf")


# (path, index in a font collection): a heavy italic, so the monogram leans forward.
HEAVY_FONTS = (("/System/Library/Fonts/Avenir Next Condensed.ttc", 9),  # Avenir Next Condensed Heavy Italic
               ("/usr/share/fonts/**/DejaVuSans-BoldOblique.ttf", 0),
               ("C:/Windows/Fonts/arialbi.ttf", 0))
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


# Monogram looks: the letter faces run from dense at the top to lighter at the bottom, and the
# 3D sides are a hatch of slashes that follows the slant.
FACE_BANDS = "@@@@###%%%"
FACE_TINTS = "kkkkkkkddd"  # colour per band: k = key (orange), d = the card's red, for a sunset fade
SIDE_HATCH = "/"
EDGE_CHARS = " _-=^~`'\".,:;!|/\\()[]{}<>@#%*+"  # punctuation only, so edges never spell words


def glyph_grid(font_path, chars):
    """Each glyph as a 3x5 grid of ink coverage, measured in the card's monospace font."""
    import numpy as np

    font = ImageFont.truetype(font_path, 20)
    grids = []
    for c in chars:
        im = Image.new("L", (12, 25), 0)
        ImageDraw.Draw(im).text((0, 19), c, fill=255, font=font, anchor="ls")
        grids.append(np.asarray(im.resize((3, 5), Image.BOX), np.float32).ravel() / 255)
    return np.stack(grids)


def best_glyphs(target, cols, rows, chars, glyphs):
    """For each cell of `target` (a 0..1 array, 1 = densest), the glyph whose 3x5 ink pattern
    matches it best: shape first (so strokes land where the ink is: _ low, - middle, / and \\
    on diagonals), then density. Returns a rows x cols array of characters."""
    import numpy as np

    gmean = glyphs.mean(1)
    small = np.asarray(Image.fromarray(target.astype(np.float32)).resize((cols * 3, rows * 5), Image.BOX))
    cells = small.reshape(rows, 5, cols, 3).transpose(0, 2, 1, 3).reshape(rows * cols, 15) * gmean.max()
    pc, gc = cells - cells.mean(1, keepdims=True), glyphs - gmean[:, None]
    shape = (pc ** 2).sum(1)[:, None] - 2 * pc @ gc.T + (gc ** 2).sum(1)[None]
    tone = (cells.mean(1)[:, None] - gmean[None]) ** 2 * 15
    return np.array(chars)[(shape + 4 * tone).argmin(1)].reshape(rows, cols)


def monogram(letters, cols, rows, font, mono_font, depth=(3, 2)):
    """A 3D monogram: (art lines, colour lines). `depth` is the extrusion in cells (right, down).
    Colour codes: k = key (orange), d = red, c = 3D side (dim), space = default."""
    import numpy as np

    cw, ch = 12, 25  # pixels per cell while drawing, close to the card's 0.48 cell aspect
    w, h = cols * cw, rows * ch
    dx, dy = depth[0] * cw, depth[1] * ch
    margin = cw
    path, index = font
    # The largest size whose letters, plus their extrusion, fit the box.
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    size = 10
    while True:
        l, t, r, b = probe.textbbox((0, 0), letters, font=ImageFont.truetype(path, size + 4, index=index))
        if r - l + dx > w - 2 * margin or b - t + dy > h - 2 * margin:
            break
        size += 4
    f = ImageFont.truetype(path, size, index=index)
    l, t, r, b = probe.textbbox((0, 0), letters, font=f)
    ox = (w - (r - l + dx)) / 2 - l
    oy = (h - (b - t + dy)) / 2 - t

    face, side = Image.new("L", (w, h), 0), Image.new("L", (w, h), 0)
    ImageDraw.Draw(face).text((ox, oy), letters, fill=255, font=f)
    steps = int(max(dx, dy))
    for k in range(steps, 0, -1):
        ImageDraw.Draw(side).text((ox + dx * k / steps, oy + dy * k / steps), letters, fill=255, font=f)
    face = np.asarray(face, np.float32) / 255
    side = np.clip(np.asarray(side, np.float32) / 255 - face, 0, 1)

    cell = lambda a: np.asarray(Image.fromarray(a).resize((cols, rows), Image.BOX))
    face_cov, side_cov = cell(face), cell(side)
    edges = best_glyphs(face + 0.45 * side, cols, rows, list(EDGE_CHARS), glyph_grid(mono_font, EDGE_CHARS))
    used = np.nonzero(face_cov.max(1) > 0.5)[0]
    top, bottom = used.min(), used.max()
    art, colours = [], []
    for y in range(rows):
        b = min(len(FACE_BANDS) - 1, max(0, (y - top) * len(FACE_BANDS) // (bottom - top + 1)))
        band, tint = FACE_BANDS[b], FACE_TINTS[b]
        line, code = "", ""
        for x in range(cols):
            fc, sc = face_cov[y, x], side_cov[y, x]
            if fc > 0.9:
                c = band
            elif sc > 0.9 and fc < 0.05:
                c = SIDE_HATCH
            else:
                c = edges[y, x]
                if c in "@#%&$" and fc > 0.5:
                    c = band  # a dense edge cell joins its row's band
            line += c
            code += " " if c == " " else (tint if fc >= sc and fc > 0.15 else "c" if sc > 0.15 else " ")
        art.append(line.rstrip())
        colours.append(code[:len(line.rstrip())])
    # Trim empty rows and the shared left margin; the card scales the art to fit anyway.
    keep = [i for i, line in enumerate(art) if line.strip()]
    art, colours = art[keep[0]:keep[-1] + 1], colours[keep[0]:keep[-1] + 1]
    left = min(len(line) - len(line.lstrip()) for line in art if line.strip())
    return [line[left:] for line in art], [code[left:] for code in colours]


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

    chars = list(EDGE_CHARS)  # punctuation only: letters here would read as stray words on the logo
    target = np.asarray(img, np.float32) / 255
    if invert:
        target = 1 - target
    picked_all = best_glyphs(target, cols, rows, chars, glyph_grid(font_path, chars))
    inside = np.asarray(region.resize((cols, rows), Image.BOX)) > 127
    picked = {(x, y): picked_all[y, x] for y, x in zip(*np.nonzero(inside))}
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
    src.add_argument("--text", help="letters for a 3D monogram, e.g. SR")
    src.add_argument("--image", help="photo to convert")
    ap.add_argument("--mask", help="white-on-black cutout of the subject (see tools/mask.swift)")
    ap.add_argument("--crop", help="left,top,right,bottom box in photo pixels")
    ap.add_argument("--cols", type=int, help="default: 100 for photos, 92 for --text")
    ap.add_argument("--rows", type=int, help="default: 56 for photos, 40 for --text")
    ap.add_argument("--depth", default="3,2", help="--text extrusion in cells, right,down")
    ap.add_argument("--detail", type=float, default=1.5, help="how strongly to sharpen features")
    ap.add_argument("--rotate", type=float, default=0.0, help="degrees counter-clockwise, to level the head")
    ap.add_argument("--crisp", help="left,top,right,bottom box of printed lettering, e.g. a cap logo")
    ap.add_argument("--label", action="append", default=[], help='"TEXT@x,y" written into the art')
    ap.add_argument("--font", help="a heavy font for --text")
    ap.add_argument("--font-index", type=int, default=0, help="which face of a .ttc --font to use")
    ap.add_argument("--mono-font", help="the monospace font the card is shown in, for measuring glyphs")
    args = ap.parse_args()
    mono = args.mono_font or find_font(MONO_FONTS)

    if args.text:
        cols, rows = args.cols or 92, args.rows or 40
        font = (args.font, args.font_index) if args.font else find_font(HEAVY_FONTS)
        art, colours = monogram(args.text, cols, rows, font, mono, tuple(int(v) for v in args.depth.split(",")))
        (ROOT / "ascii.txt").write_text("\n".join(art) + "\n")
        (ROOT / "ascii_colors.txt").write_text("\n".join(colours) + "\n")
        (ROOT / "ascii_light.txt").unlink(missing_ok=True)  # the same art reads on both cards
    else:
        args.cols, args.rows = args.cols or 100, args.rows or 56
        (ROOT / "ascii_colors.txt").unlink(missing_ok=True)
        box = lambda v: tuple(int(n) for n in v.split(",")) if v else None
        img, mask, region, to_cell = portrait(args.image, args.mask, box(args.crop), args.cols, args.rows,
                                     args.detail, args.rotate, box(args.crisp))
        font = mono
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
