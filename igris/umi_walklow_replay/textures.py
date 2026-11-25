"""Procedural textures for the UMI replay scene (Pillow only)."""
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[2]
EXISTING = ROOT / 'assets/room_real2sim/textures'
FONTS = ['/usr/share/fonts/truetype/nanum/NanumSquareRoundB.ttf',
         '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf',
         '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc']
KRAFT = (186, 142, 96)
RED = (176, 30, 44)
DARK_RED = (150, 28, 36)


def font(size):
    for path in FONTS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise FileNotFoundError('A Korean TrueType font is required for the parcel print')


def cardboard(width, height, seed, base=KRAFT):
    """Kraft liner: fibre noise plus faint corrugation lines."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, (height // 4 + 1, width // 4 + 1))
    noise = np.asarray(Image.fromarray(noise).resize((width, height), Image.BICUBIC))
    fine = rng.normal(0, 1, (height, width))
    flute = 1.2 * np.sin(np.arange(height)[:, None] * 2 * np.pi / 9.)
    value = 5 * noise + 3 * fine + flute
    image = np.clip(np.array(base, dtype=float)[None, None] + value[..., None] * np.array([1., .9, .75]), 0, 255)
    return Image.fromarray(image.astype(np.uint8)).filter(ImageFilter.GaussianBlur(.4))


def qr(draw, x, y, size, seed):
    rng = np.random.default_rng(seed)
    cells = 17; unit = size / cells
    draw.rectangle([x - 3, y - 3, x + size + 3, y + size + 3], fill=(240, 236, 228))
    pattern = rng.random((cells, cells)) < .5
    for cx, cy in ((0, 0), (cells - 5, 0), (0, cells - 5)):
        pattern[cy:cy + 5, cx:cx + 5] = True
        pattern[cy + 1:cy + 4, cx + 1:cx + 4] = False
        pattern[cy + 2, cx + 2] = True
    for i in range(cells):
        for j in range(cells):
            if pattern[i, j]:
                draw.rectangle([x + j * unit, y + i * unit, x + (j + 1) * unit, y + (i + 1) * unit], fill=(35, 30, 28))


def parcel_long(path, seed=0):
    """27 x 15 cm print face of a Korea Post no. 2 parcel box."""
    width, height = 1080, 600
    image = cardboard(width, height, seed)
    draw = ImageDraw.Draw(image)
    band = int(height * .60)
    draw.rectangle([0, band, width, height], fill=RED)
    # Yellow/orange ribbon on top of the red band.
    for i, colour in enumerate(((226, 150, 44), (236, 178, 70))):
        points = [(x, band - 6 - 14 * i + 10 * np.sin(x / width * np.pi * 1.2 + .4)) for x in range(0, width + 1, 8)]
        draw.line(points, fill=colour, width=10 - 3 * i)
    draw.text((330, 70), '마음까지 전하는', font=font(34), fill=DARK_RED)
    draw.line([(610, 105), (800, 105)], fill=DARK_RED, width=3)
    draw.text((250, 130), '우체국', font=font(118), fill=DARK_RED)
    draw.text((590, 96), '소포', font=font(176), fill=RED)
    draw.text((70, band + 34), '전국 어디서나', font=font(40), fill=(245, 236, 226))
    draw.text((70, band + 96), '1588-1300', font=font(96), fill=(250, 244, 236))
    draw.text((720, band + 176), 'www.epost.kr', font=font(34), fill=(245, 230, 220))
    qr(draw, 900, band + 40, 120, seed + 11)
    image.filter(ImageFilter.GaussianBlur(.6)).save(path)


def parcel_short(path, seed=1):
    """18 x 15 cm end face: plain liner with the red band wrapping around."""
    width, height = 720, 600
    image = cardboard(width, height, seed)
    draw = ImageDraw.Draw(image)
    band = int(height * .60)
    draw.rectangle([0, band, width, height], fill=RED)
    draw.line([(0, band - 8), (width, band - 8)], fill=(226, 150, 44), width=9)
    draw.text((150, 120), '우체국 소포', font=font(84), fill=DARK_RED)
    draw.text((200, band + 80), '2호 (27×18×15)', font=font(44), fill=(245, 236, 226))
    image.filter(ImageFilter.GaussianBlur(.6)).save(path)


def parcel_top(path, seed=2):
    """27 x 18 cm top: flap seam, clear tape and the printed address form."""
    width, height = 1080, 720
    image = cardboard(width, height, seed)
    draw = ImageDraw.Draw(image, 'RGBA')
    draw.line([(0, height // 2), (width, height // 2)], fill=(110, 78, 48, 255), width=4)
    draw.rectangle([0, height // 2 - 96, width, height // 2 + 96], fill=(214, 190, 150, 90))     # clear tape
    draw.rectangle([0, height // 2 - 96, width, height // 2 - 92], fill=(235, 220, 190, 150))
    draw.rectangle([0, height // 2 + 92, width, height // 2 + 96], fill=(235, 220, 190, 150))
    for x0, title in ((50, '보내는 분'), (560, '받는 분')):
        draw.rectangle([x0, 40, x0 + 470, 230], outline=(170, 40, 46, 255), width=4)
        draw.text((x0 + 14, 50), title, font=font(30), fill=(170, 40, 46, 255))
        for k in range(3):
            draw.line([(x0 + 20, 120 + 34 * k), (x0 + 450, 120 + 34 * k)], fill=(170, 40, 46, 200), width=2)
    for i in range(5):
        draw.rectangle([70 + 62 * i, 560, 118 + 62 * i, 620], outline=(170, 40, 46, 255), width=3)
    draw.text((600, 560), '취급주의', font=font(44), fill=(170, 40, 46, 255))
    image.convert('RGB').filter(ImageFilter.GaussianBlur(.6)).save(path)


def delivery_label(path):
    image = Image.new('RGB', (480, 300), (244, 243, 238))
    draw = ImageDraw.Draw(image)
    draw.rectangle([4, 4, 475, 295], outline=(40, 40, 40), width=6)
    draw.text((52, 52), 'DELIVERY', font=font(72), fill=(25, 25, 25))
    draw.text((52, 140), 'LAB', font=font(72), fill=(25, 25, 25))
    draw.line([(250, 216), (430, 216)], fill=(25, 25, 25), width=8)
    image.save(path)


def case_laminate(path, seed=5):
    """Black ABS laminate of a flight case: fine hexagonal emboss."""
    rng = np.random.default_rng(seed)
    size = 512
    yy, xx = np.mgrid[:size, :size]
    emboss = (np.sin(xx * 2 * np.pi / 16) * np.sin((yy + (xx // 16 % 2) * 8) * 2 * np.pi / 16)) * 5
    value = 22 + emboss + rng.normal(0, 1.6, (size, size))
    image = np.clip(np.stack([value, value, value * 1.03], axis=-1), 0, 255).astype(np.uint8)
    Image.fromarray(image).save(path)


def case_label(path):
    image = Image.new('RGB', (900, 240), (20, 20, 21))
    draw = ImageDraw.Draw(image)
    draw.text((60, 40), 'IGRIS-C', font=font(150), fill=(232, 232, 228))
    image.save(path)


def brushed_aluminium(path, seed=7):
    rng = np.random.default_rng(seed)
    streak = rng.normal(0, 1, (256, 1)) * 7
    value = 178 + streak + rng.normal(0, 2.2, (256, 256))
    image = np.clip(np.stack([value, value * 1.005, value * 1.01], axis=-1), 0, 255).astype(np.uint8)
    Image.fromarray(image).save(path)


def tag(path, seed=9):
    """AprilTag-like fiducial board, as taped to the roll container."""
    rng = np.random.default_rng(seed)
    cells = 8; unit = 60
    image = Image.new('RGB', (cells * unit + 2 * unit, cells * unit + 2 * unit), (245, 245, 242))
    draw = ImageDraw.Draw(image)
    bits = rng.random((cells, cells)) < .5
    bits[0, :] = bits[-1, :] = bits[:, 0] = bits[:, -1] = True
    for i in range(cells):
        for j in range(cells):
            if bits[i, j]:
                draw.rectangle([unit + j * unit, unit + i * unit, 2 * unit + j * unit, 2 * unit + i * unit], fill=(18, 18, 18))
    image.save(path)


def build(out):
    """Write every texture the scene references into ``out/textures``."""
    target = Path(out) / 'textures'
    target.mkdir(parents=True, exist_ok=True)
    for name in ('kraft', 'tape', 'zinc', 'paper_normal', 'green'):
        shutil.copyfile(EXISTING / f'{name}.png', target / f'{name}.png')
    parcel_long(target / 'parcel_long.png'); parcel_short(target / 'parcel_short.png'); parcel_top(target / 'parcel_top.png')
    cardboard(512, 512, 21).save(target / 'parcel_bottom.png')
    delivery_label(target / 'delivery_label.png')
    case_laminate(target / 'case_laminate.png'); case_label(target / 'case_label.png')
    brushed_aluminium(target / 'aluminium.png'); tag(target / 'tag.png')
    return target
