"""Blacksail static social templates (PIL). Renders at 1200x1500 (4:5) master; export helpers downscale.
Brand: deep navy, cyan #2EDBFF, green #40FF8C, amber for exceptions, Inter type.
v1 - 2026-10-12 (Week 42): cover / sign / checklist / end-card / blueprint layouts.
"""
import os, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
FD = '/usr/share/fonts/opentype/inter/'
def F(n, s): return ImageFont.truetype(FD + n, s)
def MONO(s): return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', s)
def MONOB(s): return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', s)

CYAN = (46, 219, 255); GREEN = (64, 255, 140); AMBER = (255, 184, 51); RED = (255, 82, 92)
WHITE = (255, 255, 255); MUTED = (150, 172, 198); DIM = (92, 112, 138)
PANEL = (10, 22, 40); LINE = (34, 66, 98)

LOGO_PATH = os.environ.get('BS_LOGO', os.path.join(HERE, '..', 'video_toolkit', 'blacksail_logo.png'))

def logo_white(width):
    im = Image.open(LOGO_PATH).convert('RGB')
    g = np.asarray(im.convert('L'), np.float32) / 255
    a = np.clip((g - 0.18) / 0.6, 0, 1)
    ys, xs = np.where(a > 0.5)
    a = a[ys.min() - 6:ys.max() + 7, xs.min() - 6:xs.max() + 7]
    al = Image.fromarray((a * 255).astype(np.uint8))
    h = int(al.height * width / al.width)
    al = al.resize((width, h), Image.LANCZOS)
    out = Image.new('RGBA', (width, h), WHITE + (0,)); out.putalpha(al)
    return out

def background(w, h, seed=0, glow_xy=(0.75, 0.18), grid=True):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = yy / h
    top = np.array([7, 15, 30], np.float32); bot = np.array([3, 7, 16], np.float32)
    img = top * (1 - g[..., None]) + bot * g[..., None]
    gx, gy = glow_xy
    r = np.sqrt(((xx - gx * w) / w) ** 2 + ((yy - gy * h) / h) ** 2)
    img += np.exp(-r * r / 0.08)[..., None] * np.array([8, 40, 70], np.float32)
    r2 = np.sqrt(((xx - 0.1 * w) / w) ** 2 + ((yy - 0.95 * h) / h) ** 2)
    img += np.exp(-r2 * r2 / 0.05)[..., None] * np.array([4, 30, 26], np.float32)
    rng = np.random.default_rng(seed)
    img += rng.normal(0, 2.2, (h, w, 1))
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    if grid:
        d = ImageDraw.Draw(im)
        for x in range(60, w, 60):
            for y in range(60, h, 60):
                d.point((x, y), fill=(28, 52, 80))
    return im.convert('RGBA')

def text_w(d, s, f): return d.textlength(s, font=f)

def wrap(d, text, font, maxw):
    out = []
    for para in text.split('\n'):
        words, line = para.split(' '), ''
        for w_ in words:
            t = (line + ' ' + w_).strip()
            if text_w(d, t, font) <= maxw: line = t
            else: out.append(line); line = w_
        out.append(line)
    return out

def draw_lines(d, x, y, lines, font, fill, lh):
    for ln in lines:
        d.text((x, y), ln, font=font, fill=fill); y += lh
    return y

def rich_line(d, x, y, parts, font):
    for s, c in parts:
        d.text((x, y), s, font=font, fill=c); x += text_w(d, s, font)

def glow_layer(base, draw_fn, radius=18, strength=1.0):
    L = Image.new('RGBA', base.size, (0, 0, 0, 0)); draw_fn(ImageDraw.Draw(L))
    b = L.filter(ImageFilter.GaussianBlur(radius))
    if strength != 1.0:
        a = np.asarray(b).astype(np.float32); a[..., 3] *= strength; b = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    base.alpha_composite(b); base.alpha_composite(L)

def chrome(im, kicker, page=None, total=None, footer=True, kicker_col=CYAN):
    w, h = im.size; d = ImageDraw.Draw(im); M = int(w * 0.075)
    d.text((M, M), kicker, font=F('Inter-SemiBold.otf', int(w * 0.022)), fill=kicker_col)
    if page:
        s = f'{page:02d} / {total:02d}'; f = F('Inter-Medium.otf', int(w * 0.02))
        d.text((w - M - text_w(d, s, f), M), s, font=f, fill=DIM)
    if footer:
        lg = logo_white(int(w * 0.17)); im.alpha_composite(lg, (M, h - M - lg.height + 6))
        s = 'blacksailco.com'; f = F('Inter-Medium.otf', int(w * 0.02))
        d.text((w - M - text_w(d, s, f), h - M - int(w * 0.024)), s, font=f, fill=DIM)

def export(im, path, size):
    im.convert('RGB').resize(size, Image.LANCZOS).save(path, quality=95)
