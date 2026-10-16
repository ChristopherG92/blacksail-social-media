import numpy as np, cv2, math, os, subprocess
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1080, 1920, 30
HERE = os.path.dirname(os.path.abspath(__file__))

CYAN = np.array([0.18, 0.86, 1.0], np.float32)
GREEN = np.array([0.25, 1.0, 0.55], np.float32)
AMBER = np.array([1.0, 0.72, 0.2], np.float32)
RED = np.array([1.0, 0.32, 0.36], np.float32)
WHITE = np.array([1, 1, 1], np.float32)
HUDG = np.array([0.75, 1.0, 0.8], np.float32)

FD = '/usr/share/fonts/opentype/inter/'
def F(name, size): return ImageFont.truetype(FD + name, size)
def MONO(s): return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', s)
def MONOB(s): return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', s)
def SANS(s): return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', s)
def SANSB(s): return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', s)

def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def prog(t, a, b): return clamp((t - a) / (b - a))
def eo(p): return 1 - (1 - p) ** 3
def ei(p): return p ** 3
def eio(p): return 4 * p ** 3 if p < 0.5 else 1 - (-2 * p + 2) ** 3 / 2
def bell(t, a, b, c, d): return prog(t, a, b) * (1 - prog(t, c, d))
def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t)

YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)

def glow(layer, strength=1.0, sigmas=(4, 14, 40), weights=(0.9, 0.6, 0.35)):
    small = cv2.resize(layer, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    acc = layer.copy()
    for s, w in zip(sigmas, weights):
        b = cv2.GaussianBlur(small, (0, 0), s / 4)
        acc += cv2.resize(b, (W, H), interpolation=cv2.INTER_LINEAR) * (w * strength)
    return acc

def u8(): return np.zeros((H, W, 3), np.uint8)
def c8(c, a=1.0): return tuple(int(255 * clamp(float(v) * a)) for v in c)

_tc = {}
def text_rgba(s, font, color=(1, 1, 1), tracking=0, key=None):
    k = (s, id(font), tuple(np.round(color, 3)), tracking) if key is None else key
    if k in _tc: return _tc[k]
    tw = int(sum(font.getlength(ch) for ch in s) + tracking * max(0, len(s) - 1)) + 10
    asc, desc = font.getmetrics()
    img = Image.new('L', (max(tw, 4), asc + desc + 10), 0)
    d = ImageDraw.Draw(img)
    x = 3
    if tracking:
        for ch in s:
            d.text((x, 3), ch, font=font, fill=255); x += font.getlength(ch) + tracking
    else:
        d.text((3, 3), s, font=font, fill=255)
    a = np.asarray(img, np.float32) / 255
    rgba = np.zeros(a.shape + (4,), np.float32)
    rgba[..., :3] = np.array(color, np.float32); rgba[..., 3] = a
    if len(_tc) < 4000: _tc[k] = rgba
    return rgba

def rich_rgba(parts, font):
    tw = int(sum(font.getlength(t) for t, _ in parts)) + 10
    asc, desc = font.getmetrics()
    img = Image.new('RGBA', (max(tw, 4), asc + desc + 10), (0, 0, 0, 0))
    d = ImageDraw.Draw(img); x = 3
    for t, c in parts:
        d.text((x, 3), t, font=font, fill=tuple(int(255 * clamp(float(v))) for v in c) + (255,)); x += font.getlength(t)
    return np.asarray(img, np.float32) / 255

def _place(canvas, rgba, x, y, anchor):
    h, w = rgba.shape[:2]
    if anchor[0] == 'c': x -= w / 2
    elif anchor[0] == 'r': x -= w
    if anchor[1] == 'c': y -= h / 2
    elif anchor[1] == 'b': y -= h
    x, y = int(round(x)), int(round(y))
    x0, y0 = max(0, x), max(0, y); x1, y1 = min(canvas.shape[1], x + w), min(canvas.shape[0], y + h)
    if x1 <= x0 or y1 <= y0: return None
    return (x0, y0, x1, y1), rgba[y0 - y:y1 - y, x0 - x:x1 - x]

def over(canvas, rgba, x, y, alpha=1.0, anchor='lt'):
    r = _place(canvas, rgba, x, y, anchor)
    if r is None: return
    (x0, y0, x1, y1), src = r
    a = src[..., 3:4] * alpha
    canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1] * (1 - a) + src[..., :3] * a

def add(canvas, rgba, x, y, alpha=1.0, anchor='lt'):
    r = _place(canvas, rgba, x, y, anchor)
    if r is None: return
    (x0, y0, x1, y1), src = r
    canvas[y0:y1, x0:x1] += src[..., :3] * src[..., 3:4] * alpha

def scale_about(img, s, cx, cy, tx=None, ty=None, border=cv2.BORDER_REFLECT):
    tx = cx if tx is None else tx; ty = cy if ty is None else ty
    M = np.float32([[s, 0, tx - s * cx], [0, s, ty - s * cy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=border)

def zoom_blur(img, s, cx, cy, amount, n=6, tx=None, ty=None):
    if amount < 0.004: return scale_about(img, s, cx, cy, tx, ty)
    acc = np.zeros_like(img)
    for k in range(n):
        acc += scale_about(img, s * (1 - amount * k / (n - 1)), cx, cy, tx, ty)
    return acc / n

def chroma(img, px=2):
    if px <= 0: return img
    out = img.copy()
    out[:, px:, 0] = img[:, :-px, 0]; out[:, :-px, 2] = img[:, px:, 2]
    return out

def grain(img, f, amt=0.02):
    r = np.random.default_rng(1000 + f)
    g = r.normal(0, 1, (H // 2, W // 2)).astype(np.float32)
    return img + cv2.resize(g, (W, H))[..., None] * amt

VIGN = (1 - 0.6 * np.clip(np.sqrt(((XX - W / 2) / (W * 0.72)) ** 2 + ((YY - H / 2) / (H * 0.72)) ** 2), 0, 1) ** 2.2)[..., None]

def glitch(img, f, amt):
    if amt <= 0: return img
    r = np.random.default_rng(f * 31 + 7)
    out = img.copy()
    for _ in range(int(6 + 16 * amt)):
        y = int(r.integers(0, H)); h = int(r.integers(6, 80)); dx = int(r.normal(0, 70 * amt))
        out[y:y + h] = np.roll(out[y:y + h], dx, axis=1)
    return chroma(out, max(1, int(10 * amt)))

def bg_dark():
    g = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    bg = np.array([0.012, 0.022, 0.045], np.float32) * (1 - g) + np.array([0.004, 0.010, 0.022], np.float32) * g
    bg = np.repeat(bg, W, axis=1)
    r = np.sqrt(((XX - W / 2) / W) ** 2 + ((YY - H * 0.45) / H) ** 2)
    bg += (np.exp(-r * r / 0.05) * 0.06)[..., None] * np.array([0.1, 0.35, 0.6], np.float32)
    return bg.astype(np.float32)
BG = bg_dark()

def caption(img, text, t, t0, t1, y=1560, size=86, color=(1, 1, 1), sub=None):
    """Big kinetic caption: words rise in with blur-free stagger, fade out."""
    if t < t0 or t > t1 + 0.35: return
    font = F('InterDisplay-Bold.otf', size)
    while font.getlength(text) > 960 and size > 40:
        size -= 4; font = F('InterDisplay-Bold.otf', size)
    words = text.split(' ')
    widths = [font.getlength(w + ' ') for w in words]
    total = sum(widths) - font.getlength(' ')
    x = W / 2 - total / 2
    out = 1 - prog(t, t1, t1 + 0.35)
    # dark backing gradient for legibility
    band = np.exp(-((YY[:, :1] - (y + size * 0.6)) / (size * 1.6)) ** 2)[..., None][:, :, :1]
    img *= 1 - 0.45 * band * out * prog(t, t0, t0 + 0.3)
    for i, w in enumerate(words):
        p = eo(prog(t, t0 + i * 0.09, t0 + i * 0.09 + 0.45))
        if p > 0:
            spr = text_rgba(w, font, color)
            over(img, spr, x, y + (1 - p) * 40, alpha=p * out)
        x += widths[i]
    if sub:
        p = eo(prog(t, t0 + 0.35, t0 + 0.8))
        s2 = text_rgba(sub, F('Inter-SemiBold.otf', 34), (0.55, 0.85, 1.0), tracking=7)
        over(img, s2, W / 2, y + size * 1.25 + (1 - p) * 20, alpha=p * out, anchor='ct')

class Writer:
    def __init__(self, path):
        self.p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                                   '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
                                   '-pix_fmt', 'yuv420p', path], stdin=subprocess.PIPE)
    def write(self, img):
        self.p.stdin.write((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())
    def close(self):
        self.p.stdin.close(); self.p.wait()
