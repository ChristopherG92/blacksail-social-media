from common import *
def make_card():
    w, h = 1000, 630
    img = Image.new('RGBA', (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
    # base gradient
    g = np.zeros((h, w, 3), np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g[:] = (0.05 + 0.06 * (xx / w) + 0.03 * (1 - yy / h))[..., None] * np.array([1, 1.05, 1.15], np.float32)
    # brushed noise
    r = np.random.default_rng(3)
    n = cv2.GaussianBlur(r.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigmaX=12, sigmaY=0.6) * 0.012
    g += n[..., None]
    # sail-like diagonal accent
    acc = np.clip(1 - np.abs((xx - 0.62 * w) + (yy - h) * 0.55) / 260, 0, 1) ** 3 * 0.10
    g += acc[..., None] * np.array([0.2, 0.6, 1.0], np.float32)
    base = Image.fromarray((np.clip(g, 0, 1) * 255).astype(np.uint8)).convert('RGBA')
    mask = Image.new('L', (w, h), 0); ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), 48, fill=255)
    img.paste(base, (0, 0), mask); d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, w - 1, h - 1), 48, outline=(90, 110, 130, 255), width=2)
    # chip
    d.rounded_rectangle((90, 220, 230, 320), 16, fill=(196, 166, 98, 255))
    for k in range(3): d.line((90, 245 + k * 25, 230, 245 + k * 25), fill=(150, 120, 60, 255), width=3)
    d.line((160, 220, 160, 320), fill=(150, 120, 60, 255), width=3)
    # contactless
    for k in range(4):
        rr = 18 + k * 16
        d.arc((300 - rr, 270 - rr, 300 + rr, 270 + rr), -50, 50, fill=(220, 230, 240, 255), width=6)
    d.text((90, 420), '••••  ••••  ••••  4821', font=MONO(48), fill=(225, 232, 240, 255))
    d.text((90, 510), 'BUSINESS DEBIT', font=F('Inter-SemiBold.otf', 30), fill=(150, 170, 190, 255))
    return np.asarray(img, np.float32) / 255
CARD = make_card()

def draw_card(img, cx, cy, scale, ry, rx, rz=0.0, spec=0.0):
    h, w = CARD.shape[:2]
    f = 1600
    pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    out = []
    for (x, y) in pts:
        X, Y, Z = (x - w / 2) * scale, (y - h / 2) * scale, 0.0
        X, Y = X * math.cos(rz) - Y * math.sin(rz), X * math.sin(rz) + Y * math.cos(rz)
        X, Z = X * math.cos(ry) + Z * math.sin(ry), -X * math.sin(ry) + Z * math.cos(ry)
        Y, Z = Y * math.cos(rx) - Z * math.sin(rx), Y * math.sin(rx) + Z * math.cos(rx)
        out.append([cx + f * X / (f + Z), cy + f * Y / (f + Z)])
    Hm = cv2.getPerspectiveTransform(pts, np.float32(out))
    lay = cv2.warpPerspective(CARD, Hm, (W, H), flags=cv2.INTER_LINEAR)
    a = lay[..., 3:4]
    rgb = lay[..., :3]
    if spec > 0:
        band = np.exp(-(((XX - cx) * 0.8 + (YY - cy) * 0.6 - spec) / 90) ** 2)[..., None]
        rgb = rgb + band * 0.35 * a
    # soft shadow
    sh = cv2.GaussianBlur(a, (0, 0), 30)[..., None] if a.ndim == 2 else cv2.GaussianBlur(a[..., 0], (0, 0), 30)[..., None]
    sh = np.roll(sh, 50, axis=0)
    img[:] = img * (1 - 0.6 * sh) 
    img[:] = img * (1 - a) + rgb * a
    return out
