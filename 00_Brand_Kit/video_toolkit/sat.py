from common import *

LEVELS = [11, 12, 13, 14, 15, 16, 17, 18, 19]
REF = 19
TGT = {11: (800.3, 1063.1), 12: (799.7, 1102.2), 13: (800.3, 1180.4), 14: (799.7, 1080.7), 15: (800.4, 1137.5), 16: (799.8, 1251.0), 17: (799.6, 1221.9), 18: (800.1, 1163.8), 19: (800.2, 1047.6)}
FOCUS18 = np.array([800.2, 1047.6])
LAT0, LON0 = -33.8927, 18.506701
X0_18 = 310

def _feather(h, w, f):
    m = np.ones((h, w), np.float32)
    ramp = np.linspace(0, 1, f, dtype=np.float32)
    m[:f] *= ramp[:, None]; m[-f:] *= ramp[::-1][:, None]
    m[:, :f] *= ramp[None]; m[:, -f:] *= ramp[None, ::-1]
    return m ** 1.5

PYR = {}
_prev = None
for l in LEVELS:
    im = cv2.imread(os.path.join(HERE, 'satellite', f'sat2_z{l}.png'))[:, :, ::-1].astype(np.float32) / 255
    if _prev is not None:
        pl, pim = _prev
        k = 2 ** (l - pl)
        c = np.array(TGT[pl]); hh, ww = im.shape[:2]
        x0 = int(c[0] - TGT[l][0] / k); y0 = int(c[1] - TGT[l][1] / k)
        reg = pim[max(0, y0):y0 + int(hh / k), max(0, x0):x0 + int(ww / k)]
        small = cv2.resize(im, (reg.shape[1], reg.shape[0]), interpolation=cv2.INTER_AREA)
        for ch in range(3):
            ms, ss = small[..., ch].mean(), small[..., ch].std() + 1e-4
            mr, sr = reg[..., ch].mean(), reg[..., ch].std() + 1e-4
            adj = (im[..., ch] - ms) * (sr / ss) + mr
            im[..., ch] = 0.3 * im[..., ch] + 0.7 * adj
        im = np.clip(im, 0, 1)
    _prev = (l, im.copy())
    # detail enhancement: local contrast (CLAHE on L) + two-scale unsharp
    lab = cv2.cvtColor((im * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    cl = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    lab[..., 0] = cl.apply(lab[..., 0])
    im2 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255
    im = im * 0.45 + im2 * 0.55
    for sg, amt in ((0.9, 0.55), (2.5, 0.25)):
        bl = cv2.GaussianBlur(im, (0, 0), sg)
        im = np.clip(im + (im - bl) * amt, 0, 1)
    m = _feather(im.shape[0], im.shape[1], 340)
    valid = cv2.GaussianBlur((im.max(2) > 0.035).astype(np.float32), (0, 0), 6)
    valid = cv2.erode(valid, np.ones((9, 9), np.uint8))
    m = m * valid
    lv = []
    for j in range(3):
        rgba = np.dstack([im, m]).astype(np.float32)
        lv.append(rgba)
        im = cv2.resize(im, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        m = cv2.resize(m, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    PYR[l] = lv

def cam_level(cam18, l):
    return np.array(TGT[l]) + (np.asarray(cam18) - np.array(TGT[REF])) / 2 ** (REF - l)

def sat_view(Z, cam18, theta=0.0, cx=W / 2, cy=H / 2):
    R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    corners = np.array([[0, 0], [W, 0], [0, H], [W, H]], np.float64) - [cx, cy]
    # finest level that fully covers the screen
    start = 0
    for i, l in enumerate(LEVELS):
        s = 2 ** (Z - l)
        c = cam_level(cam18, l)
        pts = c + (corners @ R) / s  # inverse rotation: R^T applied => row @ R
        hh, ww = PYR[l][0].shape[:2]
        if (pts[:, 0].min() > 345 and pts[:, 0].max() < ww - 345 and pts[:, 1].min() > 345 and pts[:, 1].max() < hh - 345):
            start = i
    out = np.zeros((H, W, 3), np.float32)
    for l in LEVELS[start:]:
        s = 2 ** (Z - l)
        if s < 0.12: break
        j = 0 if s >= 0.7 else (1 if s >= 0.35 else 2)
        src = PYR[l][j]
        se = s * 2 ** j
        c = cam_level(cam18, l) / 2 ** j
        A = se * R
        off = np.array([cx, cy]) - A @ c
        M = np.float32([[A[0, 0], A[0, 1], off[0]], [A[1, 0], A[1, 1], off[1]]])
        lay = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        a = lay[..., 3:4]
        out = lay[..., :3].copy() if l == LEVELS[start] else out * (1 - a) + lay[..., :3] * a
    return out

def pix_to_latlon(cam18):
    n = 2 ** REF
    fx = (LON0 + 180) / 360 * n
    lr = math.radians(LAT0)
    fy = (1 - math.log(math.tan(lr) + 1 / math.cos(lr)) / math.pi) / 2 * n
    x0 = math.floor(fx - 4); y0 = math.floor(fy - 4)
    wx = (x0 * 256 + X0_18 + cam18[0]) / 256; wy = (y0 * 256 + cam18[1]) / 256
    lon = wx / n * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * wy / n))))
    return lat, lon

def dms(v, pos, neg):
    h = pos if v >= 0 else neg; v = abs(v)
    d = int(v); m = int((v - d) * 60); s = (v - d - m / 60) * 3600
    return f"{d:02d}°{m:02d}'{s:04.1f}\"{h}"

# ---------- clouds ----------
def make_clouds(seed, size=1400):
    r = np.random.default_rng(seed)
    acc = np.zeros((size, size), np.float32)
    for k, amp in [(6, 1.0), (12, 0.5), (24, 0.25), (48, 0.13), (96, 0.07)]:
        n = r.normal(0, 1, (k, k)).astype(np.float32)
        acc += cv2.resize(n, (size, size), interpolation=cv2.INTER_CUBIC) * amp
    acc = (acc - acc.mean()) / acc.std()
    a = np.clip((acc - 0.35) * 0.9, 0, 1) ** 1.2
    a = cv2.GaussianBlur(a, (0, 0), 3)
    shade = np.clip(1 - 0.25 * cv2.GaussianBlur(a, (0, 0), 25), 0.6, 1)
    rgba = np.dstack([shade, shade, shade * 1.02, a]).astype(np.float32)
    return rgba
CLOUDS = [make_clouds(5), make_clouds(11)]

def draw_clouds(img, Z, theta, layers=((12.35, 0, (0, 0)), (13.1, 1, (250, -180)))):
    for zc, ci, (ox, oy) in layers:
        s = 2 ** (Z - zc) * 1.4
        if s < 0.25 or s > 9: continue
        a_ = bell(Z, zc - 1.6, zc - 0.7, zc + 0.15, zc + 0.75) * 0.62
        if a_ <= 0.01: continue
        src = CLOUDS[ci]
        c = np.array([700 + ox, 700 + oy], np.float64)
        Rm = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]) * s
        off = np.array([W / 2, H / 2]) - Rm @ c
        M = np.float32([[Rm[0, 0], Rm[0, 1], off[0]], [Rm[1, 0], Rm[1, 1], off[1]]])
        lay = cv2.warpAffine(src, M, (W, H), borderMode=cv2.BORDER_WRAP)
        al = lay[..., 3:4] * a_
        if s > 2.5:
            k = int(min(25, (s - 2.5) * 6)) * 2 + 1
            al = cv2.GaussianBlur(al, (k, k), 0)[..., None] if al.ndim == 3 else al
        img[:] = img * (1 - al) + lay[..., :3] * al

# ---------- grading ----------
def sensor_grade(img, Z):
    haze = clamp((14.2 - Z) / 3.2) * 0.32
    img = img * (1 - haze) + haze * np.array([0.55, 0.68, 0.85], np.float32)
    img = np.clip(img, 0, 1)
    # filmic contrast + teal/orange split
    img = img ** 1.08
    l = img.mean(2, keepdims=True)
    img = img + (l < 0.35) * (0.35 - l) * np.array([-0.02, 0.01, 0.05], np.float32) * 0.6
    return img

def night_vision(img, f):
    l = (img[..., 0] * 0.3 + img[..., 1] * 0.59 + img[..., 2] * 0.11)
    l = np.clip((l - 0.08) * 1.6, 0, 1) ** 0.8
    nv = l[..., None] * np.array([0.55, 1.0, 0.6], np.float32)
    nv = grain(nv, f, 0.05)
    return nv

# ---------- HUD ----------
HUDF = MONO(27)
HUDFS = MONO(22)
HUDFB = MONOB(30)

def hud(img, t, Z, cam18, theta, lockp=0.0, mode='DESCENT', alpha=1.0, f=0):
    L = u8(); col = c8(HUDG, 0.9)
    m = 56; ln = 90
    for (x, y, dx, dy) in [(m, m, 1, 1), (W - m, m, -1, 1), (m, H - m, 1, -1), (W - m, H - m, -1, -1)]:
        cv2.line(L, (x, y), (x + dx * ln, y), col, 3, cv2.LINE_AA)
        cv2.line(L, (x, y), (x, y + dy * ln), col, 3, cv2.LINE_AA)
    cx, cy = W // 2, H // 2
    g = 34
    for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        cv2.line(L, (cx + dx * g, cy + dy * g), (cx + dx * 150, cy + dy * 150), col, 2, cv2.LINE_AA)
    cv2.circle(L, (cx, cy), 4, col, -1, cv2.LINE_AA)
    rr = 230
    rot = t * 25
    for k in range(4):
        cv2.ellipse(L, (cx, cy), (rr, rr), rot + k * 90, 8, 62, c8(HUDG, 0.6), 2, cv2.LINE_AA)
    for k in range(72):
        a = math.radians(k * 5 + theta * 57.3)
        r0 = 265 if k % 6 else 250
        cv2.line(L, (int(cx + r0 * math.cos(a)), int(cy + r0 * math.sin(a))),
                 (int(cx + 275 * math.cos(a)), int(cy + 275 * math.sin(a))), c8(HUDG, 0.45), 1, cv2.LINE_AA)
    # altitude ladder (right)
    alt_km = 160 * 2 ** (11 - Z)
    lx = W - 110
    for k in range(-12, 13):
        y = int(cy + k * 38 + ((Z * 6) % 1) * 38)
        if abs(y - cy) > 470: continue
        a_ = 1 - abs(y - cy) / 480
        cv2.line(L, (lx, y), (lx + (26 if k % 3 == 0 else 14), y), c8(HUDG, a_ * 0.8), 2, cv2.LINE_AA)
    cv2.line(L, (lx - 20, cy), (lx - 4, cy), col, 3, cv2.LINE_AA)
    # scale bar
    mpp = 156543.03 * math.cos(math.radians(LAT0)) / 2 ** Z
    target_m = mpp * 220
    nice = 10 ** math.floor(math.log10(target_m))
    for k in (1, 2, 5, 10):
        if nice * k >= target_m * 0.6: nice *= k; break
    blen = int(nice / mpp)
    by = H - 150
    cv2.line(L, (80, by), (80 + blen, by), col, 3, cv2.LINE_AA)
    cv2.line(L, (80, by - 12), (80, by + 2), col, 3, cv2.LINE_AA)
    cv2.line(L, (80 + blen, by - 12), (80 + blen, by + 2), col, 3, cv2.LINE_AA)
    # lock box
    if lockp > 0:
        p = eo(lockp)
        half = int(520 - 330 * p)
        cl = c8(GREEN if lockp >= 1 else AMBER, 1.0)
        bl = 46
        for (sx, sy) in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            x, y = cx + sx * half, cy + sy * half
            cv2.line(L, (x, y), (x - sx * bl, y), cl, 4, cv2.LINE_AA)
            cv2.line(L, (x, y), (x, y - sy * bl), cl, 4, cv2.LINE_AA)
    lay = L.astype(np.float32) / 255
    img += glow(lay, 0.5) * alpha
    # text
    lat, lon = pix_to_latlon(cam18)
    tl = ['BLACKSAIL // ORBITAL RECON', f'MODE  {mode}', f'LAT  {dms(lat, "N", "S")}', f'LON  {dms(lon, "E", "W")}']
    for i, s in enumerate(tl):
        over(img, text_rgba(s, HUDFB if i == 0 else HUDF, HUDG), 80, 82 + i * 40, alpha=alpha * 0.95)
    alt = f'ALT  {alt_km:7.2f} KM' if alt_km >= 1 else f'ALT  {alt_km*1000:7.0f} M'
    tr = [alt, f'ZOOM {Z:5.2f}', f'GSD  {mpp:6.2f} M/PX', f'HDG  {(-theta*57.3)%360:05.1f}°']
    for i, s in enumerate(tr):
        over(img, text_rgba(s, HUDF, HUDG), W - 80, 82 + 40 + i * 40, alpha=alpha * 0.95, anchor='rt')
    blink = int(t * 2) % 2 == 0
    if blink:
        D = u8(); cv2.circle(D, (W - 220, 100), 10, (255, 50, 60), -1, cv2.LINE_AA); img += glow(D.astype(np.float32) / 255, 0.6) * alpha
    over(img, text_rgba('REC', HUDFB, (1, 0.35, 0.4)), W - 80, 82, alpha=alpha, anchor='rt')
    sl = f'{int(nice)} M' if nice < 1000 else f'{nice/1000:g} KM'
    over(img, text_rgba(sl, HUDF, HUDG), 80, by - 52, alpha=alpha)
    secs = 9 * 3600 + 41 * 60 + 7 + t
    ts = f'2026-10-06  {int(secs//3600):02d}:{int(secs%3600//60):02d}:{secs%60:05.2f}Z'
    over(img, text_rgba(ts, HUDF, HUDG), 80, H - 120, alpha=alpha * 0.9)
    over(img, text_rgba('IMAGERY © ESRI, MAXAR, EARTHSTAR GEOGRAPHICS', HUDFS, (0.7, 0.8, 0.75)), W / 2, H - 78, alpha=alpha * 0.7, anchor='ct')
    if lockp > 0:
        p = eo(lockp)
        half = int(520 - 330 * p)
        lab = 'TARGET LOCKED' if lockp >= 1 else 'ACQUIRING…'
        colr = GREEN if lockp >= 1 else AMBER
        over(img, text_rgba(lab, HUDFB, colr), cx - half, cy - half - 50, alpha=alpha)
        if lockp >= 1:
            over(img, text_rgba('THE FORUM · CENTURY CITY · CAPE TOWN', HUDF, HUDG), cx - half, cy + half + 18, alpha=alpha)

def scanlines(img, amt=0.05):
    s = (0.5 + 0.5 * np.sin(YY[:, :1] * math.pi / 2)) [..., None][:, :, :1]
    return img * (1 - amt * s)
