from common import *

FOC = 820.0
VPX, VPY = W / 2, H * 0.47
XW = 1.25          # wall half-width
YF, YT, YC = -0.85, 1.75, 2.1   # floor, rack top, ceiling
PPU = 220          # texture px per world unit
SPAN = 60.0        # texture length in world units (wraps)
NEAR, FAR = 0.3, 34.0

def _rack_textures(seed):
    r = np.random.default_rng(seed)
    tw, th = int(SPAN * PPU), int((YT - YF) * PPU)
    base = np.zeros((th, tw, 3), np.float32)
    leds = []  # (x, y, color, rate, phase)
    rack_w = 0.62
    x = 0.0
    while x < SPAN:
        x0 = int(x * PPU); x1 = int((x + rack_w) * PPU)
        # frame
        base[:, x0:x1] = (0.035, 0.04, 0.05)
        base[:, x0:x0 + 6] = (0.11, 0.12, 0.14); base[:, x1 - 6:x1] = (0.11, 0.12, 0.14)
        # units
        y = 14
        while y < th - 20:
            uh = int(r.choice([10, 10, 20, 20, 30, 40]))
            if y + uh > th - 14: break
            shade = r.uniform(0.05, 0.13)
            base[y:y + uh - 2, x0 + 10:x1 - 10] = (shade, shade * 1.05, shade * 1.15)
            # vents
            if uh >= 20 and r.random() < 0.6:
                for vx in range(x0 + 30, x1 - 40, 7):
                    base[y + 4:y + uh - 6, vx:vx + 3] *= 0.45
            # handle highlight
            base[y:y + 2, x0 + 10:x1 - 10] += 0.05
            nl = int(r.integers(1, 5))
            for k in range(nl):
                c = r.choice(4, p=[0.55, 0.3, 0.1, 0.05])
                colr = [(0.2, 1.0, 0.45), (0.25, 0.75, 1.0), (1.0, 0.7, 0.2), (1.0, 0.25, 0.3)][c]
                leds.append((x1 - 22 - k * 13, y + uh // 2, colr, r.uniform(1.5, 14), r.uniform(0, 6.28), r.random() < 0.7))
            y += uh
        x += rack_w + 0.04
    # glass door reflection streaks
    gx = np.linspace(0, 1, tw, dtype=np.float32)
    streak = (0.5 + 0.5 * np.sin(gx * 260)) ** 30 * 0.05
    base += streak[None, :, None]
    return base, leds

BASE_L, LEDS_L = _rack_textures(1)
BASE_R, LEDS_R = _rack_textures(2)
TH, TW = BASE_L.shape[:2]

def led_tex(leds, t):
    L = np.zeros((TH, TW, 3), np.float32)
    for (x, y, c, rate, ph, blink) in leds:
        on = 1.0 if not blink else (0.25 + 0.75 * (math.sin(t * rate + ph) > -0.2))
        L[y - 2:y + 3, x - 3:x + 4] = np.array(c, np.float32) * on * 1.6
    return L

def proj(X, Y, Z, sway=(0, 0)):
    return (VPX + FOC * (X - sway[0]) / Z, VPY - FOC * (Y - sway[1]) / Z)

def wall_quad(side, cz, sway):
    X = -XW if side < 0 else XW
    zs = (NEAR, FAR)
    pts = [proj(X, YT, zs[0], sway), proj(X, YT, zs[1], sway), proj(X, YF, zs[1], sway), proj(X, YF, zs[0], sway)]
    if side > 0:  # right wall: texture runs near->far left->right as well
        pass
    return np.float32(pts)

def warp_plane(tex, cz, dst):
    u0 = ((cz + NEAR) * PPU) % TW
    u1 = u0 + (FAR - NEAR) * PPU
    # slice with wrap
    i0 = int(u0); i1 = int(u1) + 2
    if i1 <= TW:
        sl = tex[:, i0:i1]
    else:
        sl = np.concatenate([tex[:, i0:], tex[:, :i1 - TW]], 1)
    du = u0 - i0
    src = np.float32([[du, 0], [du + (u1 - u0), 0], [du + (u1 - u0), sl.shape[0]], [du, sl.shape[0]]])
    Hm = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(sl, Hm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def _floor_tex():
    th, tw = 400, int(SPAN * PPU / 4)
    t = np.full((th, tw, 3), 0.02, np.float32)
    for x in range(0, tw, int(0.6 * PPU / 4)):
        t[:, x:x + 2] = 0.06
    for y in range(0, th, 66):
        t[y:y + 2] = 0.06
    t[:18] = (0.05, 0.4, 0.6); t[-18:] = (0.05, 0.4, 0.6)
    return t
FLOOR = _floor_tex()
def _ceil_tex():
    th, tw = 300, int(SPAN * PPU / 4)
    t = np.full((th, tw, 3), 0.015, np.float32)
    for x in range(0, tw, int(1.6 * PPU / 4)):
        t[120:180, x:x + int(0.5 * PPU / 4)] = (0.45, 0.55, 0.7)
    return t
CEIL = _ceil_tex()

def warp_hplane(tex, cz, Y, sway, scale=4):
    u0 = ((cz + NEAR) * PPU / scale) % tex.shape[1]
    span = (FAR - NEAR) * PPU / scale
    i0 = int(u0); i1 = int(u0 + span) + 2
    sl = tex[:, i0:i1] if i1 <= tex.shape[1] else np.concatenate([tex[:, i0:], tex[:, :i1 - tex.shape[1]]], 1)
    du = u0 - i0; hh = sl.shape[0]
    # texture rows: 0 -> left wall (X=-XW), hh -> right wall
    src = np.float32([[du, 0], [du + span, 0], [du + span, hh], [du, hh]])
    dst = np.float32([proj(-XW, Y, NEAR, sway), proj(-XW, Y, FAR, sway), proj(XW, Y, FAR, sway), proj(XW, Y, NEAR, sway)])
    Hm = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(sl, Hm, (W, H), flags=cv2.INTER_LINEAR)

# depth estimate per pixel (for fog)
def depth_map(sway):
    vx = VPX - FOC * sway[0]; vy = VPY + FOC * sway[1]
    dx = np.abs(XX - vx) / XW; dy_up = (vy - YY) / (YC - sway[1]); dy_dn = (YY - vy) / (sway[1] - YF)
    d = np.maximum(dx, np.maximum(dy_up, dy_dn))
    return FOC / np.maximum(d, 1e-3)

PACK = np.random.default_rng(3).uniform([-0.9, 1.3, 0], [0.9, 1.9, 30], (40, 3))

def corridor(t, speed=2.4, intensity=1.0):
    cz = t * speed
    sway = (0.06 * math.sin(t * 0.9), 0.04 * math.sin(t * 0.7 + 1))
    img = np.zeros((H, W, 3), np.float32)
    # ceiling & floor
    img += warp_hplane(CEIL, cz, YC, sway)
    fl = warp_hplane(FLOOR, cz, YF, sway)
    img += fl
    # gap between rack top and ceiling: dark
    for side, base, leds in [(-1, BASE_L, LEDS_L), (1, BASE_R, LEDS_R)]:
        X = -XW if side < 0 else XW
        dst = np.float32([proj(X, YT, NEAR, sway), proj(X, YT, FAR, sway), proj(X, YF, FAR, sway), proj(X, YF, NEAR, sway)])
        wb = warp_plane(base, cz, dst)
        wl = warp_plane(led_tex(leds, t), cz, dst)
        m = (wb.max(2, keepdims=True) > 0.001)
        img = np.where(m, wb, img)
        img += wl
        # reflection on floor (mirror about Y=YF)
        dstr = np.float32([proj(X, 2 * YF - YT, NEAR, sway), proj(X, 2 * YF - YT, FAR, sway), proj(X, YF, FAR, sway), proj(X, YF, NEAR, sway)])
        rb = warp_plane(base * 0.25 + led_tex(leds, t) * 0.5, cz, dstr)
        img += cv2.GaussianBlur(rb, (0, 0), 6) * 0.5
    # data packets streaking toward camera
    P = u8()
    for (px, py, pz0) in PACK:
        z = (pz0 - cz * 1.8) % 30 + 0.4
        z2 = z + 0.9
        a = proj(px * 0.8, py, z, sway); b = proj(px * 0.8, py, z2, sway)
        br = clamp(1.2 - z / 30)
        cv2.line(P, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), c8(CYAN, br), max(1, int(4 / z * 2)), cv2.LINE_AA)
    img += glow(P.astype(np.float32) / 255, 1.2) * 0.9
    # fog / depth
    D = depth_map(sway)
    fog = 1 - np.exp(-D / 16.0)
    fogc = np.array([0.03, 0.09, 0.16], np.float32)
    img = img * (1 - fog[..., None]) + fogc * fog[..., None]
    # end-of-aisle light
    vx = VPX - FOC * sway[0]; vy = VPY + FOC * sway[1]
    r2 = ((XX - vx) / 60) ** 2 + ((YY - vy) / 90) ** 2
    img += (np.exp(-r2) * 1.2 + np.exp(-r2 / 30) * 0.18)[..., None] * np.array([0.6, 0.85, 1.0], np.float32)
    img = glow(img * intensity, 0.35, sigmas=(6, 24), weights=(0.5, 0.35))
    # DOF: blur near edges
    blur = cv2.GaussianBlur(img, (0, 0), 7)
    edge = np.clip((np.abs(XX - W / 2) / (W / 2)) ** 3 * 1.2, 0, 1)[..., None]
    img = img * (1 - edge) + blur * edge
    return img
