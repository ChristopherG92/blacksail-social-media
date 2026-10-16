from common import *
_im = cv2.imread(os.path.join(HERE, 'blacksail_logo.png'))
_l = cv2.cvtColor(_im, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
_a = np.clip((_l - 0.18) / 0.6, 0, 1)
ys, xs = np.where(_a > 0.5)
_a = _a[ys.min() - 6:ys.max() + 7, xs.min() - 6:xs.max() + 7]
LW = 900
_a = cv2.resize(_a, (LW, int(_a.shape[0] * LW / _a.shape[1])), interpolation=cv2.INTER_AREA)
LOGO = np.dstack([np.ones_like(_a)] * 3 + [_a]).astype(np.float32)
LH = LOGO.shape[0]

def end_logo(u):
    img = BG.copy() * 0.6
    x0 = W // 2 - LW // 2; y0 = H // 2 - LH // 2
    s = 1.06 - 0.06 * eo(prog(u, 0, 1.6))
    lay = np.zeros((H, W, 4), np.float32)
    lay[y0:y0 + LH, x0:x0 + LW] = LOGO
    # left-to-right reveal
    rv = eo(prog(u, 0.05, 0.75))
    edge = x0 - 40 + (LW + 80) * rv
    m = np.clip((edge - XX) / 40, 0, 1)
    a = lay[..., 3] * m
    a = scale_about(a, s, W / 2, H / 2, border=cv2.BORDER_CONSTANT)
    rgb = np.ones((H, W, 3), np.float32) * a[..., None]
    # reveal edge flare + light sweep
    flare = np.exp(-((XX - edge) / 18) ** 2) * (np.abs(YY - H / 2) < LH * 0.7) * (rv < 1)
    sw = prog(u, 1.0, 1.9)
    band = np.exp(-(((XX + (YY - H / 2) * 0.35) - (x0 - 200 + sw * (LW + 400))) / 60) ** 2) * (0 < sw < 1)
    lit = rgb * (1 + band[..., None] * 1.2)
    img = img * (1 - a[..., None]) + lit
    img += glow(rgb * 0.35 + flare[..., None] * CYAN * 0.8, 0.9)
    return img
