import sys
from common import *
from globe import *
from card import draw_card
from logo import end_logo
FL = np.array([0.7, 0.9, 1.0], np.float32)
TOT = 14.5
LOGF = MONO(26)
HOPS = [  # (t0, from, to, colour, log text)
    (4.0, 'CPT', 'JNB', CYAN, ('+038 ms', 'PSP · JOHANNESBURG', 'tokenised · routed')),
    (4.75, 'JNB', 'LON', CYAN, ('+112 ms', 'CARD NETWORK · LONDON', 'auth request')),
    (5.5, 'LON', 'JNB', GREEN, ('+164 ms', 'ISSUER', 'AUTHORISED ✓')),
    (6.25, 'JNB', 'CPT', GREEN, ('+212 ms', 'MERCHANT · CAPE TOWN', 'APPROVED ✓')),
    (7.1, 'LON', 'CPT', AMBER, ('T+1', 'SETTLEMENT', 'R 1 249.00 → bank')),
]

def log_panel(img, t):
    a = eo(prog(t, 3.8, 4.2))
    if a <= 0: return
    x0, y0, x1, y1 = 60, 1330, 1020, 1690
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - 0.65 * a) + np.array([0.01, 0.03, 0.06], np.float32) * 0.65 * a
    E = u8(); cv2.rectangle(E, (x0, y0), (x1, y1), c8(CYAN, 0.5 * a), 1, cv2.LINE_AA); img += E.astype(np.float32) / 255
    over(img, text_rgba('TXN 7f3a·91c2 · R 1 249.00 · TRACE', MONOB(24), (0.55, 0.85, 1.0)), x0 + 24, y0 + 18, alpha=a)
    for i, (t0, a_, b_, col, (ms, who, what)) in enumerate(HOPS):
        p = prog(t, t0 + 0.45, t0 + 0.6)
        if p <= 0: continue
        y = y0 + 70 + i * 54
        over(img, rich_rgba([(f'{ms:>8}  ', (0.5, 0.6, 0.7)), (f'{who:<24}', (0.9, 0.95, 1.0)), (what, col)], LOGF), x0 + 20, y, alpha=p * a)

def frame(f):
    t = f / FPS
    img = BG.copy() * 0.55
    if t < 2.6:
        # card hero
        sp = prog(t, 2.0, 2.55)
        sc = 0.82 * (1 - eo(sp) * 0.97)
        ry = 0.35 * math.sin(t * 0.9 + 0.4) + sp * 3.0
        rx = 0.22 + 0.05 * math.sin(t * 1.3)
        cy = 880 - 260 * ei(sp)
        draw_card(img, 540, cy, max(sc, 0.02), ry, rx, -0.08, spec=-500 + t * 520)
        # contactless pulses
        P = u8()
        for k in range(3):
            ph = (t - 0.9 - k * 0.25)
            if 0 < ph < 1.0:
                cv2.circle(P, (540, int(cy)), int(120 + 520 * eo(ph)), c8(CYAN, (1 - ph) * 0.8), 3, cv2.LINE_AA)
        img += glow(P.astype(np.float32) / 255, 1.0)
        nn = int(clamp((t - 1.0) / 0.8) * 42)
        tx = 'TAP · R 1 249.00 · 09:41:07.218 · CPT'[:nn]
        if nn: over(img, text_rgba(tx, MONO(30), (0.6, 0.9, 1.0)), 540, 470, anchor='ct')
        if sp > 0:
            D = u8(); cv2.circle(D, (540, int(cy)), int(6 + 30 * sp), (255, 255, 255), -1, cv2.LINE_AA)
            img += glow(D.astype(np.float32) / 255, 2.0) * sp
            img += ei(prog(t, 2.35, 2.6)) * FL * 1.3
        caption(img, 'Where does a payment actually go?', t, 0.25, 2.3, y=1480, size=84)
    elif t < 9.4:
        u = prog(t, 2.6, 4.3)
        Rad = 2600 * (1 - eo(u)) + 520 * eo(u)
        dive = ei(prog(t, 8.8, 9.4))
        Rad = Rad * (1 + 6 * dive)
        lon0 = 18.4 + (16 - 18.4) * eo(u) + (18.4 - 16) * dive + 2.0 * math.sin(t * 0.2)
        lat0 = -33.9 + (6 + 33.9) * eo(u) - (6 + 33.9) * dive
        cy = 860
        arcs = []
        for (t0, a, b, col, _) in HOPS:
            p = prog(t, t0, t0 + 0.55)
            if p > 0: arcs.append((a, b, eo(p), col))
        R = draw_globe(img, lon0, lat0, 540, cy, Rad, t, arcs=arcs, ambient=t > 3.0, nodes=['CPT', 'JNB', 'LON'])
        # city callouts
        for c, lab in [('CPT', 'CAPE TOWN'), ('JNB', 'JOHANNESBURG'), ('LON', 'LONDON')]:
            x, y, z = city_screen(c, R, 540, cy, Rad)
            if z > 0.1 and 0 < x < W and 0 < y < H and t > 3.6:
                over(img, text_rgba(lab, MONOB(24), (0.85, 0.95, 1.0)), x + 22, y - 34, alpha=prog(t, 3.6, 4.0) * (1 - dive))
        if t < 2.95: img += (1 - eo(prog(t, 2.6, 2.95))) * FL * 1.3
        log_panel(img, t)
        caption(img, 'Authorised in 212 ms.', t, 6.5, 8.4, y=230, size=80)
        if dive > 0:
            img = zoom_blur(img, 1.0, 540, cy, 0.3 * dive, n=6)
            img += dive ** 2 * FL * 1.4
    elif t < 12.6:
        u = t - 9.4
        img = lifecycle(u)
        if u < 0.35: img += (1 - eo(prog(u, 0, 0.35))) * FL * 1.3
        caption(img, 'Every hop. Matched automatically.', t, 9.9, 12.3, y=1560, size=80)
        if t > 12.3: img += prog(t, 12.3, 12.6) ** 2 * FL * 1.3
    else:
        img = end_logo(t - 12.6)
        if t < 12.95: img += (1 - eo(prog(t, 12.6, 12.95))) * FL * 1.2
    img = img * VIGN
    img = chroma(img, 1)
    img = grain(img, f, 0.018)
    return np.clip(img, 0, 1)

SRC = [('CARD NETWORK', 'auth 7f3a·91c2', 'R 1 249.00'), ('PSP SETTLEMENT', 'batch 7781 · line 412', 'R 1 249.00'),
       ('BANK STATEMENT', 'Settlement · 07 OCT', 'R 1 211.53'), ('FEES', 'MDR 3.0%', 'R 37.47')]
def lifecycle(u):
    img = BG.copy()
    hp = eo(prog(u, 0, 0.4))
    over(img, text_rgba('TRANSACTION LIFECYCLE', F('Inter-SemiBold.otf', 32), (0.55, 0.85, 1.0), tracking=7), W / 2, 190, alpha=hp, anchor='ct')
    over(img, text_rgba('TXN 7f3a·91c2', F('InterDisplay-SemiBold.otf', 84), (1, 1, 1)), W / 2, 240, alpha=hp, anchor='ct')
    G = u8()
    cx, cy = 540, 1180
    for i, (nm, sub, amt) in enumerate(SRC):
        a = eo(prog(u, 0.2 + i * 0.12, 0.6 + i * 0.12))
        if a <= 0: continue
        x = 70 + (i % 2) * 490; y = 450 + (i // 2) * 230
        from_ = (x + 210, y + 180)
        ok = u > 1.0 + i * 0.28
        col = GREEN if ok else (0.5, 0.6, 0.7)
        box = np.zeros_like(img)
        cv2.rectangle(G, (x, y), (x + 450, y + 180), c8(col if ok else CYAN, 0.8 * a), 2, cv2.LINE_AA)
        img[y:y + 180, x:x + 450] = img[y:y + 180, x:x + 450] * (1 - 0.7 * a) + np.array([0.02, 0.05, 0.09], np.float32) * 0.7 * a
        over(img, text_rgba(nm, F('Inter-SemiBold.otf', 28), (0.6, 0.85, 1.0), tracking=4), x + 26, y + 20, alpha=a)
        over(img, text_rgba(sub, MONO(26), (0.6, 0.68, 0.78)), x + 26, y + 66, alpha=a)
        over(img, text_rgba(amt, F('InterDisplay-SemiBold.otf', 46), (1, 1, 1)), x + 26, y + 104, alpha=a)
        if ok:
            p = eo(prog(u, 1.0 + i * 0.28, 1.35 + i * 0.28))
            ex = int(from_[0] + (cx - from_[0]) * p); ey = int(y + 180 + (cy - 110 - y - 180) * p)
            cv2.line(G, (from_[0], y + 180), (ex, ey), c8(GREEN, 0.9), 3, cv2.LINE_AA)
            pts = np.array([[x + 400, y + 40], [x + 412, y + 54], [x + 432, y + 26]], np.int32)
            cv2.polylines(G, [pts], False, c8(GREEN), 4, cv2.LINE_AA)
    # central result
    rp = eo(prog(u, 2.2, 2.6))
    if rp > 0:
        cv2.circle(G, (cx, cy), int(110 * rp), c8(GREEN), 4, cv2.LINE_AA)
        pts = np.array([[cx - 45, cy], [cx - 12, cy + 34], [cx + 50, cy - 36]], np.int32)
        cv2.polylines(G, [pts], False, c8(GREEN, rp), 10, cv2.LINE_AA)
        over(img, text_rgba('R 1 249.00 = R 1 211.53 + R 37.47', MONOB(32), (0.9, 1, 0.95)), cx, cy + 140, alpha=rp, anchor='ct')
        over(img, text_rgba('MATCHED · JE-20431 POSTED', F('Inter-SemiBold.otf', 34), GREEN, tracking=4), cx, cy + 200, alpha=rp, anchor='ct')
    img += glow(G.astype(np.float32) / 255, 0.9)
    return img

if __name__ == '__main__':
    if sys.argv[1] == 'test':
        fs = [int(x) for x in sys.argv[2].split(',')]
        ims = [cv2.resize((frame(f) * 255).astype(np.uint8)[:, :, ::-1], (270, 480)) for f in fs]
        cv2.imwrite('t_v1.jpg', np.hstack(ims))
    else:
        a, b = int(sys.argv[2]), int(sys.argv[3]); w = Writer(f'v1_{a:04d}.mp4')
        for f in range(a, b): w.write(frame(f))
        w.close()
