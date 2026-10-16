import sys
from common import *
from logo import end_logo
from recon import tilt
FL = np.array([0.7, 0.9, 1.0], np.float32)
STAGES = [
    ('INGEST', 'Bank feeds · PSPs · card networks · ERP', 'events', 1248903, '{:,}'),
    ('NORMALISE', 'Currencies · formats · time zones', 'records', 1248903, '{:,}'),
    ('MATCH', 'Many-to-many · fuzzy · rules', 'auto-matched', 99.9, '{:.1f}%'),
    ('EXCEPTIONS', 'Flagged, explained, resolved', 'open', 0, '{:.0f}'),
    ('JOURNALS', 'Auto-generated · balanced · posted', 'posted', 2418, '{:,}'),
    ('CLOSE', 'Reconciled · locked · audit trail', '', 0, ''),
]
SRC = ['BANK FEEDS', 'PSPs', 'CARD NETWORKS', 'ERP']
Y0 = 560; GAP = 470
SY = [Y0 + 330 + k * GAP for k in range(len(STAGES))]
TA = [2.0 + k * 1.3 for k in range(len(STAGES))]
KEYS = [(0.0, 760), (1.6, 900)] + [(TA[k] + 0.35, SY[k] - 80) for k in range(len(STAGES))]
KEYS[-1] = (TA[-1] + 0.35, SY[-1] - 200)

def cam_y(t):
    for (t0, y0), (t1, y1) in zip(KEYS, KEYS[1:]):
        if t <= t1: return y0 + (y1 - y0) * eio(prog(t, t0, t1))
    return KEYS[-1][1]

def bez(p0, p3, s):
    p1 = np.array([p0[0], p0[1] + 140]); p2 = np.array([p3[0], p3[1] - 140])
    s = np.asarray(s)[..., None]
    return (1 - s) ** 3 * p0 + 3 * (1 - s) ** 2 * s * p1 + 3 * (1 - s) * s * s * p2 + s ** 3 * p3

def node_card(img, G, x0, y0, w, h, state, title, sub, val, unit, a):
    yy0, yy1 = max(0, y0), min(H, y0 + h)
    if yy1 <= yy0 or a <= 0: return
    img[yy0:yy1, x0:x0 + w] = img[yy0:yy1, x0:x0 + w] * (1 - 0.75 * a) + np.array([0.02, 0.05, 0.09], np.float32) * 0.75 * a
    col = {0: (0.3, 0.4, 0.5), 1: CYAN, 2: GREEN}[state]
    cv2.rectangle(G, (x0, y0), (x0 + w, y0 + h), c8(col, a * (0.6 if state == 0 else 1)), 2, cv2.LINE_AA)
    cv2.circle(G, (x0 + 70, y0 + h // 2), 34, c8(col, a), 2, cv2.LINE_AA)
    if state == 2:
        pts = np.array([[x0 + 54, y0 + h // 2], [x0 + 66, y0 + h // 2 + 13], [x0 + 88, y0 + h // 2 - 14]], np.int32)
        cv2.polylines(G, [pts], False, c8(GREEN, a), 4, cv2.LINE_AA)
    elif state == 1:
        ang = (time_now * 360) % 360
        cv2.ellipse(G, (x0 + 70, y0 + h // 2), (34, 34), ang, 0, 90, c8((1, 1, 1), a), 4, cv2.LINE_AA)
    over(img, text_rgba(title, F('Inter-Bold.otf', 40), (1, 1, 1), tracking=4), x0 + 130, y0 + 34, alpha=a)
    over(img, text_rgba(sub, F('Inter-Medium.otf', 25), (0.55, 0.65, 0.78)), x0 + 130, y0 + 92, alpha=a)
    if val:
        over(img, text_rgba(val, F('InterDisplay-SemiBold.otf', 52), GREEN if state == 2 else (1, 1, 1)), x0 + w - 30, y0 + 26, alpha=a, anchor='rt')
        over(img, text_rgba(unit.upper(), F('Inter-SemiBold.otf', 22), (0.55, 0.65, 0.78), tracking=4), x0 + w - 30, y0 + 98, alpha=a, anchor='rt')

time_now = 0.0
def scene(t):
    global time_now; time_now = t
    img = BG.copy() * 0.8
    cy = cam_y(t); off = cy - H / 2
    # background grid scrolling
    Gr = u8()
    for y in range(-int(off) % 90 - 90, H, 90): cv2.line(Gr, (0, y), (W, y), (14, 30, 44), 1)
    for x in range(0, W, 90): cv2.line(Gr, (x, 0), (x, H), (14, 30, 44), 1)
    img += Gr.astype(np.float32) / 255 * 0.6
    G = u8(); P = u8()
    # sources
    sa = eo(prog(t, 0.2, 0.8))
    sx = [150 + i * 260 for i in range(4)]
    for i, s in enumerate(SRC):
        y = int(Y0 - off)
        if -100 < y < H + 100:
            cv2.rectangle(G, (sx[i] - 110, y - 40), (sx[i] + 110, y + 40), c8(CYAN, 0.7 * sa), 2, cv2.LINE_AA)
            img[max(0, y - 40):max(0, y + 40), sx[i] - 110:sx[i] + 110] *= 0.5
            over(img, text_rgba(s, F('Inter-SemiBold.otf', 24), (0.8, 0.92, 1.0), tracking=2), sx[i], y - 14, alpha=sa, anchor='ct')
    # edges & packets
    edges = [(np.array([sx[i], Y0 + 40.0]), np.array([540.0, SY[0] - 120.0]), TA[0] - 0.8) for i in range(4)]
    edges += [(np.array([540.0, SY[k] + 120.0]), np.array([540.0, SY[k + 1] - 120.0]), TA[k + 1] - 0.6) for k in range(len(STAGES) - 1)]
    for ei_, (p0, p3, ton) in enumerate(edges):
        pp0 = p0 - [0, off]; pp3 = p3 - [0, off]
        if max(pp0[1], pp3[1]) < -50 or min(pp0[1], pp3[1]) > H + 50: continue
        ss = np.linspace(0, 1, 50); pts = bez(pp0, pp3, ss).astype(np.int32)
        cv2.polylines(G, [pts], False, c8(CYAN, 0.25), 2, cv2.LINE_AA)
        if t > ton:
            dens = 10 if ei_ < 4 else 16
            for q in range(dens):
                s0 = (t * 0.55 + q / dens + ei_ * 0.13) % 1
                if s0 > (t - ton) * 1.2: continue
                exc = (ei_ == 4 + 2) and q % 5 == 0  # MATCH->EXCEPTIONS edge
                c = AMBER if exc else (GREEN if ei_ >= 4 + 3 else CYAN)
                for tr in range(6):
                    st = s0 - tr * 0.012
                    if st < 0: break
                    p = bez(pp0, pp3, st)
                    cv2.circle(P, (int(p[0]), int(p[1])), max(1, 6 - tr), c8(c, 1 - tr / 6), -1, cv2.LINE_AA)
    # nodes
    for k, (title, sub, unit, val, fmt) in enumerate(STAGES):
        y = int(SY[k] - off - 120)
        if y > H + 50 or y + 240 < -50: continue
        a = eo(prog(t, TA[k] - 0.9, TA[k] - 0.5))
        state = 0 if t < TA[k] else (1 if t < TA[k] + 1.0 else 2)
        if k == len(STAGES) - 1:
            state = 0 if t < TA[k] else (1 if t < TA[k] + 0.8 else 2)
        pv = eio(prog(t, TA[k], TA[k] + 1.0))
        if title == 'EXCEPTIONS':
            v = fmt.format(round(12 * (1 - pv))) if t >= TA[k] else '12'
        elif fmt:
            v = fmt.format(int(val * pv)) if ',' in fmt else fmt.format(val * pv)
        else:
            v = ''
        node_card(img, G, 160, y, 760, 240, state, title, sub, v, unit if title != 'EXCEPTIONS' else 'open', a)
        if title == 'CLOSE' and state == 2:
            p = eo(prog(t, TA[k] + 0.8, TA[k] + 1.3))
            over(img, text_rgba('CLOSED', F('InterDisplay-Bold.otf', 150), GREEN), 540, y + 300 + (1 - p) * 30, alpha=p, anchor='ct')
            over(img, text_rgba('AUDIT-READY  ·  OCTOBER 2026', F('Inter-SemiBold.otf', 30), (0.6, 0.75, 0.85), tracking=6), 540, y + 480, alpha=p, anchor='ct')
            cv2.circle(G, (540, y + 120), int(380 * p + 200), c8(GREEN, (1 - p) * 0.8), 3, cv2.LINE_AA)
    img += glow(G.astype(np.float32) / 255, 0.7)
    img += glow(P.astype(np.float32) / 255, 1.3)
    return img

def frame(f):
    t = f / FPS
    if t >= 12.2:
        img = end_logo(t - 12.2)
        if t < 12.55: img += (1 - eo(prog(t, 12.2, 12.55))) * FL * 1.2
        return np.clip(grain(img * VIGN, f, 0.018), 0, 1)
    img = scene(t)
    img = tilt(img, 0.06 * math.sin(t * 0.4), 0.10 + 0.03 * math.sin(t * 0.3), zoom=1.0)
    caption(img, 'What if month-end ran itself?', t, 0.3, 2.4, y=1620, size=84)
    if t > 11.9: img += prog(t, 11.9, 12.2) ** 2 * FL * 1.3
    img = img * VIGN
    img = chroma(img, 1)
    img = grain(img, f, 0.018)
    return np.clip(img, 0, 1)

if __name__ == '__main__':
    if sys.argv[1] == 'test':
        fs = [int(x) for x in sys.argv[2].split(',')]
        ims = [cv2.resize((frame(f) * 255).astype(np.uint8)[:, :, ::-1], (270, 480)) for f in fs]
        cv2.imwrite('t_v3.jpg', np.hstack(ims))
    else:
        a, b = int(sys.argv[2]), int(sys.argv[3]); w = Writer(f'v3_{a:04d}.mp4')
        for f in range(a, b): w.write(frame(f))
        w.close()
