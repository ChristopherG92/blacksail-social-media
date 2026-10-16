"""Exception of the Week #01 (v2, USD) - "4 payments didn't reconcile. $186,420 unexplained."
v2 2026-10-12: USD amounts, larger values, four breaks in one payout, exception-queue scene with live unresolved counter.
Scenes: ledger hook (tilted, DOF) -> payment trail with camera follow -> exception workflow -> payoff -> logo.
All figures are illustrative. Text kept inside the 10% top / 20% bottom safe zones (y 192..1536).
Usage: PYTHONPATH=.. python3 exception_journey.py test 10,60,150,200,280,330,400,460
       PYTHONPATH=.. python3 exception_journey.py render 0 528
"""
import sys
from common import *
from logo import end_logo
from recon import tilt, rounded, check
from PIL import ImageFilter

FL = np.array([0.7, 0.9, 1.0], np.float32)
T_B, T_C, T_D, T_E, TOT = 3.0, 7.4, 12.6, 14.6, 17.6
SAFE_TOP, SAFE_BOT = 192, 1536

# ---------------------------------------------------------------- scene A: ledger hook
def tilt_clean(img, ang_y, ang_x=0.0, zoom=1.0):
    """Like recon.tilt but fills outside the panel with the brand background (no edge smearing)."""
    f = 1800
    pts = np.float32([[0, 0], [W, 0], [W, H], [0, H]]); out = []
    for (x, y) in pts:
        X, Y = (x - W / 2) * zoom, (y - H / 2) * zoom
        Zc = f + X * math.sin(ang_y) + Y * math.sin(ang_x)
        out.append([W / 2 + f * X * math.cos(ang_y) / Zc, H / 2 + f * Y * math.cos(ang_x) / Zc])
    Hm = cv2.getPerspectiveTransform(pts, np.float32(out))
    warped = cv2.warpPerspective(img, Hm, (W, H), borderMode=cv2.BORDER_CONSTANT)
    mask = cv2.warpPerspective(np.ones((H, W), np.float32), Hm, (W, H), borderMode=cv2.BORDER_CONSTANT)[..., None]
    return warped + BG * 0.9 * (1 - mask)

r_ = np.random.default_rng(7)
ROWS = []
for i in range(34):
    ROWS.append((f'TX-{r_.integers(10000, 99999)}', f'{6 + i // 6:02d} OCT', float(r_.integers(4000, 250000)) + r_.integers(0, 100) / 100))
BREAKS = [('TX-48213', 72500.00, 'Refund netted from payout', 'Matched 1 : 2 · JE posted'),
          ('TX-50917', 48120.00, 'Chargeback CB-3381 deducted', 'Journal posted'),
          ('TX-61044', 39800.00, 'Missed cut-off · settled T+1', 'Matched next day'),
          ('TX-73302', 26000.00, 'Rolling reserve withheld', 'Reserve booked')]
GAP_TOTAL = sum(b[1] for b in BREAKS)   # 186,420.00
FOCUS_SET = [9, 11, 12, 14]
for k, i in enumerate(FOCUS_SET): ROWS[i] = (BREAKS[k][0], '07 OCT', BREAKS[k][1])
FOCUS = 11.5
RH_A = 112

def fmt_r(v): return f'${v:,.2f}'

def _ledger_sheet():
    w, h = 980, RH_A * len(ROWS) + 40
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w - 1, h - 1), 28, fill=(7, 16, 30, 235), outline=(40, 140, 200, 200), width=2)
    for i, (ref, dt, amt) in enumerate(ROWS):
        y = 20 + i * RH_A
        if i in FOCUS_SET:
            d.rounded_rectangle((14, y + 6, w - 14, y + RH_A - 6), 16, fill=(60, 40, 8, 255), outline=(255, 184, 51, 255), width=3)
        d.text((40, y + 22), ref, font=MONOB(32), fill=(235, 245, 255, 255))
        d.text((40, y + 66), dt, font=MONO(22), fill=(120, 150, 180, 255))
        s = fmt_r(amt); tw = d.textlength(s, font=MONO(32))
        d.text((w - 300 - tw, y + 22), s, font=MONO(32), fill=(225, 235, 245, 255))
        lab, col = ('UNMATCHED', (255, 184, 51, 255)) if i in FOCUS_SET else ('MATCHED', (64, 255, 140, 255))
        tw2 = d.textlength(lab, font=MONOB(24))
        d.text((w - 40 - tw2, y + 28), lab, font=MONOB(24), fill=col)
        if i not in FOCUS_SET:
            d.line((40, y + RH_A - 1, w - 40, y + RH_A - 1), fill=(30, 55, 80, 255), width=1)
    return np.asarray(im, np.float32) / 255
SHEET = _ledger_sheet()

def scene_a(t, f):
    img = BG.copy() * 0.9
    # focus row centred around y=1000 at t ~ 1.0, slow drift upward afterwards
    sc = 18 * eo(prog(t, 0, 1.2)) + 30 * t
    fy = 20 + FOCUS * RH_A + RH_A / 2
    y0 = 1000 - fy + 260 * (1 - eo(prog(t, 0, 1.2))) - sc * 0.4
    lay = np.zeros((H, W, 3), np.float32); a = np.zeros((H, W), np.float32)
    canvas = np.zeros((H, W, 4), np.float32)
    over_rgba = SHEET
    x0 = (W - SHEET.shape[1]) // 2
    ys, ye = int(max(0, y0)), int(min(H, y0 + SHEET.shape[0]))
    if ye > ys:
        canvas[ys:ye, x0:x0 + SHEET.shape[1]] = SHEET[ys - int(y0):ye - int(y0)]
    rgb = canvas[..., :3] * canvas[..., 3:4]
    base = img * (1 - canvas[..., 3:4]) + rgb
    # 3D tilt + slow push in
    zoom = 1.0 + 0.10 * eio(prog(t, 0, T_B))
    base = tilt_clean(base, -0.18 + 0.06 * t / T_B, 0.32, zoom)
    # depth of field: sharp band around focus row, blur elsewhere
    blur = cv2.GaussianBlur(base, (0, 0), 7)
    m = smoothstep(330, 560, np.abs(YY - 1000))[..., None]
    img = base * (1 - m) + blur * m
    # focus row pulse + glitch
    pul = 0.5 + 0.5 * math.sin(t * 9)
    img += (np.exp(-((YY - 1000) / 260) ** 2) * (np.abs(XX - W / 2) < 500))[..., None] * AMBER * 0.05 * pul
    # legibility bands behind hook + question
    img *= 1 - 0.8 * np.exp(-((YY[:, :1] - 390) / 190) ** 2)[..., None][:, :, :1]
    img *= 1 - 0.88 * np.exp(-((YY[:, :1] - 1380) / 150) ** 2)[..., None][:, :, :1] * eo(prog(t, 1.0, 1.4))
    # hook text (visible from frame 3)
    hp = eo(prog(t, 0.03, 0.35))
    over(img, text_rgba('EXCEPTION OF THE WEEK · 01', F('Inter-SemiBold.otf', 30), (0.55, 0.85, 1.0), tracking=8), W / 2, 250 - (1 - hp) * 20, alpha=hp, anchor='ct')
    font = F('InterDisplay-Bold.otf', 92)
    for k, ln in enumerate(['4 payments', "didn't reconcile."]):
        p = eo(prog(t, 0.05 + k * 0.12, 0.45 + k * 0.12))
        col = (1, 1, 1) if k == 0 else (1.0, 0.75, 0.25)
        over(img, text_rgba(ln, font, col), W / 2, 320 + k * 110 + (1 - p) * 30, alpha=p, anchor='ct')
    sp = eo(prog(t, 1.1, 1.5))
    over(img, text_rgba(f'${GAP_TOTAL:,.0f} unexplained.', F('InterDisplay-Bold.otf', 70), (1.0, 0.75, 0.25)), W / 2, 1300 + (1 - sp) * 20, alpha=sp, anchor='ct')
    sp2 = eo(prog(t, 1.4, 1.8))
    over(img, text_rgba('What happens next?', F('InterDisplay-SemiBold.otf', 52), (0.8, 0.9, 1.0)), W / 2, 1392 + (1 - sp2) * 20, alpha=sp2, anchor='ct')
    over(img, text_rgba('ILLUSTRATIVE DATA', F('Inter-Medium.otf', 22), (0.45, 0.55, 0.65), tracking=5), W / 2, 1480, alpha=sp * 0.9, anchor='ct')
    g = bell(t, 0.25, 0.32, 0.42, 0.55) + bell(t, 1.0, 1.05, 1.1, 1.2) * 0.6
    if g > 0: img = glitch(img, f, g * 0.8)
    if t > T_B - 0.3:
        u = prog(t, T_B - 0.3, T_B)
        img = zoom_blur(img, 1 + 0.25 * ei(u), W / 2, 1000, 0.25 * u)
        img += ei(u) * FL * 1.2
    return img

# ---------------------------------------------------------------- scene B: payment trail
NODES = [
    ('CHECKOUT', '2,418 card payments · 06 OCT', '$1,248,900.00', CYAN),
    ('PSP SETTLEMENT', 'Batch 7781 · gross settlement', '$1,248,900.00', CYAN),
    ('BANK STATEMENT', 'Payout received · 07 OCT', '$1,062,480.00', CYAN),
    ('LEDGER MATCH', 'Expected $1,248,900.00 · received $1,062,480.00', 'Δ $186,420.00', AMBER),
]
NY = [520, 900, 1280, 1660]          # world y of node tops
NH = 230
NT = [0.25, 1.05, 1.85, 2.65]         # local arrival times
def camB(u):
    return 640 * eio(prog(u, 0.9, 3.0))

def scene_b(t, f):
    u = t - T_B
    img = BG.copy() * 0.85
    off = camB(u)
    # parallax dot grid (depth)
    Gr = u8()
    for y in range(int(-off * 0.5) % 80 - 80, H, 80):
        for x in range(40, W, 80): cv2.circle(Gr, (x, y), 2, (25, 60, 90), -1)
    img += cv2.GaussianBlur(Gr.astype(np.float32) / 255, (0, 0), 1.5) * 0.8
    hp = eo(prog(u, 0, 0.3))
    over(img, text_rgba('FOLLOW THE MONEY', F('Inter-SemiBold.otf', 30), (0.55, 0.85, 1.0), tracking=8), W / 2, 230 - min(off, 60) * 0, alpha=hp * (1 - prog(off, 0, 120)), anchor='ct')
    G = u8()
    for k, (title, sub, amt, col) in enumerate(NODES):
        a = eo(prog(u, NT[k] - 0.25, NT[k] + 0.15))
        if a <= 0: continue
        y0 = int(NY[k] - off + (1 - a) * 50)
        done = u > NT[k] + 0.2
        is_exc = k == 3 and u > NT[3] + 0.25
        bcol = AMBER if is_exc else (GREEN if done and k < 3 else CYAN)
        rounded(img, 90, y0, 990, y0 + NH, (0.025, 0.06, 0.11), 0.9 * a, border=tuple(bcol), r=26)
        over(img, text_rgba(f'0{k + 1}', MONOB(26), (0.45, 0.6, 0.75)), 130, y0 + 34, alpha=a)
        over(img, text_rgba(title, F('Inter-Bold.otf', 40), (1, 1, 1), tracking=3), 200, y0 + 28, alpha=a)
        over(img, text_rgba(sub, F('Inter-Medium.otf', 27), (0.6, 0.7, 0.82)), 130, y0 + 96, alpha=a)
        acol = (1.0, 0.75, 0.25) if k == 3 else ((1.0, 0.75, 0.25) if (k == 2 and u > NT[3] + 0.25) else (1, 1, 1))
        over(img, text_rgba(amt, F('InterDisplay-SemiBold.otf', 60), acol), 130, y0 + 142, alpha=a)
        if is_exc:
            p = prog(u, NT[3] + 0.25, NT[3] + 0.5)
            pill = text_rgba('EXCEPTION', MONOB(30), (1.0, 0.75, 0.25))
            rounded(img, 700, y0 + 150, 950, y0 + 205, (0.25, 0.15, 0.02), 0.9 * p, border=(1.0, 0.72, 0.2), r=14)
            over(img, pill, 825, y0 + 158, alpha=p, anchor='ct')
            pul = 0.5 + 0.5 * math.sin(u * 10)
            cv2.rectangle(G, (90, y0), (990, y0 + NH), c8(AMBER, 0.5 * pul), 6, cv2.LINE_AA)
        elif done and k < 3:
            check(G, 940, y0 + 52, 13, c8(GREEN))
        # connector + travelling packet
        if k < 3:
            ya, yb = int(NY[k] + NH - off), int(NY[k + 1] - off)
            pp = prog(u, NT[k] + 0.2, NT[k + 1])
            if pp > 0:
                cv2.line(G, (540, ya), (540, int(ya + (yb - ya) * pp)), c8(CYAN, 0.7), 3, cv2.LINE_AA)
                if pp < 1:
                    py = int(ya + (yb - ya) * eio(pp))
                    for j in range(8):  # motion-blurred packet trail
                        cv2.circle(G, (540, py - j * 9), max(2, 13 - j), c8(WHITE if j == 0 else CYAN, 1 - j / 8), -1, cv2.LINE_AA)
    img += glow(G.astype(np.float32) / 255, 0.9)
    # caption above bottom safe line
    if u > NT[3] + 0.5:
        p = eo(prog(u, NT[3] + 0.5, NT[3] + 0.9))
        # this caption sits on a dark band so it reads over the node
        over(img, text_rgba('$186,420 is missing.', F('InterDisplay-Bold.otf', 72), (1, 1, 1)), W / 2, 1330 + (1 - p) * 20, alpha=p, anchor='ct')
        p2 = eo(prog(u, NT[3] + 0.8, NT[3] + 1.2))
        over(img, text_rgba('Four breaks. Nobody knows why. Yet.', F('InterDisplay-SemiBold.otf', 50), (0.75, 0.85, 0.95)), W / 2, 1425 + (1 - p2) * 20, alpha=p2, anchor='ct')
    g = bell(u, NT[3] + 0.22, NT[3] + 0.27, NT[3] + 0.35, NT[3] + 0.48)
    if g > 0: img = glitch(img, f, g * 0.7)
    if u < 0.3: img += (1 - eo(prog(u, 0, 0.3))) * FL * 1.2
    if t > T_C - 0.3:
        v = prog(t, T_C - 0.3, T_C)
        img = zoom_blur(img, 1 + 0.2 * ei(v), W / 2, 900, 0.22 * v)
        img += ei(v) * FL * 1.2
    return img

# ---------------------------------------------------------------- scene C: exception queue
SY_C = [470, 705, 940, 1175]; SH = 205
ST = [0.9, 1.7, 2.5, 3.3]          # resolve times (local)

def scene_c(t, f):
    u = t - T_C
    img = BG.copy() * 0.9
    zoom = 1.04 - 0.04 * eo(prog(u, 0, T_D - T_C))
    hp = eo(prog(u, 0, 0.4))
    over(img, text_rgba('EXCEPTION QUEUE · PAYOUT 7781', F('Inter-SemiBold.otf', 30), (0.55, 0.85, 1.0), tracking=6), 90, 250, alpha=hp)
    # live unresolved total
    rem = GAP_TOTAL
    for k, b in enumerate(BREAKS):
        rem -= b[1] * eio(prog(u, ST[k], ST[k] + 0.35))
    n_open = sum(1 for k in range(4) if u <= ST[k] + 0.1)
    over(img, text_rgba(f'{n_open} OPEN', MONOB(34), GREEN if n_open == 0 else AMBER), 990, 248, alpha=hp, anchor='rt')
    col = (0.25, 1.0, 0.55) if rem < 1 else (1.0, 0.75, 0.25)
    over(img, text_rgba(f'Δ ${rem:,.2f}', F('InterDisplay-Bold.otf', 104), col), 82, 300, alpha=hp)
    G = u8()
    sp = eio(prog(u, 0.3, ST[-1] + 0.3))
    ytop, ybot = SY_C[0] + SH // 2, SY_C[-1] + SH // 2
    cv2.line(G, (140, ytop), (140, ybot), (30, 60, 85), 3, cv2.LINE_AA)
    cv2.line(G, (140, ytop), (140, int(ytop + (ybot - ytop) * sp)), c8(GREEN, 0.9), 3, cv2.LINE_AA)
    for k, (ref, amt, cause, fix) in enumerate(BREAKS):
        a = eo(prog(u, 0.1 + k * 0.1, 0.5 + k * 0.1))
        if a <= 0: continue
        y0 = SY_C[k] + int((1 - a) * 40)
        done = u > ST[k]
        active = (not done) and (k == 0 or u > ST[k - 1])
        bcol = GREEN if done else (AMBER if active else (0.45, 0.35, 0.15))
        rounded(img, 210, y0, 990, y0 + SH, (0.025, 0.06, 0.11), 0.88 * a, border=tuple(bcol), r=24)
        over(img, text_rgba(ref, MONOB(34), (1, 1, 1)), 250, y0 + 28, alpha=a)
        over(img, text_rgba(f'${amt:,.2f}', F('InterDisplay-SemiBold.otf', 44), (1, 1, 1) if done else (1.0, 0.75, 0.25)), 950, y0 + 22, alpha=a, anchor='rt')
        over(img, text_rgba(cause, F('Inter-Medium.otf', 29), (0.75, 0.82, 0.92)), 250, y0 + 92, alpha=a)
        st = ('RESOLVED · ' + fix) if done else ('INVESTIGATING…' if active else 'OPEN')
        over(img, text_rgba(st, F('Inter-SemiBold.otf', 25), GREEN if done else (AMBER if active else (0.6, 0.55, 0.45)), tracking=2), 250, y0 + 146, alpha=a * (prog(u, ST[k], ST[k] + 0.2) if done else 1))
        cy = SY_C[k] + SH // 2
        cv2.circle(G, (140, cy), 26, c8(bcol, a), 3, cv2.LINE_AA)
        if done:
            q = eo(prog(u, ST[k], ST[k] + 0.25))
            cv2.circle(G, (140, cy), int(26 + 40 * q), c8(GREEN, (1 - q) * 0.8), 2, cv2.LINE_AA)
            check(G, 140, cy + 2, 11, c8(GREEN))
        elif active:
            ang = (u * 400) % 360
            cv2.ellipse(G, (140, cy), (26, 26), ang, 0, 100, c8(WHITE, a), 4, cv2.LINE_AA)
    img += glow(G.astype(np.float32) / 255, 0.8)
    cp = eo(prog(u, ST[-1] + 0.4, ST[-1] + 0.8))
    if cp > 0:
        over(img, text_rgba('Detected. Explained. Resolved. Logged.', F('InterDisplay-Bold.otf', 54), (1, 1, 1)), W / 2, 1430 + (1 - cp) * 20, alpha=cp, anchor='ct')
    img = scale_about(img, zoom, W / 2, H / 2)
    if u < 0.3: img += (1 - eo(prog(u, 0, 0.3))) * FL * 0.85
    if t > T_D - 0.25:
        v = prog(t, T_D - 0.25, T_D)
        img = img * (1 - v) + BG * v
    return img

# ---------------------------------------------------------------- scene D: payoff
def scene_d(t, f):
    u = t - T_D
    img = BG.copy()
    G = u8()
    rp = eio(prog(u, 0.0, 0.6))
    cx, cy = W // 2, 640
    cv2.circle(G, (cx, cy), 150, (25, 50, 70), 10, cv2.LINE_AA)
    cv2.ellipse(G, (cx, cy), (150, 150), -90, 0, 360 * rp, c8(GREEN), 10, cv2.LINE_AA)
    if rp >= 1:
        pts = np.array([[cx - 60, cy], [cx - 16, cy + 46], [cx + 66, cy - 48]], np.int32)
        cv2.polylines(G, [pts], False, c8(GREEN), 14, cv2.LINE_AA)
    img += glow(G.astype(np.float32) / 255, 1.0)
    for k, (ln, col) in enumerate([('Caught on day one.', (1, 1, 1)), ('Not at month-end.', (0.25, 1.0, 0.55))]):
        p = eo(prog(u, 0.25 + k * 0.2, 0.7 + k * 0.2))
        over(img, text_rgba(ln, F('InterDisplay-Bold.otf', 96), col), W / 2, 900 + k * 120 + (1 - p) * 30, alpha=p, anchor='ct')
    p = eo(prog(u, 0.9, 1.3))
    over(img, text_rgba('AUTOMATED MATCHING · EXCEPTION WORKFLOWS · AUDIT TRAIL', F('Inter-SemiBold.otf', 24), (0.55, 0.7, 0.85), tracking=4), W / 2, 1200, alpha=p, anchor='ct')
    if u < 0.25: img = img * eo(prog(u, 0, 0.25)) + BG * (1 - eo(prog(u, 0, 0.25)))
    if t > T_E - 0.3:
        v = prog(t, T_E - 0.3, T_E)
        img = zoom_blur(img, 1 + 0.15 * ei(v), W / 2, 900, 0.2 * v)
        img += ei(v) * FL * 1.1
    return img

def frame(f):
    t = f / FPS
    if t < T_B: img = scene_a(t, f)
    elif t < T_C: img = scene_b(t, f)
    elif t < T_D: img = scene_c(t, f)
    elif t < T_E: img = scene_d(t, f)
    else:
        img = end_logo(t - T_E)
        if t < T_E + 0.35: img += (1 - eo(prog(t, T_E, T_E + 0.35))) * FL * 1.2
    img = img * VIGN
    img = chroma(img, 1)
    img = grain(img, f, 0.018)
    return np.clip(img, 0, 1)

def audio(path):
    from sfx import Mix, drone, impact, whoosh, ping, blip, click, riser, chime, beat, beep, finish, hp, tt
    m = Mix(TOT, seed=11)
    drone(m, 0, TOT, 0.5)
    impact(m, 0.05, 0.9)
    for k in range(10): click(m, 0.4 + k * 0.22, 0.08, pan=(-1) ** k * 0.3)
    m.place(hp(m.noise(0.25), 3000) * 0.3, 0.27, 0.6)                    # glitch hiss
    beat(m, T_B, T_D, bpm=120, gain=0.32)
    whoosh(m, T_B - 0.35, 0.5, 0.45)
    for k in range(4): blip(m, T_B + NT[k], 880 * (1.12 ** k), 0.07)
    impact(m, T_B + NT[3] + 0.25, 0.55)
    m.place(hp(m.noise(0.25), 2500) * 0.35, T_B + NT[3] + 0.22, 0.6)
    beep(m, T_B + NT[3] + 0.3, 700, 0.12, 0.08); beep(m, T_B + NT[3] + 0.48, 700, 0.12, 0.08)
    whoosh(m, T_C - 0.35, 0.5, 0.45)
    for k in range(4): ping(m, T_C + ST[k], 990 + 120 * k, 0.1)
    riser(m, T_D - 1.2, 1.2, 0.25)
    chime(m, T_D + 0.55, 0.8)
    riser(m, T_E - 1.0, 1.0, 0.3)
    impact(m, T_E + 0.05, 0.8)
    chime(m, T_E + 0.9, 0.6)
    finish(m, path, fade=0.8)

if __name__ == '__main__':
    if sys.argv[1] == 'test':
        fs = [int(x) for x in sys.argv[2].split(',')]
        ims = [cv2.resize((frame(f) * 255).astype(np.uint8)[:, :, ::-1], (360, 640)) for f in fs]
        rows = [np.hstack(ims[i:i + 4] + [np.zeros_like(ims[0])] * (4 - len(ims[i:i + 4]))) for i in range(0, len(ims), 4)]
        cv2.imwrite('t_exc2.jpg', np.vstack(rows))
    elif sys.argv[1] == 'audio':
        audio(sys.argv[2])
    else:
        a, b = int(sys.argv[2]), int(sys.argv[3]); w = Writer(f'exc2_{a:04d}.mp4')
        for f in range(a, b): w.write(frame(f))
        w.close()
