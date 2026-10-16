from common import *

RF = MONO(29); RFB = MONOB(29); RFS = MONO(22)
LABF = F('Inter-SemiBold.otf', 28)
r_ = np.random.default_rng(42)
NR = 9
BANK = []
for i in range(NR):
    amt = float(r_.integers(1200, 98000)) + r_.integers(0, 100) / 100
    BANK.append((f'{6 + i // 3:02d} OCT', f'TX-{r_.integers(10000, 99999)}', amt))
PERM = [2, 0, 4, 1, 3, 6, 5, 8, 7]
LEDG = [None] * NR
for i, j in enumerate(PERM):
    acct = ['Revenue', 'Settlements', 'Fees', 'Payouts', 'Refunds', 'Chargebacks', 'FX', 'Revenue', 'Payouts'][j]
    LEDG[j] = (f'JE-{r_.integers(1000, 9999)}', acct, BANK[i][2])
EXC = {3: 'FX VARIANCE', 6: 'TIMING DIFF'}

LX0, LX1, RX0, RX1 = 40, 470, 610, 1040
ROW0, RH = 600, 96

def fmt_r(v): return 'R ' + f'{v:,.2f}'.replace(',', ' ')

def panel(L, x0, y0, x1, y1, a=1.0):
    cv2.rectangle(L, (x0, y0), (x1, y1), (0, 0, 0), -1)

def rounded(img, x0, y0, x1, y1, fill, alpha, border=None, r=22):
    m = np.zeros((H, W), np.uint8)
    cv2.rectangle(m, (x0 + r, y0), (x1 - r, y1), 255, -1); cv2.rectangle(m, (x0, y0 + r), (x1, y1 - r), 255, -1)
    for (cx, cy) in [(x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)]:
        cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
    a = m.astype(np.float32)[..., None] / 255 * alpha
    img[:] = img * (1 - a) + np.array(fill, np.float32) * a
    if border is not None:
        E = u8()
        cv2.rectangle(E, (x0 + r, y0), (x1 - r, y0), c8(border), 2, cv2.LINE_AA)
        cv2.rectangle(E, (x0 + r, y1), (x1 - r, y1), c8(border), 2, cv2.LINE_AA)
        cv2.line(E, (x0, y0 + r), (x0, y1 - r), c8(border), 2, cv2.LINE_AA); cv2.line(E, (x1, y0 + r), (x1, y1 - r), c8(border), 2, cv2.LINE_AA)
        for (cx, cy, a0) in [(x0 + r, y0 + r, 180), (x1 - r, y0 + r, 270), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90)]:
            cv2.ellipse(E, (cx, cy), (r, r), a0, 0, 90, c8(border), 2, cv2.LINE_AA)
        img += E.astype(np.float32) / 255 * alpha

def check(L, cx, cy, s, col):
    pts = np.array([[cx - s, cy], [cx - s * 0.3, cy + s * 0.7], [cx + s, cy - s * 0.8]], np.int32)
    cv2.polylines(L, [pts], False, col, 4, cv2.LINE_AA)

def recon_ui(u):
    """u: local time (s). Returns float RGB UI over dark bg (flat)."""
    img = BG.copy()
    # header
    hp = eo(prog(u, 0.0, 0.5))
    over(img, text_rgba('LIVE RECONCILIATION', F('Inter-SemiBold.otf', 32), (0.55, 0.85, 1.0), tracking=8), 60, 190 - (1 - hp) * 30, alpha=hp)
    nmatch = int(1_248_903 * eio(prog(u, 0.8, 3.6)))
    over(img, text_rgba(f'{nmatch:,}', F('InterDisplay-SemiBold.otf', 124), (1, 1, 1)), 52, 240, alpha=hp)
    over(img, text_rgba('TRANSACTIONS MATCHED', F('Inter-Medium.otf', 28), (0.55, 0.65, 0.78), tracking=5), 62, 400, alpha=hp)
    acc = 99.9 * eio(prog(u, 0.8, 3.4))
    over(img, text_rgba(f'{acc:.1f}%', F('InterDisplay-SemiBold.otf', 64), GREEN), W - 60, 270, alpha=hp, anchor='rt')
    over(img, text_rgba('ACCURACY', F('Inter-Medium.otf', 26), (0.55, 0.65, 0.78), tracking=5), W - 62, 350, alpha=hp, anchor='rt')
    # column panels
    for (x0, x1, title, delay) in [(LX0, LX1, 'BANK FEED', 0.1), (RX0, RX1, 'GENERAL LEDGER', 0.2)]:
        p = eo(prog(u, delay, delay + 0.5))
        if p <= 0: continue
        dy = int((1 - p) * 80)
        rounded(img, x0, 490 + dy, x1, ROW0 + NR * RH + 30 + dy, (0.03, 0.07, 0.12), 0.85 * p, border=(0.15, 0.55, 0.8))
        over(img, text_rgba(title, LABF, (0.6, 0.85, 1.0), tracking=4), x0 + 26, 512 + dy, alpha=p)
    scan_y = ROW0 + (u - 0.9) / 2.4 * NR * RH
    L = u8()
    for i in range(NR):
        ra = eo(prog(u, 0.35 + i * 0.05, 0.75 + i * 0.05))
        if ra <= 0: continue
        y = ROW0 + i * RH
        j = PERM[i]; yr = ROW0 + j * RH
        ti = 0.9 + (i + 0.5) / NR * 2.4
        mp = prog(u, ti, ti + 0.3)
        state = 'pend'
        if mp > 0: state = 'match'
        if i in EXC and mp > 0:
            state = 'exc' if u < ti + 0.75 else 'resolved'
        colr = {'pend': (0.45, 0.5, 0.58), 'match': GREEN, 'exc': AMBER, 'resolved': GREEN}[state]
        tint = {'pend': 0.0, 'match': 0.12, 'exc': 0.16, 'resolved': 0.12}[state] * prog(u, ti, ti + 0.2)
        for (x0, x1, yy) in [(LX0, LX1, y), (RX0, RX1, yr)]:
            if tint > 0:
                img[yy + 6:yy + RH - 6, x0 + 12:x1 - 12] += np.array(colr, np.float32) * tint
            cv2.line(L, (x0 + 20, yy + RH), (x1 - 20, yy + RH), (30, 50, 70), 1)
        d, ref, amt = BANK[i]
        over(img, text_rgba(ref, RFB, (0.92, 0.96, 1.0)), LX0 + 26, y + 14, alpha=ra)
        over(img, text_rgba(d, RFS, (0.5, 0.6, 0.72)), LX0 + 26, y + 54, alpha=ra)
        over(img, text_rgba(fmt_r(amt), RF, (0.92, 0.96, 1.0)), LX1 - 26, y + 14, alpha=ra, anchor='rt')
        je, acct, amt2 = LEDG[j]
        over(img, text_rgba(je, RFB, (0.92, 0.96, 1.0)), RX0 + 26, yr + 14, alpha=ra)
        over(img, text_rgba(acct.upper(), RFS, (0.5, 0.6, 0.72)), RX0 + 26, yr + 54, alpha=ra)
        over(img, text_rgba(fmt_r(amt2), RF, (0.92, 0.96, 1.0)), RX1 - 26, yr + 14, alpha=ra, anchor='rt')
        if state != 'pend':
            lab = {'match': 'MATCHED', 'exc': EXC.get(i, ''), 'resolved': 'AUTO-RESOLVED'}[state]
            over(img, text_rgba(lab, RFS, colr), LX1 - 26, y + 56, alpha=prog(u, ti, ti + 0.2), anchor='rt')
            # connector
            n = max(2, int(40 * eo(mp)))
            ss = np.linspace(0, eo(mp), n)
            p0 = np.array([LX1, y + RH / 2]); p3 = np.array([RX0, yr + RH / 2])
            p1 = p0 + [60, 0]; p2 = p3 - [60, 0]
            pts = ((1 - ss) ** 3)[:, None] * p0 + (3 * (1 - ss) ** 2 * ss)[:, None] * p1 + (3 * (1 - ss) * ss ** 2)[:, None] * p2 + (ss ** 3)[:, None] * p3
            cv2.polylines(L, [pts.astype(np.int32)], False, c8(colr), 3, cv2.LINE_AA)
            cv2.circle(L, tuple(pts[-1].astype(int)), 6, c8(colr), -1, cv2.LINE_AA)
            if state in ('match', 'resolved') and mp >= 1:
                check(L, RX1 - 40, yr + 70, 11, c8(GREEN))
    # scanner beam
    if 0.9 <= u <= 3.4:
        img += (np.exp(-((YY[:, :1] - scan_y) / 10) ** 2) * ((XX[:1] > LX0) & (XX[:1] < RX1)))[..., None] * CYAN * 0.5
    img += glow(L.astype(np.float32) / 255, 0.7)
    return img

def tilt(img, ang_y, ang_x=0.0, zoom=1.0):
    """Perspective tilt of a flat UI frame (radians)."""
    f = 1800
    pts = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    out = []
    for (x, y) in pts:
        X, Y = (x - W / 2) * zoom, (y - H / 2) * zoom
        Zc = f + X * math.sin(ang_y) + Y * math.sin(ang_x)
        Xr = X * math.cos(ang_y); Yr = Y * math.cos(ang_x)
        out.append([W / 2 + f * Xr / Zc, H / 2 + f * Yr / Zc])
    Hm = cv2.getPerspectiveTransform(pts, np.float32(out))
    return cv2.warpPerspective(img, Hm, (W, H), borderMode=cv2.BORDER_REPLICATE)

# ---------- month-end close ----------
TASKS = ['Bank reconciliations', 'AR / AP sub-ledgers', 'Accruals & prepaids', 'Intercompany', 'Audit trail & sign-off']

def close_ui(u):
    img = BG.copy()
    hp = eo(prog(u, 0, 0.5))
    over(img, text_rgba('MONTH-END CLOSE', F('Inter-SemiBold.otf', 32), (0.55, 0.85, 1.0), tracking=8), W / 2, 190 - (1 - hp) * 30, alpha=hp, anchor='ct')
    over(img, text_rgba('OCTOBER 2026', F('Inter-Medium.otf', 28), (0.5, 0.6, 0.72), tracking=6), W / 2, 240, alpha=hp, anchor='ct')
    # ring
    p = eio(prog(u, 0.3, 2.4))
    cx, cy, R = W // 2, 600, 230
    L = u8()
    cv2.circle(L, (cx, cy), R, (25, 45, 65), 18, cv2.LINE_AA)
    if p > 0:
        cv2.ellipse(L, (cx, cy), (R, R), -90, 0, 360 * p, c8(GREEN if p >= 1 else CYAN), 18, cv2.LINE_AA)
    img += glow(L.astype(np.float32) / 255, 0.9 if p >= 1 else 0.6)
    over(img, text_rgba(f'{int(round(p * 100))}%', F('InterDisplay-SemiBold.otf', 150), (1, 1, 1)), cx, cy, alpha=hp, anchor='cc')
    over(img, text_rgba('CLOSED' if p >= 1 else 'IN PROGRESS', F('Inter-SemiBold.otf', 30), GREEN if p >= 1 else (0.55, 0.65, 0.78), tracking=6), cx, cy + 110, alpha=hp, anchor='ct')
    # tasks
    C = u8()
    for i, tsk in enumerate(TASKS):
        a = eo(prog(u, 0.2 + i * 0.08, 0.6 + i * 0.08))
        if a <= 0: continue
        y = 930 + i * 92
        done = p >= (i + 1) / len(TASKS) - 0.001
        rounded(img, 90, y, 990, y + 76, (0.03, 0.07, 0.12), 0.8 * a, border=(0.12, 0.4, 0.6) if not done else (0.2, 0.8, 0.45), r=18)
        over(img, text_rgba(tsk, F('Inter-Medium.otf', 34), (0.9, 0.95, 1.0)), 130, y + 14, alpha=a)
        if done:
            check(C, 940, y + 38, 13, c8(GREEN))
        else:
            cv2.circle(C, (940, y + 38), 12, (60, 90, 110), 2, cv2.LINE_AA)
    img += glow(C.astype(np.float32) / 255, 0.6)
    # stats
    sp = eo(prog(u, 2.3, 2.8))
    if sp > 0:
        v = int(round(85 * eio(prog(u, 2.3, 3.0))))
        over(img, text_rgba(f'{v}%', F('InterDisplay-Bold.otf', 120), GREEN), 290, 1420 + (1 - sp) * 30, alpha=sp, anchor='ct')
        over(img, text_rgba('TIME SAVED', F('Inter-SemiBold.otf', 28), (0.6, 0.7, 0.8), tracking=6), 290, 1570, alpha=sp, anchor='ct')
        v2 = int(round(100 * eio(prog(u, 2.4, 3.1))))
        over(img, text_rgba(f'{v2}%', F('InterDisplay-Bold.otf', 120), (1, 1, 1)), 790, 1420 + (1 - sp) * 30, alpha=sp, anchor='ct')
        over(img, text_rgba('AUDIT-READY', F('Inter-SemiBold.otf', 28), (0.6, 0.7, 0.8), tracking=6), 790, 1570, alpha=sp, anchor='ct')
    return img

# ---------- stats grid (video B) ----------
STATS = [('99.9%', 'RECONCILIATION\nACCURACY', 99.9, '{:.1f}%'), ('85%', 'TIME\nSAVED', 85, '{:.0f}%'),
         ('48hrs', 'AVERAGE\nRESOLUTION', 48, '{:.0f}hrs'), ('100%', 'AUDIT\nCOMPLIANCE', 100, '{:.0f}%')]
def stats_ui(u):
    img = BG.copy()
    for i, (_, lab, val, fm) in enumerate(STATS):
        cx = 290 + (i % 2) * 500; cy = 560 + (i // 2) * 560
        a = eo(prog(u, 0.1 + i * 0.12, 0.6 + i * 0.12))
        if a <= 0: continue
        rounded(img, cx - 230, cy - 230 + int((1 - a) * 60), cx + 230, cy + 230 + int((1 - a) * 60), (0.03, 0.07, 0.12), 0.85 * a, border=(0.15, 0.55, 0.8), r=30)
        v = val * eio(prog(u, 0.2 + i * 0.12, 1.4 + i * 0.12))
        col = GREEN if i == 0 else (1, 1, 1)
        over(img, text_rgba(fm.format(v), F('InterDisplay-Bold.otf', 104), col), cx, cy - 70 + (1 - a) * 60, alpha=a, anchor='cc')
        for k, ln in enumerate(lab.split('\n')):
            over(img, text_rgba(ln, F('Inter-SemiBold.otf', 30), (0.6, 0.72, 0.85), tracking=5), cx, cy + 40 + k * 44 + (1 - a) * 60, alpha=a, anchor='ct')
    return img

# ---------- automatic journal entries ----------
JES = [
    ('JE-20431', 'FX revaluation · USD settlements', [('Dr', 'FX Gain/Loss', 4812.36), ('Cr', 'Settlements Clearing', 4812.36)]),
    ('JE-20432', 'Bank charges · statement 06 Oct', [('Dr', 'Bank Charges', 1240.00), ('Cr', 'Cash at Bank', 1240.00)]),
    ('JE-20433', 'Payout clearing · batch 7781', [('Dr', 'Payouts Payable', 86420.15), ('Cr', 'Cash at Bank', 86420.15)]),
    ('JE-20434', 'Month-end accrual · processor fees', [('Dr', 'Processing Fees', 12905.80), ('Cr', 'Accrued Liabilities', 12905.80)]),
    ('JE-20435', 'Revenue recognition · October', [('Dr', 'Deferred Revenue', 248300.00), ('Cr', 'Revenue', 248300.00)]),
    ('JE-20436', 'Chargeback reversal · case 4410', [('Dr', 'Chargebacks', 3150.00), ('Cr', 'Merchant Receivable', 3150.00)]),
    ('JE-20437', 'Intercompany settlement · ZA → UK', [('Dr', 'Intercompany Payable', 57200.00), ('Cr', 'Cash at Bank', 57200.00)]),
    ('JE-20438', 'Prepaid amortisation · licences', [('Dr', 'Software Expense', 9875.00), ('Cr', 'Prepaid Expenses', 9875.00)]),
]
JF = MONO(28); JFB = MONOB(28); JFS = F('Inter-Medium.otf', 26)

def je_card(je, typed, posted, w=940):
    ref, desc, lines = je
    h = 300
    img = Image.new('RGBA', (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, w - 1, h - 1), 24, fill=(8, 18, 32, 225), outline=(50, 190, 255, 200) if not posted else (60, 220, 130, 220), width=2)
    d.text((30, 22), ref, font=JFB, fill=(235, 245, 255, 255))
    d.text((210, 26), '31 OCT 2026  ·  AUTO', font=MONO(22), fill=(110, 150, 180, 255))
    d.text((30, 68), desc, font=JFS, fill=(140, 180, 210, 255))
    d.line((30, 112, w - 30, 112), fill=(40, 90, 120, 255), width=1)
    y = 132
    for i, (dc, acct, amt) in enumerate(lines):
        if typed <= i: break
        indent = 30 if dc == 'Dr' else 90
        d.text((indent, y), dc, font=JFB, fill=(80, 200, 255, 255) if dc == 'Dr' else (255, 190, 90, 255))
        d.text((indent + 60, y), acct, font=JF, fill=(230, 240, 250, 255))
        amt_s = 'R ' + f'{amt:,.2f}'.replace(',', ' ')
        col_x = w - 330 if dc == 'Dr' else w - 40
        tw = d.textlength(amt_s, font=JF)
        d.text((col_x - tw, y), amt_s, font=JF, fill=(230, 240, 250, 255))
        y += 52
    if typed >= 2:
        d.text((30, 248), 'Dr = Cr  ✓ BALANCED', font=MONO(22), fill=(110, 220, 150, 255))
    if posted:
        d.rounded_rectangle((w - 210, 236, w - 30, 280), 10, outline=(60, 220, 130, 255), width=3)
        d.text((w - 186, 242), 'POSTED', font=MONOB(28), fill=(60, 230, 130, 255))
    return np.asarray(img, np.float32) / 255

def journal_ui(u, speed=1.0):
    u = u * speed
    img = BG.copy()
    hp = eo(prog(u, 0, 0.5))
    over(img, text_rgba('AUTOMATED JOURNAL ENTRIES', F('Inter-SemiBold.otf', 32), (0.55, 0.85, 1.0), tracking=7), 60, 190 - (1 - hp) * 30, alpha=hp)
    n = int(2418 * eio(prog(u, 0.3, 3.4)))
    over(img, text_rgba(f'{n:,}', F('InterDisplay-SemiBold.otf', 124), (1, 1, 1)), 52, 240, alpha=hp)
    over(img, text_rgba('JOURNALS CREATED & POSTED', F('Inter-Medium.otf', 28), (0.55, 0.65, 0.78), tracking=5), 62, 400, alpha=hp)
    over(img, text_rgba('0', F('InterDisplay-SemiBold.otf', 64), GREEN), W - 60, 270, alpha=hp, anchor='rt')
    over(img, text_rgba('MANUAL ENTRIES', F('Inter-Medium.otf', 26), (0.55, 0.65, 0.78), tracking=5), W - 62, 350, alpha=hp, anchor='rt')
    # cascading cards: each new card enters at top of stack, older ones push down & fade
    period = 0.62
    k_now = u / period
    G = u8()
    for i in range(len(JES)):
        age = k_now - i - 0.3
        if age < 0: continue
        newer = max(0.0, k_now - 0.3 - (i + 1))
        fl = math.floor(newer); fr = newer - fl
        pos = 500 + 320 * (fl + eo(clamp(fr / 0.35)))
        if pos > 1500: continue
        enter = eo(clamp((age - 0.3) / 0.3))
        typed = int(clamp((age - 0.4) / 0.3) * 2 + 0.001)
        posted = age > 0.85
        card = je_card(JES[i], typed, posted)
        a = enter * (1 - prog(pos, 1150, 1500))
        over(img, card, 70 - (1 - enter) * 60, pos, alpha=a)
        if posted and age < 1.2:
            p = prog(age, 0.85, 1.2)
            cv2.rectangle(G, (70, int(pos)), (1010, int(pos) + 300), c8(GREEN, 1 - p), 3, cv2.LINE_AA)
    img += glow(G.astype(np.float32) / 255, 0.9)
    return img
