import sys
from common import *
from logo import end_logo
FL = np.array([0.7, 0.9, 1.0], np.float32)
r = np.random.default_rng(11)
NP = 300; N = NP * 2
COLS, ROWS = 10, 30
TW, TH = 0.17, 0.095
# chaos state
P0 = r.normal(0, 1, (N, 3)); P0 /= np.linalg.norm(P0, axis=1, keepdims=True); P0 *= r.uniform(0.6, 3.2, (N, 1))
A0 = r.uniform(-math.pi, math.pi, (N, 3))
SPIN = r.normal(0, 0.6, (N, 3))
# targets
pid = np.arange(N) // 2; side = np.arange(N) % 2  # 0 bank (left), 1 ledger (right)
col = pid % COLS; row = pid // COLS
GX = (col - (COLS - 1) / 2) * 0.42; GY = ((ROWS - 1) / 2 - row) * 0.13
OFF_SEP = 0.11
TCOL0 = np.where(r.random(N) < 0.55, 0, np.where(r.random(N) < 0.5, 1, 2))  # 0 red,1 amber,2 grey
COLS_C = [np.array([1.0, 0.3, 0.36]), np.array([1.0, 0.7, 0.22]), np.array([0.45, 0.52, 0.62])]
fly_delay = (np.abs(GY) / 2.0 + r.uniform(0, 0.5, N)) * 0.9
snap_t = 7.4 + (row / ROWS) * 1.4 + r.uniform(0, 0.15, NP)[pid]
DUST = r.uniform([-5, -6, -5], [5, 6, 5], (260, 3))

def rotm(a):
    cx, sx = math.cos(a[0]), math.sin(a[0]); cy, sy = math.cos(a[1]), math.sin(a[1]); cz, sz = math.cos(a[2]), math.sin(a[2])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]); Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]); Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def camera(t):
    th = 0.55 * math.sin(t * 0.25) + 0.25
    dist = 6.2 - 1.2 * eio(prog(t, 4.5, 8.0)) + 2.6 * eio(prog(t, 8.8, 11.4))
    tilt = 0.18 * math.sin(t * 0.3) - 0.25 * eio(prog(t, 8.8, 11.4))
    th = th * (1 - eio(prog(t, 6.0, 8.5))) + (-0.35 * eio(prog(t, 8.8, 11.4)))
    pos = np.array([dist * math.sin(th) * math.cos(tilt), dist * math.sin(tilt), -dist * math.cos(th) * math.cos(tilt)])
    fwd = -pos / np.linalg.norm(pos); up = np.array([0, 1, 0.0])
    rgt = np.cross(up, fwd); rgt /= np.linalg.norm(rgt); up2 = np.cross(fwd, rgt)
    return pos, np.stack([rgt, up2, fwd])

def frame(f):
    t = f / FPS
    if t >= 11.8:
        img = end_logo(t - 11.8)
        if t < 12.15: img += (1 - eo(prog(t, 11.8, 12.15))) * FL * 1.2
        return np.clip(grain(img * VIGN, f, 0.018), 0, 1)
    img = BG.copy() * 0.6
    cam, Rc = camera(t)
    foc = 1150
    def proj(p):
        v = (p - cam) @ Rc.T
        return W / 2 + foc * v[..., 0] / v[..., 2], H / 2 - foc * v[..., 1] / v[..., 2], v[..., 2]
    # dust
    dx, dy, dz = proj(DUST)
    Dl = u8()
    for x, y, z in zip(dx, dy, dz):
        if z > 0.5 and 0 <= x < W and 0 <= y < H:
            cv2.circle(Dl, (int(x), int(y)), max(1, int(4 / z)), c8(CYAN, clamp(0.5 / z * 2) * 0.5), -1, cv2.LINE_AA)
    img += cv2.GaussianBlur(Dl.astype(np.float32) / 255, (0, 0), 1.5)
    # scan pulse
    pulse_r = (t - 4.8) * 4.0
    # tile states
    fl = np.clip((t - 5.2 - fly_delay) / 1.6, 0, 1); fe = np.array([eio(x) for x in fl])
    chaos_pos = P0 + 0.08 * np.sin(t * 0.7 + P0 * 3)
    tgt = np.stack([GX + np.where(side == 0, -1, 1) * (TW / 2 + OFF_SEP * (1 - np.clip((t - snap_t[pid]) / 0.25, 0, 1))), GY, np.zeros(N)], 1)
    pos = chaos_pos * (1 - fe[:, None]) + tgt * fe[:, None]
    # arc in flight
    pos[:, 2] += -np.sin(fe * math.pi) * 0.8
    ang = (A0 + SPIN * t) * (1 - fe[:, None])
    snapped = t > snap_t[pid] + 0.25
    scanned = np.linalg.norm(P0, axis=1) < pulse_r
    tiles = []
    corners = np.array([[-TW / 2, -TH / 2, 0], [TW / 2, -TH / 2, 0], [TW / 2, TH / 2, 0], [-TW / 2, TH / 2, 0]])
    for i in range(N):
        Rm = rotm(ang[i])
        C = corners @ Rm.T + pos[i]
        x, y, z = proj(C)
        if (z < 0.3).any(): continue
        nrm = Rm[:, 2]
        facing = abs(float(nrm @ (Rc[2])))
        tiles.append((z.mean(), i, np.stack([x, y], 1), facing))
    tiles.sort(key=lambda a: -a[0])
    layers = [u8(), u8(), u8()]  # far, mid, near
    for zm, i, pts, facing in tiles:
        L = layers[0] if zm > 6.5 else (layers[2] if zm < 3.6 else layers[1])
        if snapped[i]: c = GREEN
        elif scanned[i] or fe[i] > 0: c = CYAN
        else: c = COLS_C[TCOL0[i]]
        sh = 0.25 + 0.75 * facing
        fill = c * 0.18 * sh + 0.02
        ip = pts.astype(np.int32)
        cv2.fillConvexPoly(L, ip, c8(fill), cv2.LINE_AA)
        cv2.polylines(L, [ip], True, c8(c, 0.55 + 0.45 * sh), 1, cv2.LINE_AA)
        if zm < 7.5:
            for k, (a0, a1) in enumerate([(0.12, 0.62), (0.12, 0.42)]):
                yy = 0.38 + k * 0.28
                p0 = pts[0] + (pts[1] - pts[0]) * a0 + (pts[3] - pts[0]) * yy
                p1 = pts[0] + (pts[1] - pts[0]) * a1 + (pts[3] - pts[0]) * yy
                cv2.line(L, tuple(p0.astype(int)), tuple(p1.astype(int)), c8(c, 0.6 * sh), 1, cv2.LINE_AA)
        if snapped[i] and t < snap_t[pid[i]] + 0.6:
            cv2.polylines(L, [ip], True, c8((1, 1, 1), 1 - (t - snap_t[pid[i]] - 0.25) / 0.35), 2, cv2.LINE_AA)
    far = cv2.GaussianBlur(layers[0].astype(np.float32) / 255, (0, 0), 2.5)
    mid = layers[1].astype(np.float32) / 255
    near = cv2.GaussianBlur(layers[2].astype(np.float32) / 255, (0, 0), 3.5)
    tl = far + mid + near
    img = img * (1 - np.clip(tl.max(2, keepdims=True) * 3, 0, 0.85)) + tl
    img += glow(tl, 0.55)
    # pulse shell (screen-space ring approx)
    if 4.8 < t < 6.2:
        c0 = proj(np.zeros((1, 3)))
        rr = foc * pulse_r / np.linalg.norm(cam)
        Pl = u8(); cv2.circle(Pl, (int(c0[0][0]), int(c0[1][0])), int(rr), c8(CYAN, 1 - prog(t, 4.8, 6.2)), 6, cv2.LINE_AA)
        img += glow(Pl.astype(np.float32) / 255, 1.4)
        img += np.exp(-((t - 4.85) / 0.08) ** 2) * FL * 0.4
    # HUD counters
    nsn = int(snapped.sum() // 2)
    m = int(1_248_903 * eio(prog(t, 7.4, 9.6)))
    a = eo(prog(t, 7.0, 7.5))
    if a > 0:
        over(img, text_rgba('MATCHED', F('Inter-SemiBold.otf', 30), GREEN, tracking=8), W / 2, 150, alpha=a, anchor='ct')
        over(img, text_rgba(f'{m:,}', F('InterDisplay-SemiBold.otf', 120), (1, 1, 1)), W / 2, 195, alpha=a, anchor='ct')
    ua = 1 - prog(t, 4.6, 5.0)
    if ua > 0:
        nu = int(1_248_903 * eio(prog(t, 0.0, 2.2)))
        over(img, text_rgba('UNMATCHED', F('Inter-SemiBold.otf', 30), RED, tracking=8), W / 2, 150, alpha=ua, anchor='ct')
        over(img, text_rgba(f'{nu:,}', F('InterDisplay-SemiBold.otf', 120), (1, 1, 1)), W / 2, 195, alpha=ua, anchor='ct')
    caption(img, 'Your month-end, as a spreadsheet.', t, 0.4, 2.6, y=1560, size=80)
    caption(img, 'Now watch this.', t, 2.9, 4.6, y=1560, size=90)
    caption(img, 'Matched. Automatically.', t, 9.2, 11.5, y=1600, size=88, sub='99.9% RECONCILIATION ACCURACY')
    if t > 11.5: img += prog(t, 11.5, 11.8) ** 2 * FL * 1.3
    img = img * VIGN
    img = chroma(img, 1)
    img = grain(img, f, 0.018)
    return np.clip(img, 0, 1)

if __name__ == '__main__':
    if sys.argv[1] == 'test':
        fs = [int(x) for x in sys.argv[2].split(',')]
        ims = [cv2.resize((frame(f) * 255).astype(np.uint8)[:, :, ::-1], (270, 480)) for f in fs]
        cv2.imwrite('t_v2.jpg', np.hstack(ims))
    else:
        a, b = int(sys.argv[2]), int(sys.argv[3]); w = Writer(f'v2_{a:04d}.mp4')
        for f in range(a, b): w.write(frame(f))
        w.close()
