from common import *
from global_land_mask import globe as _g

# land dot cloud
_lats = np.arange(-84, 84, 0.9); _lons = np.arange(-180, 180, 0.9)
LA, LO = np.meshgrid(_lats, _lons, indexing='ij')
m = _g.is_land(LA, LO)
DLAT = np.radians(LA[m]); DLON = np.radians(LO[m])
def ll2xyz(lat, lon):
    return np.stack([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)], -1)
DOTS = ll2xyz(DLAT, DLON)
_r = np.random.default_rng(1)
DOTB = _r.uniform(0.55, 1.0, len(DOTS))

CITIES = {
    'CPT': (-33.92, 18.42), 'JNB': (-26.2, 28.05), 'LON': (51.5, -0.12), 'FRA': (50.11, 8.68),
    'NYC': (40.71, -74.0), 'SIN': (1.35, 103.8), 'DXB': (25.2, 55.27), 'NBO': (-1.29, 36.82),
    'LOS': (6.52, 3.38), 'AMS': (52.37, 4.9), 'SAO': (-23.55, -46.63), 'MUM': (19.07, 72.88)}

def rot(lon0, lat0):
    a, b = math.radians(-lon0), math.radians(lat0)
    Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    return Rx @ Ry

def project(P, R, cx, cy, Rad):
    Q = P @ R.T
    return cx + Q[..., 0] * Rad, cy - Q[..., 1] * Rad, Q[..., 2]

def city_xyz(c): la, lo = CITIES[c]; return ll2xyz(np.radians(la), np.radians(lo))

def arc_pts(a, b, n=80, lift=0.25):
    pa, pb = city_xyz(a), city_xyz(b)
    om = math.acos(np.clip(pa @ pb, -1, 1))
    t = np.linspace(0, 1, n)
    s = (np.sin((1 - t) * om)[:, None] * pa + np.sin(t * om)[:, None] * pb) / math.sin(om)
    h = 1 + lift * om / math.pi * 2 * np.sin(np.pi * t)
    nrm = np.cross(pa, pb); nrm /= (np.linalg.norm(nrm) + 1e-9)
    s = s * h[:, None] + nrm[None] * (0.35 * om / math.pi * np.sin(np.pi * t))[:, None]
    return s

STARS = np.random.default_rng(4).uniform([0, 0, 0.2], [W, H, 1.0], (500, 3))

def draw_globe(img, lon0, lat0, cx, cy, Rad, t, arcs=(), ambient=True, nodes=()):
    R = rot(lon0, lat0)
    # stars
    S = np.zeros((H, W), np.float32)
    sx = STARS[:, 0].astype(int); sy = STARS[:, 1].astype(int)
    S[sy.clip(0, H - 1), sx.clip(0, W - 1)] = STARS[:, 2] * (0.6 + 0.4 * np.sin(t * 2 + STARS[:, 0]))
    img += cv2.GaussianBlur(S, (0, 0), 0.8)[..., None] * 0.9
    # ocean sphere
    d2 = ((XX - cx) ** 2 + (YY - cy) ** 2) / Rad ** 2
    inside = d2 < 1
    z = np.sqrt(np.clip(1 - d2, 0, 1))
    shade = (0.35 + 0.65 * z) * inside
    img[:] = img * (1 - inside[..., None]) + (shade[..., None] * np.array([0.02, 0.06, 0.11], np.float32))
    # land dots
    x, y, zz = project(DOTS, R, cx, cy, Rad)
    vis = (zz > 0.02) & (x >= 1) & (x < W - 2) & (y >= 1) & (y < H - 2)
    L = np.zeros((H, W), np.float32)
    xi = x[vis].astype(int); yi = y[vis].astype(int); br = (0.25 + 0.75 * zz[vis] ** 0.7) * DOTB[vis]
    sz = max(2, int(Rad / 300))
    for dx in range(sz):
        for dy in range(sz):
            np.maximum.at(L, (np.clip(yi + dy, 0, H - 1), np.clip(xi + dx, 0, W - 1)), br)
    land = L[..., None] * np.array([0.35, 0.8, 1.0], np.float32)
    img += land * 1.25 + glow(land, 0.6, sigmas=(3, 12), weights=(0.6, 0.4))
    # atmosphere rim
    r = np.sqrt(d2)
    rim = np.exp(-((r - 1) / 0.035) ** 2) * 0.55 + np.exp(-np.clip(r - 1, 0, None) / 0.08) * (r > 1) * 0.35
    img += rim[..., None] * np.array([0.2, 0.55, 1.0], np.float32)
    # arcs
    A = u8()
    def draw_arc(P, prog_, col, thick, head=True, fade=1.0):
        px, py, pz = project(P, R, cx, cy, Rad)
        n = len(P); k = max(2, int(n * prog_))
        occl = (pz < 0) & (((px - cx) ** 2 + (py - cy) ** 2) < Rad ** 2)
        pts = []
        for i in range(k):
            if occl[i]:
                if len(pts) > 1: cv2.polylines(A, [np.array(pts, np.int32)], False, c8(col, fade), thick, cv2.LINE_AA)
                pts = []; continue
            pts.append((int(px[i]), int(py[i])))
        if len(pts) > 1: cv2.polylines(A, [np.array(pts, np.int32)], False, c8(col, fade), thick, cv2.LINE_AA)
        if head and 0 < prog_ < 1 and not occl[k - 1]:
            cv2.circle(A, (int(px[k - 1]), int(py[k - 1])), thick * 3, c8((1, 1, 1), fade), -1, cv2.LINE_AA)
    if ambient:
        rr = np.random.default_rng(9)
        keys = list(CITIES)
        for i in range(14):
            a, b = rr.choice(keys, 2, replace=False)
            ph = (t * 0.35 + i / 14) % 1
            P = arc_pts(a, b, 60, 0.18)
            # travelling segment
            k0 = int(ph * 60); seg = P[max(0, k0 - 14):k0 + 1]
            if len(seg) > 1: draw_arc(seg, 1.0, CYAN, 1, head=False, fade=0.35)
    for (a, b, p, col) in arcs:
        draw_arc(arc_pts(a, b, 120, 0.55), p, col, 4)
    for c in nodes:
        P = city_xyz(c)[None]
        px, py, pz = project(P, R, cx, cy, Rad)
        if pz[0] > 0:
            ph = (t * 1.2) % 1
            cv2.circle(A, (int(px[0]), int(py[0])), 7, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(A, (int(px[0]), int(py[0])), int(10 + 40 * ph), c8(CYAN, 1 - ph), 2, cv2.LINE_AA)
    img += glow(A.astype(np.float32) / 255, 1.2)
    return R

def city_screen(c, R, cx, cy, Rad):
    P = city_xyz(c)[None]; px, py, pz = project(P, R, cx, cy, Rad)
    return float(px[0]), float(py[0]), float(pz[0])
