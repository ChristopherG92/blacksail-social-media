import sys
from common import *
from sat import *
from servers import corridor
from logo import end_logo
from recon import recon_ui, close_ui, tilt, journal_ui

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sign_clip.mp4')
raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', SRC, '-vf', 'scale=1080:1920:flags=lanczos,unsharp=5:5:0.45',
                      '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True).stdout
CLIP = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
CEND = len(CLIP) - 1

T_TOTAL = 28.0
T0F = np.array(TGT[REF])

def zcurve(t):
    if t < 0.4: return 11.3
    if t < 6.6: return 11.3 + (19.25 - 11.3) * eio(prog(t, 0.4, 6.6))
    if t < 7.9: return 19.25 + 0.12 * prog(t, 6.6, 7.9)
    return 19.37 + 2.3 * ei(prog(t, 7.9, 8.6))

def camf(Z):
    b = eio(clamp((Z - 15.5) / 2.85))
    return T0F + (FOCUS18 - T0F) * b

def sat_frame(t, f):
    Z = zcurve(t)
    th = -0.18 * (1 - clamp((Z - 11.3) / 7.95))
    cam = T0F
    shake = (math.sin(t * 13.1) * 1.2 + math.sin(t * 7.3) * 0.8, math.cos(t * 11.7) * 1.2)
    img = sat_view(Z, cam, th, W / 2 + shake[0], H / 2 + shake[1])
    draw_clouds(img, Z, th)
    img = sensor_grade(img, Z)
    if t > 7.9:
        sp = ei(prog(t, 7.9, 8.6))
        img = zoom_blur(img, 1.0, W / 2, H / 2, 0.25 * sp, n=6)
    # edge scan wireframe
    if t > 6.95:
        g8 = (np.clip(img.mean(2), 0, 1) * 255).astype(np.uint8)
        e = cv2.Canny(cv2.GaussianBlur(g8, (3, 3), 0), 50, 130).astype(np.float32) / 255
        sweep = H * eo(prog(t, 6.95, 7.75))
        mask = (YY < sweep).astype(np.float32)
        front = np.exp(-((YY - sweep) / 18) ** 2)
        lay = e[..., None] * mask[..., None] * CYAN * (0.9 + 1.5 * prog(t, 7.9, 8.5))
        img = img * (1 - 0.55 * mask[..., None] * prog(t, 6.95, 7.4)) + glow(lay, 0.8)
        img += front[..., None] * CYAN * 0.6 * (1 - prog(t, 7.6, 7.8))
    # night vision flash at lock
    if 6.85 < t < 7.1:
        img = night_vision(img, f)
    lockp = prog(t, 5.7, 6.75) if t < 6.75 else 1.0
    if t >= 6.75: lockp = 1.0
    ha = 1 - prog(t, 7.9, 8.3)
    img = scanlines(img, 0.06)
    if ha > 0:
        hud(img, t, Z, cam, th, lockp=lockp if t > 5.7 else 0.0, mode='DESCENT' if t < 6.75 else 'LOCK / SCAN', alpha=ha, f=f)
    # boot-up
    if t < 0.7:
        r = np.random.default_rng(f)
        nz = cv2.resize(r.random((H // 6, W // 6)).astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)[..., None]
        a = prog(t, 0.15, 0.7)
        img = img * a + nz * 0.35 * (1 - a)
        if t < 0.5:
            over(img, text_rgba('ESTABLISHING SAT-LINK…', HUDFB, HUDG), W / 2, H / 2 - 20, alpha=1 - prog(t, 0.35, 0.5), anchor='ct')
    caption(img, 'Cape Town.', t, 1.0, 3.4, y=1440)
    caption(img, 'The Forum, Century City.', t, 3.9, 6.3, y=1440)
    return img

def frame(f):
    t = f / FPS
    gr = 0.018
    if t < 8.6:
        img = sat_frame(t, f)
        if t > 8.3:
            img = img + prog(t, 8.3, 8.6) ** 2 * np.array([0.7, 0.9, 1.0], np.float32) * 1.4
    elif t < 12.6:
        u = t - 8.6
        img = corridor(u + 0.3, intensity=1.0 + 3.0 * ei(prog(t, 12.1, 12.6)))
        p = prog(t, 8.6, 9.1)
        if p < 1:
            img = zoom_blur(img, 0.6 + 0.4 * eo(p), W / 2, H * 0.47, 0.2 * (1 - p), n=6)
            img = img + (1 - eo(p)) * np.array([0.7, 0.9, 1.0], np.float32) * 1.4
        caption(img, 'Millions of transactions.', t, 9.2, 12.0, y=1500)
        if t > 12.3:
            img = img + prog(t, 12.3, 12.6) ** 2 * np.array([0.7, 0.9, 1.0], np.float32) * 1.2
    elif t < 17.2:
        u = t - 12.6
        ui = recon_ui(u)
        ui = tilt(ui, 0.10 * math.sin(u * 0.45 + 0.6), 0.04, zoom=1.0 - 0.03 * u / 4.6)
        p = prog(t, 12.6, 13.0)
        if p < 1:
            ui = zoom_blur(ui, 0.85 + 0.15 * eo(p), W / 2, H / 2, 0.12 * (1 - p), n=5)
            ui = ui + (1 - eo(p)) * np.array([0.7, 0.9, 1.0], np.float32) * 1.2
        caption(ui, 'Reconciled automatically.', t, 13.4, 16.8, y=1600)
        img = ui
        if t > 16.95:
            img = glitch(img, f, prog(t, 16.95, 17.2))
    elif t < 21.2:
        u = t - 17.2
        img = journal_ui(u, speed=1.0)
        img = tilt(img, -0.08 * math.sin(u * 0.5 + 0.3), 0.03)
        if t < 17.45:
            img = glitch(img, f, 1 - prog(t, 17.2, 17.45))
        caption(img, 'Journals, created automatically.', t, 17.6, 20.9, y=1620)
        if t > 20.95:
            img = glitch(img, f, prog(t, 20.95, 21.2))
    elif t < 24.7:
        u = t - 21.2
        img = close_ui(u * 1.2)
        img = tilt(img, -0.08 * math.sin(u * 0.5 + 0.3), 0.03, zoom=1.0)
        if t < 21.45:
            img = glitch(img, f, 1 - prog(t, 21.2, 21.45))
        if t > 24.4:
            img = img + prog(t, 24.4, 24.7) ** 2 * np.array([1, 1, 1], np.float32) * 1.4
    else:
        u = t - 24.7
        img = end_logo(u)
        p = prog(t, 24.7, 25.05)
        if p < 1: img = img + (1 - eo(p)) * np.array([1, 1, 1], np.float32) * 1.2
    img = img * VIGN
    img = chroma(img, 1)
    img = grain(img, f, gr)
    return np.clip(img, 0, 1)

if __name__ == '__main__':
    if sys.argv[1] == 'test':
        fs = [int(x) for x in sys.argv[2].split(',')]
        ims = [cv2.resize((frame(f) * 255).astype(np.uint8)[:, :, ::-1], (270, 480)) for f in fs]
        rows = [np.hstack(ims[i:i + 8]) for i in range(0, len(ims), 8)]
        if len(rows) > 1 and rows[-1].shape != rows[0].shape:
            rows[-1] = np.hstack([rows[-1], np.zeros((480, rows[0].shape[1] - rows[-1].shape[1], 3), np.uint8)])
        cv2.imwrite('t_A.jpg', np.vstack(rows))
    else:
        a, b = int(sys.argv[2]), int(sys.argv[3])
        w = Writer(f'A_{a:04d}.mp4')
        for f in range(a, b):
            w.write(frame(f))
        w.close()
