import numpy as np
from scipy.signal import butter, sosfilt
SR = 48000

class Mix:
    def __init__(self, dur, seed=5):
        self.N = int(dur * SR); self.out = np.zeros((self.N, 2)); self.r = np.random.default_rng(seed); self.dur = dur
    def S(self, t): return int(t * SR)
    def place(self, sig, t, gain=1.0, pan=0.0):
        i = self.S(t)
        if i >= self.N: return
        sig = np.asarray(sig)[:self.N - i]
        if sig.ndim == 1: sig = np.stack([sig * (1 - pan), sig * (1 + pan)], 1)
        self.out[i:i + len(sig)] += sig * gain
    def noise(self, d): return self.r.normal(0, 1, int(d * SR))

def bp(x, lo, hi): return sosfilt(butter(2, [lo, hi], 'band', fs=SR, output='sos'), x)
def lp(x, f): return sosfilt(butter(2, f, 'low', fs=SR, output='sos'), x)
def hp(x, f): return sosfilt(butter(2, f, 'high', fs=SR, output='sos'), x)
def tt(d): return np.arange(int(d * SR)) / SR

def sweep_noise(m, d, f0, f1, chunks=50):
    x = m.noise(d); seg = np.array_split(np.arange(len(x)), chunks); y = np.zeros_like(x)
    for k, s in enumerate(seg):
        fc = f0 * (f1 / f0) ** (k / (chunks - 1)); y[s] = bp(x, fc * 0.7, min(fc * 1.4, SR / 2 - 100))[s]
    return y

def riser(m, t, d, gain=0.4):
    x = np.linspace(0, 1, int(d * SR))
    s = sweep_noise(m, d, 300, 7000) * x ** 2 * 0.35 + np.sin(2 * np.pi * np.cumsum(180 * (6 ** x)) / SR) * x ** 3 * 0.1
    m.place(s, t, gain)

def impact(m, t, gain=1.0):
    x = tt(1.6); f = 34 + 45 * np.exp(-x * 8)
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-x * 2.2)
    hit = lp(m.noise(1.6), 2500) * np.exp(-x * 14) * 0.6
    m.place(sub * 0.9 + hit, t, gain)

def whoosh(m, t, d=0.6, gain=0.5, up=True):
    x = np.linspace(0, 1, int(d * SR))
    s = sweep_noise(m, d, 400 if up else 5000, 5000 if up else 400) * np.sin(np.pi * x) ** 2
    m.place(s, t, gain)

def ping(m, t, f=1150, gain=0.12):
    x = tt(1.2); s = np.sin(2 * np.pi * f * x) * np.exp(-x * 5)
    m.place(s, t, gain); m.place(s, t + 0.18, gain * 0.35, pan=0.4); m.place(s, t + 0.36, gain * 0.15, pan=-0.4)

def beep(m, t, f=1800, d=0.07, gain=0.08):
    x = tt(d); m.place(np.sin(2 * np.pi * f * x) * np.minimum(1, (d - x) * 200), t, gain)

def blip(m, t, f, gain=0.05, pan=0.0):
    x = tt(0.12); m.place(np.sin(2 * np.pi * f * x) * np.exp(-x * 35), t, gain, pan)

def click(m, t, gain=0.1, pan=0.0):
    c = hp(m.noise(0.012), 2000) * np.exp(-tt(0.012) * 500); m.place(c, t, gain, pan)

def drone(m, t0, t1, gain=0.6, notes=((41.2, .3), (82.4, .14), (123.5, .07), (164.8, .05))):
    d = t1 - t0; x = tt(d)
    s = sum(np.sin(2 * np.pi * fr * x + i) * g for i, (fr, g) in enumerate(notes))
    s += lp(m.noise(d), 300) * 0.05
    s *= np.minimum(1, x / 1.2) * np.minimum(1, (d - x) / 0.6) * (0.85 + 0.15 * np.sin(2 * np.pi * x * 0.2))
    m.place(np.stack([s, np.roll(s, 300)], 1), t0, gain)

def beat(m, t0, t1, bpm=120, gain=0.5):
    bt = 60 / bpm
    kt = tt(0.35); kick = np.sin(2 * np.pi * np.cumsum(50 + 90 * np.exp(-kt * 30)) / SR) * np.exp(-kt * 9)
    hat = hp(m.noise(0.06), 7000) * np.exp(-tt(0.06) * 70)
    k = 0; t = t0
    while t < t1 - 0.1:
        m.place(kick, t, gain)
        m.place(hat, t + bt / 2, gain * 0.32, pan=0.2 * (-1) ** k)
        m.place(hat, t + bt * 0.75, gain * 0.14, pan=-0.3)
        if k % 4 == 2:
            sn = bp(m.noise(0.2), 1500, 6000) * np.exp(-tt(0.2) * 22)
            m.place(sn, t, gain * 0.35)
        k += 1; t += bt

def wind(m, t0, t1, gain=0.25):
    d = t1 - t0; x = np.linspace(0, 1, int(d * SR))
    s = sweep_noise(m, d, 200, 1200, 30) * (0.3 + 0.7 * np.sin(np.pi * x))
    m.place(np.stack([s, np.roll(s, 2000)], 1), t0, gain)

def chime(m, t, gain=0.9):
    x = tt(2.4)
    c = sum(np.sin(2 * np.pi * fr * x) * g for fr, g in [(880, 0.1), (1318.5, 0.07), (1760, 0.04), (659.25, 0.08)]) * np.exp(-x * 1.5)
    m.place(np.stack([c, np.roll(c, 600)], 1), t, gain)

def finish(m, path, fade=0.5):
    o = np.tanh(m.out * 1.1); o /= np.abs(o).max() / 0.9
    fo = int(fade * SR); o[-fo:] *= np.linspace(1, 0, fo)[:, None]
    o.astype(np.float32).tofile(path)
