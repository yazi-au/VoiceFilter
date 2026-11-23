"""音频 IO 与 DSP 工具：扫描、重采样、缓存、LTAS 匹配、合成 RIR。

设计原则：所有处理都必须**时间对齐**——目标语音在混入混合信号之前做完 EQ 和
卷积，混合信号的目标分量就是参考信号本身，不存在样点偏移。
"""
from __future__ import annotations

import os
import numpy as np

AUDIO_EXT = (".wav", ".flac", ".ogg", ".aiff", ".aif")


def scan(root: str) -> list[str]:
    """递归扫描音频文件，按路径排序保证可复现。"""
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(AUDIO_EXT):
                out.append(os.path.join(dp, fn))
    out.sort()
    if not out:
        raise FileNotFoundError(
            f"{root} 下没有找到音频。支持 {AUDIO_EXT}；mp3/m4a 请先转 wav：\n"
            f"  ffmpeg -i in.m4a -ac 1 -ar 16000 out.wav"
        )
    return out


def load_mono(path: str, sr: int) -> np.ndarray:
    """读成单声道 float32 @ sr。优先 soundfile，失败退回 ffmpeg。"""
    try:
        import soundfile as sf
        x, fs = sf.read(path, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
    except Exception:
        import subprocess
        import io
        raw = subprocess.run(
            ["ffmpeg", "-v", "quiet", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
            capture_output=True, check=True).stdout
        x = np.frombuffer(raw, dtype=np.float32).copy()
        fs = sr
    if fs != sr:
        x = resample(x, fs, sr)
    return np.ascontiguousarray(x, dtype=np.float32)


def resample(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    if fs_in == fs_out:
        return x
    from math import gcd
    g = gcd(int(fs_in), int(fs_out))
    from scipy.signal import resample_poly
    return resample_poly(x, fs_out // g, fs_in // g).astype(np.float32)


class Bank:
    """音频缓存。

    preload=True 时构造完就把所有文件解码进内存（数据量小的时候最省事：
    训练全程零磁盘 IO，也不会在几个 worker 里反复解码）。总时长超过缓存上限
    时自动退回按需加载。
    """

    def __init__(self, paths: list[str], sr: int, cache_gb: float = 6.0,
                 preload: bool = True, preload_max_min: float = 30.0, tag: str = ""):
        self.sr = sr
        self.cache: dict[int, np.ndarray] = {}
        self.order: list[int] = []
        self.cap = int(cache_gb * (1024 ** 3) / 4)  # float32 样点数
        self.n = 0

        # 只保留能读出时长的文件，读不出来的直接剔除，避免后面反复解码
        durs = []
        keep = []
        for p in paths:
            d = _probe_dur(p, sr)
            if d > 0.1:
                keep.append(p)
                durs.append(d)
        self.paths = keep
        self.dur = np.array(durs, dtype=np.float32)
        if len(keep) < len(paths):
            print(f"[{tag or 'bank'}] 跳过 {len(paths)-len(keep)} 个无法识别的文件")

        total_min = float(self.dur.sum()) / 60.0
        need = int(self.dur.sum() * sr)
        # 双重上限：既不能超过缓存，也不能超过时长上限。宁可慢一点，也不能吃爆内存。
        self.preloaded = bool(preload and 0 < need <= self.cap * 0.9
                              and total_min <= preload_max_min)
        if self.preloaded:
            for i in range(len(self.paths)):
                self.get(i)
            print(f"[{tag or 'bank'}] 预加载 {len(self.paths)} 个文件 / "
                  f"{total_min:.1f} 分钟 / {self.n*4/1024**2:.1f} MB 进内存")
        elif preload and total_min > 0:
            print(f"[{tag or 'bank'}] {total_min:.1f} 分钟超过预加载上限 "
                  f"{preload_max_min:.0f} 分钟，改为按需加载（不会吃爆内存）")

    def get(self, i: int) -> np.ndarray:
        if i in self.cache:
            return self.cache[i]
        x = load_mono(self.paths[i], self.sr)
        while self.n + len(x) > self.cap and self.order:
            j = self.order.pop(0)
            self.n -= len(self.cache.pop(j))
        self.cache[i] = x
        self.order.append(i)
        self.n += len(x)
        return x

    def crop(self, i: int, n: int, rng: np.random.Generator) -> np.ndarray:
        """取长度 n 的片段；文件太短就循环平铺。"""
        x = self.get(i)
        if len(x) < n:
            k = int(np.ceil(n / max(len(x), 1)))
            x = np.tile(x, k)
        s = int(rng.integers(0, len(x) - n + 1))
        return x[s:s + n].copy()

    def total_hours(self) -> float:
        return float(self.dur.sum()) / 3600.0


def _probe_dur(path: str, sr: int) -> float:
    try:
        import soundfile as sf
        info = sf.info(path)
        return info.frames / info.samplerate
    except Exception:
        try:
            return len(load_mono(path, sr)) / sr
        except Exception:
            return 0.0


# ---------------------------------------------------------------- 电平

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))


def db(x) -> np.ndarray:
    """支持标量和数组。"""
    return 20.0 * np.log10(np.maximum(np.asarray(x, dtype=np.float64), 1e-9))


def frame_rms(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return np.sqrt(np.mean(np.square(x[idx], dtype=np.float64), axis=1) + 1e-12)


def active_rms(x: np.ndarray, sr: int = 16000, thresh_db: float = -42.0) -> float:
    """只在语音活跃帧上统计 RMS，避免长静音把电平拉低。"""
    fr = frame_rms(x, int(0.02 * sr), int(0.01 * sr))
    if fr.max() <= 0:
        return 0.0
    act = fr[db(fr) > db(fr.max()) - abs(thresh_db)]
    return float(np.sqrt(np.mean(np.square(act)) + 1e-12)) if len(act) else 0.0


def active_ratio(x: np.ndarray, sr: int = 16000) -> float:
    fr = frame_rms(x, int(0.02 * sr), int(0.01 * sr))
    if fr.max() <= 0:
        return 0.0
    return float(np.mean(db(fr) > db(fr.max()) - 30.0))


# ---------------------------------------------------------------- 声学匹配

def synth_rir(sr: int, t60: float, rng: np.random.Generator) -> np.ndarray:
    """合成指数衰减 RIR。直达声固定在 0 号样点，因此卷积后无需对齐。"""
    n = max(int(t60 * sr * 1.2), 64)
    t = np.arange(n) / sr
    h = rng.standard_normal(n) * np.exp(-6.9078 * t / t60)
    # 直达声占比随机，模拟不同距离
    h[0] = 1.0 * rng.uniform(0.8, 1.6)
    h[:max(2, int(0.002 * sr))] += 0.3 * rng.standard_normal(max(2, int(0.002 * sr)))
    return (h / (np.sqrt(np.sum(h ** 2)) + 1e-9)).astype(np.float32)


def convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    n = len(x) + len(h) - 1
    nfft = 1 << (n - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(h, nfft), nfft)[:len(x)]
    return y.astype(np.float32)


def ltas(x: np.ndarray, n_fft: int = 1024, sr: int = 16000, max_sec: float = 20.0) -> np.ndarray:
    """长期平均谱（dB），只统计语音活跃帧。

    只在文件里取一段不超过 max_sec 的片段来统计——包络估计不需要整段，
    但按整段算的话，一个 5 分钟文件的中间张量就要几百 MB，长录音能直接吃爆内存。
    """
    n_max = int(max_sec * sr)
    if len(x) > n_max:
        s = (len(x) - n_max) // 2
        x = x[s:s + n_max]
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    hop = n_fft // 4
    n = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    fr = x[idx] * np.hanning(n_fft)[None, :]
    mag = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    # 只统计能量较高的 60% 帧，避免把静音段的底噪算进包络
    if len(mag) > 8:
        e_fr = mag.mean(axis=1)
        keep = e_fr >= np.percentile(e_fr, 40)
        if keep.sum() >= 4:
            mag = mag[keep]
    p = np.mean(mag, axis=0)
    return 10.0 * np.log10(p + 1e-12)


def smooth_logfreq(curve_db: np.ndarray, n_fft: int, strength: float = 0.33) -> np.ndarray:
    """1/3 倍频程式平滑：窗宽随频率线性增长。"""
    k = len(curve_db)
    out = np.empty_like(curve_db)
    for i in range(k):
        w = max(1, int(round(i * strength)))
        lo, hi = max(0, i - w), min(k, i + w + 1)
        out[i] = curve_db[lo:hi].mean()
    return out


def eq_curve(src_db: np.ndarray, dst_db: np.ndarray, n_fft: int,
             max_db: float = 12.0, strength: float = 0.33) -> np.ndarray:
    """求把 src 的谱包络整形到 dst 的增益曲线（线性幅度）。"""
    g = smooth_logfreq(dst_db - src_db, n_fft, strength)
    g = np.clip(g, -max_db, max_db)
    g = smooth_logfreq(g, n_fft, strength * 2.0)  # 二次平滑，避免过冲
    g -= g.mean()
    return (10.0 ** (g / 20.0)).astype(np.float32)


def apply_eq(x: np.ndarray, gain_curve: np.ndarray, sr: int = 16000) -> np.ndarray:
    """零相位 EQ：整段 FFT 后乘实增益曲线，无群延迟。"""
    n = len(x)
    nfft = 1 << (n - 1).bit_length()
    X = np.fft.rfft(x, nfft)
    gf = np.linspace(0, sr / 2, len(X))
    g_in = np.linspace(0, sr / 2, len(gain_curve))
    X *= np.interp(gf, g_in, gain_curve)
    return np.fft.irfft(X, nfft)[:n].astype(np.float32)


def peak_norm(x: np.ndarray, target: float = 0.9) -> np.ndarray:
    p = float(np.max(np.abs(x))) if len(x) else 0.0
    return x if p <= target or p <= 0 else (x * (target / p)).astype(np.float32)
