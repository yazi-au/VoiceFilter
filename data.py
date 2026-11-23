"""混合数据集：把「你」叠到「宿舍」上，造出四类训练对。

P1 重叠   宿舍有人说话 + 你说话   -> 真正的分辨能力（≥70% 的样本要有重叠）
P2 空目标 只有宿舍声音，目标全零 -> 该安静就安静，压住泄漏
P3 纯目标 宿舍静音 + 你说话      -> 防过抑制，保住音色
P4 边界   你起或止，室友持续说   -> 门控起落，不产生咔哒

你的素材已经去掉近似静音，所以目标轨用「随机短片段 + 随机停顿」重新拼出来，
把停顿结构补回去。
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

import audio as A
from config import Cfg

K_OVERLAP, K_EMPTY, K_CLEAN, K_BOUND = 0, 1, 2, 3


def split_paths(paths: list[str], seed: int, split: str) -> list[str]:
    """按文件划分训练/验证，绝不允许同一段录音同时出现在两边。

    单文件最长 5 分钟，n<3 时无法留出验证集，此时验证集复用第一个文件，
    只用来观察趋势，不能当成泛化指标。
    """
    perm = np.random.default_rng(seed).permutation(len(paths))
    n = len(paths)
    n_val = max(1, int(round(n * 0.1))) if n >= 3 else 0
    idx = perm[:n - n_val] if split == "train" else perm[max(n - n_val, 0):]
    if split != "train" and len(idx) == 0:
        idx = perm[:1]
    return [paths[k] for k in idx]


class MixDataset(Dataset):
    def __init__(self, cfg: Cfg, split: str = "train", length: int | None = None, seed: int | None = None):
        self.cfg = cfg
        self.split = split
        self.seed = cfg.seed if seed is None else seed

        tpaths = A.scan(cfg.target_dir)
        ipaths = A.scan(cfg.interf_dir)
        self.tpaths = split_paths(tpaths, self.seed, split)
        self.ipaths = split_paths(ipaths, self.seed + 7, split)
        self.tbank = A.Bank(self.tpaths, cfg.sr, cfg.cache_gb, cfg.preload,
                            cfg.preload_max_min, tag=f'{split}/目标')
        self.ibank = A.Bank(self.ipaths, cfg.sr, cfg.cache_gb, cfg.preload,
                            cfg.preload_max_min, tag=f'{split}/宿舍')

        self.length = length if length is not None else cfg.steps * cfg.batch
        self.sr = cfg.sr
        self._eq = None

    # ------------------------------------------------------------ 声学匹配
    def eq_gain(self) -> np.ndarray | None:
        """把目标语音的谱包络整形到宿舍录音的谱包络。构造一次，全程复用。"""
        if self._eq is not None:
            return self._eq if len(self._eq) else None
        try:
            ne = 1024
            rng = np.random.default_rng(0)
            st = self._mean_ltas(self.tbank, rng, ne)
            si = self._mean_ltas(self.ibank, rng, ne)
            self._eq = A.eq_curve(st, si, ne, self.cfg.eq_max_db)
        except Exception:
            self._eq = np.zeros(1, dtype=np.float32)
        return self._eq if len(self._eq) > 1 else None

    def _mean_ltas(self, bank: A.Bank, rng, n_fft: int) -> np.ndarray:
        acc, cnt = None, 0
        for i in range(min(8, len(bank.paths))):
            x = bank.get(i)
            if len(x) < n_fft * 2:
                continue
            v = A.ltas(x, n_fft, self.sr)
            acc = v if acc is None else acc + v
            cnt += 1
        if acc is None:
            return np.zeros(n_fft // 2 + 1, dtype=np.float32)
        return acc / max(cnt, 1)

    # ------------------------------------------------------------ 拼目标
    def _compose(self, rng: np.random.Generator, T: int, lead: float = 0.0, trail: float = 0.0) -> np.ndarray:
        sr = self.sr
        out = np.zeros(T, dtype=np.float32)
        cur = int(lead * sr)
        end = T - int(trail * sr)
        guard = int(0.10 * sr)
        while cur < end - guard:
            L = int(rng.uniform(0.35, 1.40) * sr)
            L = min(L, end - cur)
            i = int(rng.integers(0, len(self.tpaths)))
            out[cur:cur + L] = self.tbank.crop(i, L, rng)
            cur += L
            if rng.random() < self.cfg.p_gap:
                cur += int(rng.uniform(0.06, 0.50) * sr)
        return out

    def _interference(self, rng: np.random.Generator, T: int, need_active: bool = True) -> np.ndarray:
        k = int(rng.integers(1, self.cfg.max_interf + 1))
        out = np.zeros(T, dtype=np.float32)
        for _ in range(k):
            seg = None
            for _try in range(4):
                i = int(rng.integers(0, len(self.ipaths)))
                seg = self.ibank.crop(i, T, rng)
                if not need_active or A.active_ratio(seg, self.sr) >= 0.35:
                    break
            out += seg * float(rng.uniform(0.6, 1.0))
        return out

    def _enroll(self, rng: np.random.Generator, paths, bank: A.Bank,
                avoid: set[int] | None = None) -> np.ndarray:
        choices = [i for i in range(len(paths)) if not avoid or i not in avoid]
        if not choices:
            choices = list(range(len(paths)))
        i = int(choices[int(rng.integers(0, len(choices)))])
        n = int(2.0 * self.sr)          # 固定 2 s，省掉 batch 内补零
        e = bank.crop(i, n, rng)
        g = self.eq_gain()
        if g is not None and rng.random() < self.cfg.p_eq:
            e = A.apply_eq(e, g, self.sr)
        return A.peak_norm(e, 0.7)

    # ------------------------------------------------------------ 取样本
    def __getitem__(self, idx: int) -> dict:
        cfg = self.cfg
        rng = np.random.default_rng((self.seed + 1) * 1_000_003 + idx)
        T = cfg.seg_len

        u = rng.random()
        if u < cfg.p_overlap:
            kind = K_OVERLAP
        elif u < cfg.p_overlap + cfg.p_empty:
            kind = K_EMPTY
        elif u < cfg.p_overlap + cfg.p_empty + cfg.p_clean:
            kind = K_CLEAN
        else:
            kind = K_BOUND

        # ---- 干扰
        if kind == K_CLEAN:
            intf = np.zeros(T, dtype=np.float32)
        else:
            intf = self._interference(rng, T, need_active=(kind in (K_OVERLAP, K_EMPTY, K_BOUND)))
            intf = A.peak_norm(intf, 0.7)

        # ---- 目标
        if kind == K_EMPTY:
            tgt = np.zeros(T, dtype=np.float32)
        else:
            lead = trail = 0.0
            if kind == K_BOUND:
                if rng.random() < 0.5:
                    lead = float(rng.uniform(0.25, 0.90))
                else:
                    trail = float(rng.uniform(0.25, 0.90))
            tgt = self._compose(rng, T, lead=lead, trail=trail)
            # 声学匹配：先把目标整形成「在宿舍里说话」的样子，再混
            g = self.eq_gain()
            if g is not None and rng.random() < cfg.p_eq:
                tgt = A.apply_eq(tgt, g, self.sr)
            if rng.random() < cfg.p_rir:
                t60 = float(rng.uniform(*cfg.rir_t60))
                tgt = A.convolve(tgt, A.synth_rir(self.sr, t60, rng))
            tgt = A.peak_norm(tgt, 0.7)

        # ---- 电平对齐到目标 SIR
        if kind != K_EMPTY and kind != K_CLEAN and A.rms(intf) > 1e-6:
            rt = A.active_rms(tgt, self.sr)
            ri = A.active_rms(intf, self.sr)
            if rt > 1e-6 and ri > 1e-6:
                sir = float(rng.uniform(cfg.sir_lo, cfg.sir_hi))
                tgt = tgt * (ri * (10.0 ** (sir / 20.0)) / rt)
        elif kind == K_CLEAN:
            tgt = tgt * float(rng.uniform(0.3, 1.0))

        if kind == K_EMPTY:
            ref = np.zeros(T, dtype=np.float32)
        else:
            ref = tgt.astype(np.float32).copy()

        mix = (intf + tgt).astype(np.float32)

        # 模拟 AGC / 削波 / 整体增益。三个信号必须同乘同除，
        # 否则干扰支路的损失会和目标支路的尺度过不上。
        g = float(rng.uniform(0.5, 1.5))
        mix, ref, intf = mix * g, ref * g, intf.astype(np.float32) * g
        p = max(float(np.max(np.abs(mix))), float(np.max(np.abs(ref))), 1e-6)
        if p > 0.98:
            mix, ref, intf = mix / (p * 1.02), ref / (p * 1.02), intf / (p * 1.02)

        # ---- 音色分支
        used = set()
        enroll = None
        enroll_neg = None
        if cfg.use_speaker:
            enroll = self._enroll(rng, self.tpaths, self.tbank, used)
            enroll_neg = self._enroll(rng, self.ipaths, self.ibank, None)

        return {
            "mix": torch.from_numpy(mix),
            "ref": torch.from_numpy(ref),
            "intf": torch.from_numpy(intf.astype(np.float32)),
            "enroll": torch.zeros(1) if enroll is None else torch.from_numpy(enroll),
            "enroll_neg": torch.zeros(1) if enroll_neg is None else torch.from_numpy(enroll_neg),
            "kind": torch.tensor(kind, dtype=torch.long),
            "has_tgt": torch.tensor(kind != K_EMPTY),
            "has_int": torch.tensor(kind != K_CLEAN),
        }

    def __len__(self) -> int:
        return self.length


def collate(batch: list[dict]) -> dict:
    out = {}
    for k in batch[0]:
        if k in ("enroll", "enroll_neg"):
            m = max(b[k].numel() for b in batch)
            out[k] = torch.stack([torch.nn.functional.pad(b[k], (0, m - b[k].numel())) for b in batch])
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out
