"""流式推理封装：与训练完全同一套 STFT/掩码/OLA 数学。

延迟 = n_fft - hop = 384 样点 = 24 ms @16 kHz（不含设备缓冲）。
用法：
    s = StreamingTSE(model, cfg)
    s.set_enrollment(wav_16k)                 # 会话开始一次，2 s 即可
    for block in mic:                         # 任意块长，通常 = hop
        out = s.process(block)                # 输出与输入等长，滞后 24 ms
"""
from __future__ import annotations

import numpy as np
import torch

import stft as S
from config import Cfg
from model import TSE


class StreamingTSE:
    def __init__(self, model: TSE, cfg: Cfg, device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.n_fft, self.hop = cfg.n_fft, cfg.hop
        self.delay = cfg.n_fft - cfg.hop
        self.win = S.hann(self.n_fft, self.device)
        self.reset()

    def reset(self):
        z = lambda n: torch.zeros(n, device=self.device)
        self.in_buf = z(self.n_fft)
        self.ola = z(self.n_fft)
        self.wsum = z(self.n_fft)
        self.state = None
        self.caches = None
        self.pending = z(0)
        self.fifo = z(0)                   # 输出缓冲；延迟由左补零的帧本身提供
        self.last_vad = 0.0

    @torch.no_grad()
    def set_enrollment(self, wav: np.ndarray):
        if getattr(self.model, "spk", None) is None:
            return
        x = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))[None].to(self.device)
        spec = S.stft(x, self.n_fft, self.hop)
        E, _ = self.model.encode_speaker(spec)
        self.model.attn.set_enrollment(E)

    @torch.no_grad()
    def _step(self, chunk: torch.Tensor) -> torch.Tensor:
        self.in_buf = torch.cat([self.in_buf[self.hop:], chunk])
        spec = torch.fft.rfft(self.in_buf * self.win)[None, None, :]     # (1,1,F)
        y, _, vad, self.state, self.caches = self.model(
            spec, None, self.state, self.caches)
        self.last_vad = float(torch.sigmoid(vad).mean())
        frame = torch.fft.irfft(y[0, 0], n=self.n_fft) * self.win
        self.ola = self.ola + frame
        self.wsum = self.wsum + self.win.pow(2)
        seg = self.ola[:self.hop] / self.wsum[:self.hop].clamp_min(1e-6)
        z = torch.zeros(self.hop, device=self.device)
        self.ola = torch.cat([self.ola[self.hop:], z])
        self.wsum = torch.cat([self.wsum[self.hop:], z])
        return seg

    @torch.no_grad()
    def process(self, block: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(block, dtype=np.float32)).to(self.device)
        self.pending = torch.cat([self.pending, x])
        out = []
        while self.pending.numel() >= self.hop:
            out.append(self._step(self.pending[:self.hop]))
            self.pending = self.pending[self.hop:]
        if out:
            self.fifo = torch.cat([self.fifo] + out)
        n = x.numel()
        if self.fifo.numel() < n:
            self.fifo = torch.cat([self.fifo, torch.zeros(n - self.fifo.numel(), device=self.device)])
        y, self.fifo = self.fifo[:n], self.fifo[n:]
        return y.cpu().numpy()

    @torch.no_grad()
    def process_array(self, wav: np.ndarray) -> np.ndarray:
        """离线整段处理，内部按 hop 分块，结果与流式一致（同样滞后 delay）。"""
        self.reset()
        x = np.ascontiguousarray(wav, dtype=np.float32)
        n = int(np.ceil(len(x) / self.hop)) * self.hop
        x = np.pad(x, (0, n - len(x)))
        out = []
        for i in range(0, n, self.hop):
            out.append(self.process(x[i:i + self.hop]))
        return np.concatenate(out)[:len(x)]
