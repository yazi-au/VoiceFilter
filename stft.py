"""STFT / iSTFT。训练与流式共用同一套数学，保证推理时和训练时完全一致。

约定：帧 k 覆盖 padded[k*hop : k*hop+n_fft]，左侧先补 n_fft-hop 个零。
这样第一帧窗口左端落在 Hann 的零点上，OLA 归一化天然成立，
且流式算法延迟恰为 n_fft - hop（16 kHz 下 384 样点 = 24 ms）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def hann(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    return torch.hann_window(n, periodic=True, device=device, dtype=dtype)


def _pad(x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    """左补 n_fft-hop、右补 n_fft-hop，再对齐到帧格。

    两侧都补，是为了让首尾样点也落在完整重叠区里——否则最后 24 ms 只被一帧覆盖，
    OLA 归一化的分母（窗平方和）趋近于 0，会引入可观的数值误差。
    """
    x = F.pad(x, (n_fft - hop, n_fft - hop))
    t = x.shape[-1]
    if t < n_fft:
        x = F.pad(x, (0, n_fft - t))
        t = n_fft
    rem = (t - n_fft) % hop
    if rem:
        x = F.pad(x, (0, hop - rem))
    return x


def n_frames(length: int, n_fft: int, hop: int) -> int:
    """给定原始样点数，返回帧数（与 stft 内部补零规则一致）。"""
    t = length + 2 * (n_fft - hop)
    rem = (t - n_fft) % hop
    if rem:
        t += hop - rem
    return 1 + (t - n_fft) // hop


def stft(x: torch.Tensor, n_fft: int, hop: int, win: torch.Tensor | None = None) -> torch.Tensor:
    """x: (B, T) -> complex (B, n_frames, n_fft//2+1)"""
    if win is None:
        win = hann(n_fft, x.device, x.dtype)
    x = _pad(x, n_fft, hop)
    t = x.shape[-1]
    n = 1 + (t - n_fft) // hop
    idx = torch.arange(n_fft, device=x.device)[None, :] + hop * torch.arange(n, device=x.device)[:, None]
    frames = x[:, idx]                                   # (B, n, n_fft)
    return torch.fft.rfft(frames * win, dim=-1)          # (B, n, F)


def istft(spec: torch.Tensor, n_fft: int, hop: int, length: int,
          win: torch.Tensor | None = None) -> torch.Tensor:
    """spec: complex (B, n, F) -> (B, length)。与 stft 严格互逆。"""
    if win is None:
        win = hann(n_fft, spec.device, spec.real.dtype)
    frames = torch.fft.irfft(spec, n=n_fft, dim=-1) * win   # (B, n, n_fft)
    B, n, _ = frames.shape
    out_len = (n - 1) * hop + n_fft
    y = F.fold(frames.transpose(1, 2), output_size=(1, out_len),
               kernel_size=(1, n_fft), stride=(1, hop))
    w = win.pow(2).view(1, n_fft, 1).expand(B, n_fft, n)
    wsum = F.fold(w, output_size=(1, out_len), kernel_size=(1, n_fft), stride=(1, hop))
    y = y.view(B, out_len) / (wsum.view(B, out_len) + 1e-8)
    off = n_fft - hop
    return y[:, off:off + length]


def frame_energy(x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    """帧级能量，与 stft 的帧划分完全对齐 -> (B, n_frames)"""
    x = _pad(x, n_fft, hop)
    t = x.shape[-1]
    n = 1 + (t - n_fft) // hop
    idx = torch.arange(n_fft, device=x.device)[None, :] + hop * torch.arange(n, device=x.device)[:, None]
    fr = x[:, idx]
    return (fr.pow(2).mean(-1) + 1e-10)


def vad_label(ref: torch.Tensor, n_fft: int, hop: int, thresh_ratio: float = 0.03) -> torch.Tensor:
    """由参考信号生成帧级活动标签，用于 VAD 辅助头。"""
    e = frame_energy(ref, n_fft, hop)                     # (B, n)
    peak = e.max(dim=-1, keepdim=True).values.clamp_min(1e-8)
    v = (e > peak * thresh_ratio).float()
    # 3 帧平滑，避免单帧抖动
    v = F.avg_pool1d(v.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
    return (v > 0.5).float()
