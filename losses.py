"""损失函数。

注意 P2（空目标）样本不能用 SI-SNR —— 参考信号全零时它无定义，
改用相对干扰能量的惩罚项，把输出推向静音。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_MR = ((512, 128), (256, 64), (128, 32))


def si_snr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """(B,T) -> (B,)，单位 dB。scale invariant。"""
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    s = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps)
    proj = s * ref
    noise = est - proj
    return 10.0 * torch.log10((proj.pow(2).sum(-1) + eps) / (noise.pow(2).sum(-1) + eps))


def mr_stft(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """多分辨率谱损失：谱收敛 + 对数幅度 L1。"""
    loss = est.new_zeros(())
    for n_fft, hop in _MR:
        win = torch.hann_window(n_fft, device=est.device, dtype=est.dtype)
        a = torch.stft(ref, n_fft, hop, window=win, center=True, return_complex=True).abs()
        b = torch.stft(est, n_fft, hop, window=win, center=True, return_complex=True).abs()
        sc = (a - b).norm(dim=(1, 2)) / (a.norm(dim=(1, 2)) + 1e-8)
        lm = F.l1_loss(torch.log(b + 1e-7), torch.log(a + 1e-7), reduction="none").mean(dim=(1, 2))
        loss = loss + (sc + lm).mean()
    return loss / len(_MR)


def empty_penalty(est: torch.Tensor, intf: torch.Tensor, floor_db: float = -60.0) -> torch.Tensor:
    """空目标样本：输出相对干扰能量的分贝值，压到 floor_db 以下即无惩罚。

    返回 0..6 的量纲，与 SI-SNR（约 -20..0）可比。
    """
    r_e = est.pow(2).mean(-1).add(1e-10).sqrt()
    r_i = intf.pow(2).mean(-1).add(1e-10).sqrt()
    lvl = 20.0 * torch.log10(r_e / (r_i + 1e-6) + 1e-6)
    return ((lvl.clamp(min=floor_db, max=0.0) - floor_db) / 10.0).mean()


def vad_bce(logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logit, label)
