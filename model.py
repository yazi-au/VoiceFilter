"""目标说话人提取网络（约 0.6 M 可训练参数，全因果，可流式）。

结构要点
  1. 复数谱输入，逐帧幅度归一化 -> 幅度无关的复比率掩码（cRM），自带相位修正
  2. 编码器在频率轴降采样、时间轴严格因果（左补零的 5 点卷积）
  3. 瓶颈：频向 BiGRU（整帧已到齐，非因果零成本）+ 音色交叉注意力 + 时向 GRU（跨帧状态）
  4. 双路掩码：每个时频点的 re/im 在两个源上做 softmax，目标/干扰互斥
  5. 目标 VAD 头，作为掩码的软门控，负责「你不出声就输出静音」

流式调用：forward(spec_t1, E, state) 逐帧喂，state 是 GRU 隐状态，跨帧保留。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Cfg


# ---------------------------------------------------------------- 基础模块

class CausalConv2d(nn.Module):
    """时间轴左补零的卷积，频率轴可指定对称 padding。

    流式推理必须保留时间轴上的感受野历史，否则逐帧喂入时每一层的输入
    都缺了过去 4 帧，输出和批量前向对不上。cache 就是该层输入的最后 k-1 帧。
    """

    def __init__(self, cin: int, cout: int, k=(5, 3), stride=(1, 1), freq_pad: int = 0):
        super().__init__()
        self.k, self.freq_pad = k, freq_pad
        self.conv = nn.Conv2d(cin, cout, k, stride, padding=0)

    def forward(self, x, cache=None):            # (B,C,T,F) -> ((B,C',T,F'), new_cache)
        if self.freq_pad:
            x = F.pad(x, (self.freq_pad, self.freq_pad, 0, 0))
        need = self.k[0] - 1
        if need == 0:
            return self.conv(x), None
        xin = F.pad(x, (0, 0, need, 0)) if cache is None else torch.cat([cache, x], dim=2)
        return self.conv(xin), xin[:, :, xin.shape[2] - need:, :]


def fnorm(x: torch.Tensor) -> torch.Tensor:
    """帧内归一化：只统计当前帧的 C×F，绝不跨时间 -> 流式安全。"""
    mu = x.mean(dim=(1, 3), keepdim=True)
    sd = x.var(dim=(1, 3), keepdim=True).add(1e-5).sqrt()
    return (x - mu) / sd


class SpeakerAttn(nn.Module):
    """混合特征做 query，注册语音的帧级音色序列做 key/value。"""

    def __init__(self, d: int, d_spk: int, heads: int = 4):
        super().__init__()
        assert d % heads == 0
        self.heads = heads
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d_spk, d, bias=False)
        self.v = nn.Linear(d_spk, d, bias=False)
        self.o = nn.Linear(d, d)
        self._cache = None

    def set_enrollment(self, E: torch.Tensor):
        """会话开始时算一次，之后每帧只算 query。E: (B,Te,d_spk)"""
        B, Te, _ = E.shape
        h = self.heads
        dh = self.q.out_features // h
        self._cache = (self.k(E).view(B, Te, h, dh).transpose(1, 2),
                       self.v(E).view(B, Te, h, dh).transpose(1, 2))
        return self

    def clear(self):
        self._cache = None

    def forward(self, x: torch.Tensor, E: torch.Tensor | None = None) -> torch.Tensor:
        B, T, d = x.shape
        h, dh = self.heads, d // self.heads
        q = self.q(x).view(B, T, h, dh).transpose(1, 2)
        if self._cache is not None and self._cache[0].shape[0] in (1, B):
            k, v = self._cache
            if k.shape[0] != B:
                k, v = k.expand(B, -1, -1, -1), v.expand(B, -1, -1, -1)
        elif E is not None:
            Te = E.shape[1]
            k = self.k(E).view(B, Te, h, dh).transpose(1, 2)
            v = self.v(E).view(B, Te, h, dh).transpose(1, 2)
        else:
            return torch.zeros_like(x)
        a = F.scaled_dot_product_attention(q, k, v)
        return self.o(a.transpose(1, 2).reshape(B, T, d))


class SpkEncoder(nn.Module):
    """帧级音色编码器。输入对数幅度谱，输出 (B, T/8, d_spk)。

    只在本项目的数据上训练（你 vs 其他人），容量全花在这几个人身上，
    比通用 x-vector 更锐利，而且只有 ~5 万参数。
    """

    def __init__(self, d_spk: int = 128):
        super().__init__()
        self.c1 = CausalConv2d(1, 16, (5, 3), (2, 2))
        self.c2 = CausalConv2d(16, 32, (5, 3), (2, 2))
        self.c3 = CausalConv2d(32, 64, (5, 3), (2, 2))
        self.p = nn.LeakyReLU(0.1)
        self.proj = CausalConv2d(64, d_spk, (1, 1), (1, 1), freq_pad=0)
        self.d_spk = d_spk

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        lm = torch.log1p(spec.abs()).unsqueeze(1)        # (B,1,T,F)
        x = self.p(self.c1(lm)[0])
        x = self.p(self.c2(x)[0])
        x = self.p(self.c3(x)[0])                       # (B,64,T/8,~32)
        x = self.p(self.proj(x)[0])                     # (B,d,T/8,~32)
        return x.mean(dim=3).transpose(1, 2)            # (B,T/8,d)


class AAMSoftmax(nn.Module):
    """角度间隔分类头，只在训练时喂损失用。"""

    def __init__(self, dim: int, n_cls: int, s: float = 30.0, m: float = 0.2):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_cls, dim) * 0.01)
        self.s, self.m = s, m

    def forward(self, emb: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        cos = F.normalize(emb, dim=-1) @ F.normalize(self.w, dim=-1).t()
        if self.training:
            onehot = F.one_hot(label, self.w.shape[0]).to(cos.dtype)
            theta = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
            cos = torch.cos(theta + self.m * onehot) - self.m * onehot
        return cos * self.s


# ---------------------------------------------------------------- 主网络

class TSE(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        ch = cfg.ch
        self.use_speaker = cfg.use_speaker
        self.vad_gate = cfg.vad_gate

        # 频率轴尺寸推演：257 -> 128 -> 64 -> 32 -> 16
        f = cfg.n_freq
        dims = []
        for _ in range(4):
            f = (f - 2) // 2 + 1
            dims.append(f)
        self.f_final = f                       # 16
        self.mid = ch // 2                     # 32

        self.enc = nn.ModuleList([
            CausalConv2d(2, 16, (5, 2), (1, 2)),
            CausalConv2d(16, 24, (3, 2), (1, 2)),
            CausalConv2d(24, 32, (3, 2), (1, 2)),
            CausalConv2d(32, ch, (3, 2), (1, 2)),
        ])
        self.act = nn.LeakyReLU(0.1)

        self.freq_gru = nn.GRU(ch, ch // 2, batch_first=True, bidirectional=True)
        self.fc_freq = CausalConv2d(ch, self.mid, (1, 1), (1, 1))

        self.proj_in = nn.Linear(self.mid * self.f_final, cfg.d_model)
        # 绝对电平单独注入：掩码路径是幅度无关的（cRM 需要），但 VAD 门控必须知道
        # 「现在到底有没有声音」。逐帧归一化会把这个信息抹掉，所以从旁路补进来。
        self.lvl = nn.Linear(1, cfg.d_model)
        self.attn = SpeakerAttn(cfg.d_model, cfg.d_spk) if cfg.use_speaker else None
        self.gru = nn.GRU(cfg.d_model, cfg.gru_hidden, batch_first=True)
        self.proj_out = nn.Linear(cfg.gru_hidden, self.mid * self.f_final)
        self.vad = nn.Linear(cfg.gru_hidden, 1)

        # 解码器是整个模型的算力瓶颈：3×3 卷积在 64/128 个频点上极贵，
        # 所以高频段只留很少的通道，时间核缩到 1（时序建模已经由 GRU 完成，
        # 解码器本质是逐帧上采样）。这一步把解码器 FLOPs 砍掉约 4 倍。
        self.dc0 = CausalConv2d(self.mid, 48, (3, 3), (1, 1), freq_pad=1)
        self.dc1 = CausalConv2d(48 + 32, 32, (1, 3), (1, 1), freq_pad=1)
        self.dc2 = CausalConv2d(32 + 24, 20, (1, 3), (1, 1), freq_pad=1)
        self.dc3 = CausalConv2d(20 + 16, 12, (1, 3), (1, 1), freq_pad=1)
        self.dc4 = CausalConv2d(12, 4, (1, 3), (1, 1), freq_pad=1)

        self.spk = SpkEncoder(cfg.d_spk) if cfg.use_speaker else None
        self.aam = AAMSoftmax(cfg.d_spk, cfg.spk_n_cls) if cfg.use_speaker else None

    # -------------------------------------------------- 音色
    def encode_speaker(self, enroll_spec: torch.Tensor):
        E = self.spk(enroll_spec)                                     # (B,Te,d)
        pooled = F.normalize(E.mean(dim=1), dim=-1)
        return E, pooled

    def set_enrollment(self, E: torch.Tensor):
        self.attn.set_enrollment(E)
        return self

    # -------------------------------------------------- 前向
    def forward(self, spec: torch.Tensor, E: torch.Tensor | None = None,
                state: torch.Tensor | None = None, caches: list | None = None):
        """spec: complex (B,T,F)。

        返回 (y_tgt, y_int, vad_logit, gru_state, conv_caches)。
        流式推理时把 state / caches 传回来，逐帧调用即可与批量前向逐样点一致。
        """
        B, T, Fq = spec.shape
        mag = spec.abs()
        scale = mag.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-4)
        xn = spec / scale
        x = torch.stack([xn.real, xn.imag], dim=1)                    # (B,2,T,F)
        lvl = torch.log1p(scale.squeeze(-1) / 0.01).clamp(0.0, 12.0)  # (B,T) 绝对电平

        cin = caches if caches is not None else [None] * 9
        cout: list = []
        skips = []
        for i, c in enumerate(self.enc):
            x, nc = c(x, cin[i])
            cout.append(nc)
            x = self.act(fnorm(x))
            skips.append(x)                                           # 16@128,24@64,32@32,64@16

        C, T, W = skips[-1].shape[1:]
        z = skips[-1].permute(0, 2, 3, 1).reshape(B * T, W, C)
        z, _ = self.freq_gru(z)
        z = z.reshape(B, T, W, C).permute(0, 3, 1, 2) + skips[-1]
        z = self.act(fnorm(self.fc_freq(z)[0]))                       # (B,32,T,16)

        zt = z.permute(0, 2, 1, 3).reshape(B, T, -1)                  # (B,T,512)
        zt = self.proj_in(zt) + self.lvl(lvl.unsqueeze(-1))           # (B,T,d_model)
        if self.attn is not None and (E is not None or self.attn._cache is not None):
            zt = zt + self.attn(zt, E)

        g, state = self.gru(zt, state)                                # (B,T,H)
        y = self.proj_out(g).reshape(B, T, self.mid, W).permute(0, 2, 1, 3)

        d, nc = self.dc0(y, cin[4]); cout.append(nc)
        d = self.act(fnorm(d))
        d = F.interpolate(d, size=(T, 32), mode="nearest")
        d, nc = self.dc1(torch.cat([d, skips[2]], 1), cin[5]); cout.append(nc)
        d = self.act(fnorm(d))
        d = F.interpolate(d, size=(T, 64), mode="nearest")
        d, nc = self.dc2(torch.cat([d, skips[1]], 1), cin[6]); cout.append(nc)
        d = self.act(fnorm(d))
        d = F.interpolate(d, size=(T, 128), mode="nearest")
        d, nc = self.dc3(torch.cat([d, skips[0]], 1), cin[7]); cout.append(nc)
        d = self.act(fnorm(d))
        d = F.interpolate(d, size=(T, Fq), mode="nearest")
        d, nc = self.dc4(d, cin[8]); cout.append(nc)                  # (B,4,T,F)

        m = d.reshape(B, 2, 2, T, Fq).softmax(dim=1)
        mt, mi = m[:, 0], m[:, 1]                                     # (B,2,T,F)

        vad_logit = self.vad(g).squeeze(-1)                           # (B,T)
        if self.vad_gate:
            gate = torch.sigmoid(vad_logit).unsqueeze(1).unsqueeze(-1)    # (B,1,T,1)
            mt = mt * gate

        y_tgt = torch.complex(mt[:, 0], mt[:, 1]) * spec
        y_int = torch.complex(mi[:, 0], mi[:, 1]) * spec
        return y_tgt, y_int, vad_logit, state, cout

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
