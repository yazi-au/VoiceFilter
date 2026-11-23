"""训练入口。

    python train.py --target_dir data/me --interf_dir data/dorm      # 正式训练
    python train.py --dry                                            # 不需要数据，自检
    python train.py --selftest                                       # 只跑 STFT/流式一致性自检
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import audio as A
import stft as S
from config import Cfg, asdict
from data import K_CLEAN, K_EMPTY, K_OVERLAP, MixDataset, collate, split_paths
from losses import empty_penalty, mr_stft, si_snr, vad_bce
from model import TSE


# ---------------------------------------------------------------- 参数

def build_cfg(argv=None) -> tuple[Cfg, argparse.Namespace]:
    ap = argparse.ArgumentParser()
    for k in ("target_dir", "interf_dir", "ckpt_dir"):
        ap.add_argument(f"--{k}", type=str)
    for k in ("steps", "batch", "workers", "val_batches", "val_every", "seed"):
        ap.add_argument(f"--{k}", type=int)
    for k in ("lr", "seg_sec"):
        ap.add_argument(f"--{k}", type=float)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--set", nargs="*", default=None, help="k=v 覆盖任意配置项")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--stats", action="store_true", help="只统计数据集时长和文件数")
    a = ap.parse_args(argv)

    cfg = Cfg()
    for k, v in vars(a).items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)
    if a.set:
        cfg.update(dict(s.split("=", 1) for s in a.set))
    return cfg, a


def get_device(cfg) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- 自检

def selftest(cfg: Cfg):
    torch.manual_seed(0)
    np.random.seed(0)
    print("=" * 62)
    x = torch.randn(1, cfg.seg_len) * 0.1
    spec = S.stft(x, cfg.n_fft, cfg.hop)
    y = S.istft(spec, cfg.n_fft, cfg.hop, cfg.seg_len)
    print(f"[1] STFT/iSTFT 往返：帧数={spec.shape[1]} 频点={spec.shape[2]} "
          f"误差={ (x-y).abs().max().item():.2e}")

    model = TSE(cfg).eval()
    print(f"[2] 可训练参数：{model.num_params()/1e6:.3f} M  "
          f"(f_final={model.f_final}, delay={cfg.delay} 样点 = {cfg.delay/cfg.sr*1000:.1f} ms)")

    if cfg.use_speaker:
        wav_e = torch.randn(1, 2 * cfg.sr) * 0.1
        E, _ = model.encode_speaker(S.stft(wav_e, cfg.n_fft, cfg.hop))
        model.attn.set_enrollment(E)
        print(f"[3] 音色帧数：{E.shape[1]}  维度：{E.shape[2]}")

    with torch.no_grad():
        yt, yi, vad, _, _ = model(spec, None)
        full = S.istft(yt, cfg.n_fft, cfg.hop, cfg.seg_len)
    print(f"[4] 批量前向：输出谱 {tuple(yt.shape)}  vad {tuple(vad.shape)}  "
          f"|y|={full.abs().mean().item():.4f}")

    from stream import StreamingTSE
    st = StreamingTSE(model, cfg)
    if model.attn is not None:
        st.model.attn._cache = model.attn._cache
    s_out = st.process_array(x[0].numpy())
    d = cfg.delay
    n = min(len(s_out) - d, cfg.seg_len - d)
    err = np.abs(s_out[d:d + n] - full[0, :n].numpy()).max()
    print(f"[5] 流式 vs 批量：最大偏差 {err:.2e}  (应 < 1e-4)")
    assert err < 1e-3, "流式与批量不一致，检查 STFT padding 或 GRU 状态处理"
    print("=" * 62)
    return err


# ---------------------------------------------------------------- 数据

def make_loader(cfg: Cfg, split: str, length: int):
    ds = MixDataset(cfg, split=split, length=length)
    return ds, DataLoader(ds, batch_size=cfg.batch, shuffle=False, num_workers=cfg.workers,
                          collate_fn=collate, drop_last=True, persistent_workers=cfg.workers > 0)


def stats(cfg: Cfg):
    for name, root, seed in (("目标语音", cfg.target_dir, cfg.seed),
                             ("宿舍录音", cfg.interf_dir, cfg.seed + 7)):
        ps = A.scan(root)
        bank = A.Bank(ps, cfg.sr, preload=False, tag=name)
        d = bank.dur[bank.dur > 0]
        if len(d) == 0:
            print(f"{name}  没有可用文件")
            continue
        ps = bank.paths
        n_tr = len(split_paths(ps, seed, "train"))
        n_va = len(split_paths(ps, seed, "val"))
        print(f"{name}  {len(ps)} 个可用文件 / {d.sum()/60:.1f} 分钟   "
              f"最短 {d.min():.1f}s  中位 {np.median(d):.1f}s  最长 {d.max():.1f}s")
        print(f"   划分：训练 {n_tr} 个 / 验证 {n_va} 个"
              + ("   (文件数<3，验证集复用训练文件，只能看趋势)" if len(ps) < 3 else ""))
        mem = float(d.sum()) * cfg.sr * 4 / 1024 ** 3
        if cfg.preload and d.sum() / 60 <= cfg.preload_max_min and mem <= cfg.cache_gb:
            print(f"   预加载：开，进内存约 {mem*1024:.0f} MB")
        else:
            print(f"   预加载：关（时长或体积超出上限），改为按需加载")
        short = [p for p, v in zip(ps, bank.dur) if v < 0.5]
        if short:
            print(f"   !! {len(short)} 个文件短于 0.5s：{short[:3]}")


# ---------------------------------------------------------------- 前向 + 损失

def run_batch(model: TSE, cfg: Cfg, b: dict, device) -> dict:
    mix = b["mix"].to(device)
    ref = b["ref"].to(device)
    intf = b["intf"].to(device)
    B = mix.shape[0]

    spec = S.stft(mix, cfg.n_fft, cfg.hop)
    E = None
    pooled = None
    if cfg.use_speaker:
        en = torch.cat([b["enroll"], b["enroll_neg"]], 0).to(device)
        e_spec = S.stft(en, cfg.n_fft, cfg.hop)
        E_all, pooled = model.encode_speaker(e_spec)
        E = E_all[:B]

    y_tgt, y_int, vad_logit, _, _ = model(spec, E)
    est = S.istft(y_tgt, cfg.n_fft, cfg.hop, cfg.seg_len)
    est_i = S.istft(y_int, cfg.n_fft, cfg.hop, cfg.seg_len)

    has_t = b["has_tgt"].to(device)
    has_i = b["has_int"].to(device)

    # 累加必须用张量本身，不能用 .item() 的浮点——那会把计算图截断，
    # backward 直接报 "does not require grad"。浮点只用于日志展示。
    loss = est.new_zeros(())
    parts = {"sisnr": 0.0, "stft": 0.0, "int": 0.0, "vad": 0.0, "empty": 0.0, "spk": 0.0}

    def _add(key: str, term: torch.Tensor):
        nonlocal loss
        parts[key] = float(term.detach().item())
        loss = loss + term

    if has_t.any():
        s = si_snr(est[has_t], ref[has_t])
        _add("sisnr", cfg.w_sisnr * (-s).mean())
        _add("stft", cfg.w_stft * mr_stft(est[has_t], ref[has_t]))
    if (~has_t).any():
        _add("empty", cfg.w_empty * empty_penalty(est[~has_t], intf[~has_t]))
    if has_i.any():
        _add("int", cfg.w_int * (-si_snr(est_i[has_i], intf[has_i])).mean())
    if cfg.w_vad > 0:
        _add("vad", cfg.w_vad * vad_bce(vad_logit, S.vad_label(ref, cfg.n_fft, cfg.hop)))

    if cfg.use_speaker and cfg.w_spk > 0:
        lab = torch.cat([torch.zeros(B, dtype=torch.long),
                         torch.ones(B, dtype=torch.long)]).to(device)
        _add("spk", cfg.w_spk * F.cross_entropy(model.aam(pooled, lab), lab))

    return {"loss": loss, "est": est, "ref": ref, "mix": mix, "intf": intf,
            "has_t": has_t, "parts": parts}


@torch.no_grad()
def evaluate(model: TSE, cfg: Cfg, loader, device, n_batch: int) -> dict:
    model.eval()
    acc = {"sisdr": [], "sisdri": [], "leak": [], "over": []}
    for i, b in enumerate(loader):
        if i >= n_batch:
            break
        out = run_batch(model, cfg, b, device)
        est, ref, mix, intf, has_t = out["est"], out["ref"], out["mix"], out["intf"], out["has_t"]
        # SI-SDRi 只在「有目标且有干扰」的样本上统计，否则纯目标样本的
        # mix≈ref 会把基线抬到 +90 dB，指标失去意义
        both = has_t & b["has_int"].to(device)
        if both.any():
            s = si_snr(est[both], ref[both])
            m = si_snr(mix[both], ref[both])
            acc["sisdr"].append(s.mean().item())
            acc["sisdri"].append((s - m).mean().item())
        if (~has_t).any():
            r_e = est[~has_t].pow(2).mean(-1).sqrt()
            r_i = intf[~has_t].pow(2).mean(-1).sqrt()
            lk = 20 * torch.log10(r_e / (r_i + 1e-6) + 1e-6)
            acc["leak"].append(lk.clamp(min=-80).mean().item())
        clean = (~b["has_int"].to(device))
        if clean.any():
            acc["over"].append(si_snr(est[clean], ref[clean]).mean().item())
    model.train()
    return {k: (float(np.mean(v)) if v else float("nan")) for k, v in acc.items()}


# ---------------------------------------------------------------- 主循环

def main():
    cfg, a = build_cfg()
    if a.selftest:
        selftest(cfg)
        return
    if a.stats:
        stats(cfg)
        return

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = get_device(cfg)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        # GPU 一步几十毫秒，数据加载留在主线程就会变成瓶颈，自动开几个 worker
        if cfg.workers == 0:
            cfg.workers = 4
        print(f"设备: cuda  {torch.cuda.get_device_name(0)}  "
              f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GB")
    else:
        print(f"设备: {device}")

    if a.dry:
        selftest(cfg)
        model = TSE(cfg).to(device)
        b = {k: torch.randn(cfg.batch, cfg.seg_len) * 0.1 for k in ("mix", "ref", "intf")}
        b["enroll"] = torch.randn(cfg.batch, 2 * cfg.sr) * 0.1
        b["enroll_neg"] = torch.randn(cfg.batch, 2 * cfg.sr) * 0.1
        b["has_tgt"] = torch.ones(cfg.batch, dtype=torch.bool)
        b["has_int"] = torch.ones(cfg.batch, dtype=torch.bool)
        b["kind"] = torch.zeros(cfg.batch, dtype=torch.long)
        b = {k: v.to(device) for k, v in b.items()}
        out = run_batch(model, cfg, b, device)
        out["loss"].backward()
        print(f"[6] 反向通过，loss={out['loss'].item():.4f}，显存 {torch.cuda.max_memory_allocated()/1e9:.2f} GB"
              if device.type == "cuda" else f"[6] 反向通过，loss={out['loss'].item():.4f}")
        return

    print(f"目标: {cfg.target_dir}   干扰: {cfg.interf_dir}   片段: {cfg.seg_sec}s")
    ds_tr, dl_tr = make_loader(cfg, "train", cfg.steps * cfg.batch)
    ds_va, dl_va = make_loader(cfg, "val", cfg.val_batches * cfg.batch)
    print(f"训练集  目标 {len(ds_tr.tpaths)} 个 / {ds_tr.tbank.total_hours()*60:.1f} 分钟   |   "
          f"干扰 {len(ds_tr.ipaths)} 个 / {ds_tr.ibank.total_hours()*60:.1f} 分钟")

    model = TSE(cfg).to(device)
    print(f"可训练参数 {model.num_params()/1e6:.3f} M")
    n_gru = sum(p.numel() for n, p in model.named_parameters() if n.startswith("gru"))
    n_att = sum(p.numel() for n, p in model.named_parameters() if n.startswith("attn"))
    print(f"  其中 时向GRU {n_gru/1e3:.0f}K   音色注意力 {n_att/1e3:.0f}K")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg.lr, weight_decay=cfg.wd, betas=(0.9, 0.98))

    def lr_at(s):
        if s < cfg.warmup:
            return cfg.lr * (s + 1) / cfg.warmup
        p = (s - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
        return cfg.lr * (0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * min(p, 1.0))))

    start = 0
    if a.resume and os.path.exists(a.resume):
        ck = torch.load(a.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start = ck.get("step", 0)
        print(f"从 {a.resume} 恢复，step={start}")

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    model.train()
    it = iter(dl_tr)
    best = -1e9
    t0 = time.time()
    run = 0.0
    ema: dict[str, float] = {}

    for step in range(start, cfg.steps):
        try:
            b = next(it)
        except StopIteration:
            it = iter(dl_tr)
            b = next(it)

        for g in opt.param_groups:
            g["lr"] = lr_at(step)

        out = run_batch(model, cfg, b, device)
        loss = out["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        run = 0.9 * run + 0.1 * loss.item()
        for k, v in out.get("parts", {}).items():
            ema[k] = 0.9 * ema.get(k, 0.0) + 0.1 * v

        if (step + 1) % 100 == 0:
            dt = time.time() - t0
            # loss 由 -SI-SNR 主导，SI-SNR 转正后 loss 就是负的，负得越多越好。
            # 拆开打印是为了判断是哪一项在动，光看总和看不出问题。
            print(f"step {step+1:6d}  loss {run:7.2f}  "
                  f"[sisnr {ema.get('sisnr',0):6.2f} | stft {ema.get('stft',0):5.2f} "
                  f"| int {ema.get('int',0):6.2f} | vad {ema.get('vad',0):5.3f} "
                  f"| empty {ema.get('empty',0):4.2f}]  "
                  f"lr {lr_at(step):.1e}  {100*cfg.batch/dt:5.1f} seg/s  "
                  f"ETA {(cfg.steps-step-1)*dt/100/60:5.1f} min")
            t0 = time.time()

        if (step + 1) % cfg.val_every == 0 or step + 1 == cfg.steps:
            m = evaluate(model, cfg, dl_va, device, cfg.val_batches)
            print(f"  [val] SI-SDR {m['sisdr']:.2f} dB  SI-SDRi {m['sisdri']:+.2f} dB  "
                  f"泄漏 {m['leak']:.1f} dB  纯目标 SI-SDR {m['over']:.2f} dB")
            score = m["sisdri"] if not np.isnan(m["sisdri"]) else -1e9
            if score > best:
                best = score
                ck = os.path.join(cfg.ckpt_dir, "best.pt")
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "step": step + 1, "cfg": asdict(cfg), "metrics": m}, ck)
                print(f"  -> 保存 {ck}")
            torch.save({"model": model.state_dict(), "step": step + 1, "cfg": asdict(cfg)},
                       os.path.join(cfg.ckpt_dir, "last.pt"))


if __name__ == "__main__":
    main()
