"""全局配置。所有超参集中于此，train.py 可用命令行覆盖（--key value）。"""
from dataclasses import dataclass, fields, asdict


@dataclass
class Cfg:
    # ---------------- 音频 ----------------
    sr: int = 16000
    n_fft: int = 512          # 32 ms
    hop: int = 128            # 8 ms
    seg_sec: float = 3.0      # 训练片段长度

    # ---------------- 数据 ----------------
    target_dir: str = "data/me"
    interf_dir: str = "data/dorm"

    # 四类样本比例
    p_overlap: float = 0.55   # P1 室友说话 + 你说话（重叠）
    p_empty: float = 0.15     # P2 只有室友，目标全零
    p_clean: float = 0.15     # P3 只有你，干扰全零
    p_bound: float = 0.15     # P4 边界：你先静后说 / 先说完后静

    sir_lo: float = -5.0      # 目标相对干扰的信噪比（dB）
    sir_hi: float = 10.0
    max_interf: int = 2       # 一段混合最多叠几段室友录音
    cache_gb: float = 6.0     # 解码缓存上限。5 分钟的单声道文件约 19 MB/个
    preload: bool = True      # 启动时全量解码进内存，训练全程零磁盘 IO
    preload_max_min: float = 30.0  # 超过这个总时长就不预加载，防止吃爆内存

    p_eq: float = 0.9         # 施加 LTAS 匹配的概率
    p_rir: float = 0.7        # 施加房间冲激响应的概率
    p_gap: float = 0.75       # 拼目标时插入静音停顿的概率
    rir_t60: tuple = (0.20, 0.55)
    eq_max_db: float = 12.0

    # ---------------- 模型 ----------------
    ch: int = 64              # 编码器末层通道
    d_model: int = 128        # 时序主干宽度
    gru_hidden: int = 192     # 时向 GRU 隐层
    d_spk: int = 128          # 音色特征维度（仅 use_speaker 时生效）
    use_speaker: bool = False # 默认关：训成「专用模型」，权重直接记住你的声线
    vad_gate: bool = True     # 目标 VAD 软门控
    spk_n_cls: int = 2        # 音色分类头类别数（你 / 其他人）

    # ---------------- 训练 ----------------
    steps: int = 60000
    batch: int = 12
    lr: float = 3e-4
    wd: float = 1e-4
    grad_clip: float = 5.0
    warmup: int = 1000
    val_every: int = 2000
    val_batches: int = 6
    ckpt_dir: str = "ckpt"
    workers: int = 0
    seed: int = 1234
    amp: bool = False

    # ---------------- 损失权重 ----------------
    w_sisnr: float = 1.0
    w_stft: float = 0.5
    w_empty: float = 0.5      # 空目标样本的能量惩罚
    w_vad: float = 0.1
    w_int: float = 0.3        # 干扰支路重建
    w_spk: float = 0.05       # 音色分类辅助

    @property
    def n_freq(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def seg_len(self) -> int:
        return int(self.seg_sec * self.sr)

    @property
    def delay(self) -> int:
        """流式算法延迟（样点）。= n_fft - hop = 24 ms @16k"""
        return self.n_fft - self.hop

    def update(self, kv: dict):
        for k, v in kv.items():
            if not hasattr(self, k):
                raise KeyError(f"未知配置项: {k}")
            cur = getattr(self, k)
            if isinstance(cur, bool):
                v = bool(int(v)) if str(v).isdigit() else str(v).lower() in ("1", "true", "yes")
            elif isinstance(cur, int):
                v = int(v)
            elif isinstance(cur, float):
                v = float(v)
            setattr(self, k, v)
        return self


def field_names():
    return [f.name for f in fields(Cfg)]
