"""GUI：两个选项卡 —— 训练 / 实时推理。单声道进、单声道出。

    python app.py

选项卡 1「训练」：选两个文件夹（你的声音 / 宿舍录音），一键从头训。
                 训练跑在子进程里，日志实时回显，可以随时中断。
选项卡 2「实时推理」：选 checkpoint，选输入/输出设备，实时提取。
                    输出可以选虚拟声卡（VB-Cable / VoiceMeeter），
                    这样 Zoom / Discord 拿到的就是处理后的声音。
"""
from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import threading
import time

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

SR = 16000
HOP = 128
HERE = os.path.dirname(os.path.abspath(__file__))
def default_python() -> str:
    """训练子进程用哪个解释器：环境变量 > 本地 python_path.txt > 当前解释器。

    要换一个带 CUDA 的 Python 训练，可以设置环境变量 VOICEFILTER_PYTHON，
    或在程序目录放一个 python_path.txt 写一行解释器路径（该文件已被 .gitignore 忽略）。
    """
    cand = os.environ.get("VOICEFILTER_PYTHON", "").strip()
    if not cand:
        p = os.path.join(HERE, "python_path.txt")
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    cand = f.readline().strip()
            except Exception:                   # noqa: BLE001
                cand = ""
    if cand and os.path.exists(cand):
        return cand
    return sys.executable


# ---------------------------------------------------------------- 重采样

class IntResampler:
    """整数倍升降采样，FIR 带状态，无块边界伪影、无漂移。"""

    def __init__(self, fs_in: int, fs_out: int, ntaps: int = 63):
        from scipy.signal import firwin
        if fs_in == fs_out:
            self.mode, self.factor = "same", 1
            return
        if fs_in > fs_out and fs_in % fs_out == 0:
            self.mode, self.factor = "down", fs_in // fs_out
            self.h = firwin(ntaps, 0.9 / self.factor)
        elif fs_out > fs_in and fs_out % fs_in == 0:
            self.mode, self.factor = "up", fs_out // fs_in
            self.h = firwin(ntaps, 0.9 / self.factor) * self.factor
        else:
            raise ValueError(f"不支持 {fs_in} -> {fs_out} 的整数倍变换，请把采样率设为 16000 或 48000")
        self.zi = np.zeros(len(self.h) - 1)

    def push(self, x: np.ndarray) -> np.ndarray:
        from scipy.signal import lfilter
        if self.mode == "same":
            return np.asarray(x, np.float32)
        x = np.asarray(x, np.float32)
        if self.mode == "down":
            y, self.zi = lfilter(self.h, 1.0, x, zi=self.zi)
            return y[::self.factor].astype(np.float32)
        z = np.zeros(len(x) * self.factor, np.float32)
        z[::self.factor] = x
        y, self.zi = lfilter(self.h, 1.0, z, zi=self.zi)
        return y.astype(np.float32)


# ---------------------------------------------------------------- 实时引擎

class Engine:
    def __init__(self, model, cfg, in_dev, out_dev, fs, block_ms=16):
        import sounddevice as sd
        from stream import StreamingTSE
        self.sd = sd
        self.cfg = cfg
        self.fs = fs
        self.in_dev, self.out_dev = in_dev, out_dev
        self.block = max(int(fs * block_ms / 1000), HOP)
        self.bypass = False
        self.gain_db = 6.0          # +6 dB = 幅度 ×2
        self._lim = 1.0
        self.running = False
        self.err = None
        self.level_in = 0.0
        self.level_out = 0.0
        self.st = StreamingTSE(model, cfg)
        self.down = IntResampler(fs, SR)
        self.up = IntResampler(SR, fs)
        self.buf16 = np.zeros(0, np.float32)
        self.out16 = np.zeros(0, np.float32)

    def start(self):
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self.running = False

    def _run(self):
        sd = self.sd
        try:
            with sd.InputStream(device=self.in_dev, channels=1, samplerate=self.fs,
                                blocksize=self.block, dtype="float32") as ins, \
                 sd.OutputStream(device=self.out_dev, channels=1, samplerate=self.fs,
                                 blocksize=self.block, dtype="float32") as outs:
                while self.running:
                    data, _ = ins.read(self.block)
                    x = data[:, 0].copy()
                    self.level_in = 0.9 * self.level_in + 0.1 * float(np.abs(x).max())
                    if self.bypass:
                        outs.write(x.reshape(-1, 1))
                        self.level_out = self.level_in
                        continue
                    self.buf16 = np.concatenate([self.buf16, self.down.push(x)])
                    while len(self.buf16) >= HOP:
                        self.out16 = np.concatenate([self.out16, self.st.process(self.buf16[:HOP])])
                        self.buf16 = self.buf16[HOP:]
                    if len(self.out16) == 0:
                        continue
                    y = self.up.push(self.out16)
                    self.out16 = np.zeros(0, np.float32)
                    y = self._apply_gain(y)
                    self.level_out = 0.9 * self.level_out + 0.1 * float(np.abs(y).max())
                    outs.write(y.reshape(-1, 1))
        except Exception as e:                      # noqa: BLE001
            self.err = e
            self.running = False

    def _apply_gain(self, y: np.ndarray) -> np.ndarray:
        """输出增益 + 峰值限制器。

        模型输出的是复掩码乘出来的信号，掩码本身不带绝对电平（cRM 是尺度无关的），
        两个人能量接近时 softmax 会把掩码压向 0.5，听起来就会明显变小。
        所以这里给一个可调的补偿增益。
        """
        if self.gain_db == 0.0 and self._lim >= 1.0:
            return y
        y = y * (10.0 ** (self.gain_db / 20.0))
        peak = float(np.abs(y).max()) if len(y) else 0.0
        if peak > 1e-6:
            tgt = min(1.0, 0.99 / peak)
            # 快压慢放：需要压时立刻压，恢复时缓慢，避免抽吸感
            self._lim = tgt if tgt < self._lim else self._lim + 0.02 * (tgt - self._lim)
            y = y * self._lim
        return y.astype(np.float32)

    @property
    def latency_ms(self):
        return (self.cfg.delay + 2 * self.block) / self.fs * 1000.0


# ---------------------------------------------------------------- 通用控件

def scan_devices():
    import sounddevice as sd
    devs = sd.query_devices()
    haps = sd.query_hostapis()
    ins, outs = [], []
    for i, d in enumerate(devs):
        tag = f"{i}: {d['name']}  [{haps[d['hostapi']]['name']}]"
        if d["max_input_channels"] > 0:
            ins.append((i, tag))
        if d["max_output_channels"] > 0:
            outs.append((i, tag))
    return ins, outs


class LogPane(ttk.Frame):
    def __init__(self, parent, height=16):
        super().__init__(parent)
        self.t = tk.Text(self, height=height, width=100, bg="#141414", fg="#c8c8c8",
                         insertbackground="#c8c8c8", relief="flat", font=("Consolas", 9))
        sb = ttk.Scrollbar(self, orient="vertical", command=self.t.yview)
        self.t.configure(yscrollcommand=sb.set)
        self.t.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    def write(self, s: str):
        self.t.insert("end", s)
        self.t.see("end")

    def clear(self):
        self.t.delete("1.0", "end")


def browse_dir(var: tk.StringVar, title: str):
    p = filedialog.askdirectory(title=title)
    if p:
        var.set(p)


# ---------------------------------------------------------------- 选项卡 1：训练

class TrainTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=12)
        self.proc = None
        self.q: queue.Queue[str] = queue.Queue()
        self.total_steps = 60000

        self.v_me = tk.StringVar(value="data/me")
        self.v_dorm = tk.StringVar(value="data/dorm")
        self.v_out = tk.StringVar(value="ckpt")
        self.v_steps = tk.StringVar(value="60000")
        self.v_batch = tk.StringVar(value="12")
        self.v_seg = tk.StringVar(value="3.0")
        self.v_lr = tk.StringVar(value="3e-4")
        self.v_resume = tk.StringVar(value="")
        self.v_py = tk.StringVar(value=default_python())

        r = 0
        self._row(r, "Python 解释器", self.v_py, "训练用它跑，必须带 CUDA 的 torch", is_file=True); r += 1
        self._row(r, "我的声音文件夹", self.v_me, "你一个人的录音，宿舍同麦、他人静音"); r += 1
        self._row(r, "宿舍声音文件夹", self.v_dorm, "只有室友说话、你不作声的录音"); r += 1
        self._row(r, "保存到", self.v_out, "checkpoint 输出目录", is_dir=True); r += 1
        self._row(r, "继续训练（可空）", self.v_resume, "上次的 last.pt", is_file=True); r += 1

        box = ttk.LabelFrame(self, text="超参", padding=8)
        box.grid(row=r, column=0, columnspan=4, sticky="we", pady=(8, 4)); r += 1
        for i, (lab, var, w) in enumerate([("迭代步数", self.v_steps, 8), ("batch", self.v_batch, 6),
                                           ("片段秒数", self.v_seg, 6), ("学习率", self.v_lr, 8)]):
            ttk.Label(box, text=lab).grid(row=0, column=i * 2, sticky="e", padx=(8, 2))
            ttk.Entry(box, textvariable=var, width=w).grid(row=0, column=i * 2 + 1, sticky="w")

        bar = ttk.Frame(self)
        bar.grid(row=r, column=0, columnspan=4, sticky="we", pady=8); r += 1
        self.btn_start = ttk.Button(bar, text="从头训练", width=14, command=self.start)
        self.btn_start.pack(side="left")
        self.btn_stop = ttk.Button(bar, text="停止", width=10, command=self.stop, state="disabled")
        self.btn_stop.pack(side="left", padx=6)
        ttk.Button(bar, text="检查数据", width=12, command=self.check).pack(side="left", padx=6)
        ttk.Button(bar, text="检测环境", width=12, command=self.check_env).pack(side="left", padx=6)
        self.pb = ttk.Progressbar(bar, length=260, maximum=int(self.v_steps.get()))
        self.pb.pack(side="left", padx=12)
        self.lbl = ttk.Label(bar, text="就绪", foreground="#888")
        self.lbl.pack(side="left", padx=6)

        self.log = LogPane(self, height=18)
        self.log.grid(row=r, column=0, columnspan=4, sticky="nsew", pady=6)
        self.rowconfigure(r, weight=1)
        self.columnconfigure(1, weight=1)
        self.log.write(f"GUI 解释器：{sys.executable}\n"
                       f"训练解释器：{self.v_py.get()}\n\n"
                       "点「检测环境」确认训练解释器里 cuda 可用（应显示 available True）。\n"
                       "训练前先点「检查数据」，确认两个文件夹的时长和文件数。\n"
                       "你的声音建议 ≥30 分钟；宿舍录音重点在不同时段、不同人数，而不是时长。\n\n")

        self.after(100, self._pump)

    def _row(self, r, label, var, hint, is_dir=False, is_file=False):
        ttk.Label(self, text=label, width=16, anchor="e").grid(row=r, column=0, sticky="e", padx=(0, 8), pady=3)
        ttk.Entry(self, textvariable=var, width=62).grid(row=r, column=1, sticky="we", pady=3)
        cmd = (lambda: browse_dir(var, hint)) if is_dir else (
            (lambda: self._pick_file(var)) if is_file else None)
        if cmd:
            ttk.Button(self, text="浏览", width=6, command=cmd).grid(row=r, column=2, padx=6)
        ttk.Label(self, text=hint, foreground="#777").grid(row=r, column=3, sticky="w", padx=6)

    def _pick_file(self, var):
        p = filedialog.askopenfilename(filetypes=[("python.exe", "*.exe"),
                                                  ("PyTorch", "*.pt"), ("全部", "*.*")])
        if p:
            var.set(p)

    # ---------------- 子进程
    def _cmd(self, extra: list[str]) -> list[str]:
        py = (self.v_py.get().strip() or sys.executable)
        c = [py, "-u", "train.py",
             "--target_dir", self.v_me.get(), "--interf_dir", self.v_dorm.get(),
             "--ckpt_dir", self.v_out.get()]
        if extra:
            c += extra
        return c

    def check_env(self):
        py = self.v_py.get().strip() or sys.executable
        code = ("import sys,torch;"
                "print('python', sys.version.split()[0]);"
                "print('torch', torch.__version__);"
                "print('cuda build', torch.version.cuda, 'available', torch.cuda.is_available());"
                "print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无')")
        self.q.put(f"\n>>> 检测环境：{py}\n")
        try:
            r = subprocess.run([py, "-c", code], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=180, cwd=HERE)
            out = (r.stdout or "") + (r.stderr or "")
            for ln in out.strip().splitlines():
                self.q.put("    " + ln + "\n")
            self.q.put("    -> " + ("GPU 可用，训练会走 cuda" if "available True" in out
                                    else "!! CUDA 不可用，训练会退回 cpu") + "\n")
        except Exception as e:                      # noqa: BLE001
            self.q.put(f"    失败：{e}\n")

    def check(self):
        self.q.put("\n>>> 检查数据\n")
        self._run(self._cmd(["--stats"]), on_exit_reset=False)

    def start(self):
        if self.proc and self.proc.poll() is None:
            return
        out = self.v_out.get()
        os.makedirs(out, exist_ok=True)
        self.total_steps = max(1, int(float(self.v_steps.get())))
        self.pb.config(maximum=self.total_steps, value=0)
        args = ["--steps", str(int(self.total_steps)),
                "--batch", str(int(float(self.v_batch.get()))),
                "--lr", self.v_lr.get(),
                "--set", f"seg_sec={self.v_seg.get()}"]
        if self.v_resume.get().strip():
            args += ["--resume", self.v_resume.get().strip()]
        self.q.put(f"\n>>> 解释器  {self.v_py.get()}\n")
        self.q.put(f">>> 开始训练  {self.v_me.get()}  +  {self.v_dorm.get()}\n")
        self._run(self._cmd(args))
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.lbl.config(text="训练中…", foreground="#3fb950")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.q.put("\n>>> 已请求停止\n")
        self._reset()

    def _run(self, cmd, on_exit_reset=True):
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        try:
            self.proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True,
                                         encoding="utf-8", errors="replace",
                                         bufsize=1, env=env)
        except Exception as e:                      # noqa: BLE001
            messagebox.showerror("启动失败", str(e))
            return
        self._on_exit_reset = on_exit_reset
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()

    def _reader(self, p):
        for line in p.stdout:
            self.q.put(line)
        p.wait()
        self.q.put(("__EXIT__", p.returncode, p.pid))

    def _pump(self):
        try:
            while True:
                item = self.q.get_nowait()
                if isinstance(item, tuple):
                    _, code, pid = item
                    if self.proc is not None and pid == self.proc.pid:
                        self.log.write(f"\n>>> 进程结束（code={code}）\n")
                        if getattr(self, "_on_exit_reset", True):
                            self._reset()
                    continue
                line = item
                self.log.write(line)
                m = re.search(r"step\s+(\d+)\s+loss\s+([\d.]+)", line)
                if m:
                    self.pb.config(value=int(m.group(1)))
                    self.lbl.config(text=f"step {m.group(1)} / {self.total_steps}   loss {m.group(2)}")
                    continue
                m = re.search(r"\[val\].*SI-SDRi\s+([+-][\d.]+)", line)
                if m:
                    self.lbl.config(text=f"验证 SI-SDRi {m.group(1)} dB")
        except queue.Empty:
            pass
        self.after(120, self._pump)

    def _reset(self):
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.lbl.config(text="已停止", foreground="#888")


# ---------------------------------------------------------------- 选项卡 2：推理

class InferTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=12)
        self.engine = None
        self.cfg = None

        ins, outs = [], []
        try:
            ins, outs = scan_devices()
        except Exception as e:                      # noqa: BLE001
            messagebox.showerror("音频设备", f"无法枚举设备：{e}\n\n先 pip install sounddevice")
        self.ins, self.outs = ins, outs

        self.v_ckpt = tk.StringVar(value=os.path.join("ckpt", "best.pt"))
        self.v_fs = tk.StringVar(value="16000")
        self.v_blk = tk.StringVar(value="16")

        r = 0
        ttk.Label(self, text="模型", width=10, anchor="e").grid(row=r, column=0, sticky="e", padx=(0, 8), pady=3)
        ttk.Entry(self, textvariable=self.v_ckpt, width=64).grid(row=r, column=1, sticky="we")
        ttk.Button(self, text="浏览", width=6, command=self.pick_ckpt).grid(row=r, column=2, padx=6)
        ttk.Label(self, text=".pt 文件或包含 best.pt 的目录", foreground="#777").grid(row=r, column=3, sticky="w")
        r += 1

        ttk.Label(self, text="输入设备", width=10, anchor="e").grid(row=r, column=0, sticky="e", padx=(0, 8), pady=3)
        self.cb_in = ttk.Combobox(self, width=62, values=[n for _, n in ins])
        self.cb_in.grid(row=r, column=1, sticky="we")
        if ins:
            self.cb_in.current(0)
        ttk.Button(self, text="刷新", width=6, command=self.refresh).grid(row=r, column=2, padx=6)
        ttk.Label(self, text="单声道", foreground="#777").grid(row=r, column=3, sticky="w")
        r += 1

        ttk.Label(self, text="输出设备", width=10, anchor="e").grid(row=r, column=0, sticky="e", padx=(0, 8), pady=3)
        self.cb_out = ttk.Combobox(self, width=62, values=[n for _, n in outs])
        self.cb_out.grid(row=r, column=1, sticky="we")
        if outs:
            self.cb_out.current(0)
        ttk.Label(self, text="选虚拟声卡可接管通话", foreground="#777").grid(row=r, column=3, sticky="w")
        r += 1

        ttk.Label(self, text="采样率", width=10, anchor="e").grid(row=r, column=0, sticky="e", padx=(0, 8), pady=3)
        f2 = ttk.Frame(self)
        f2.grid(row=r, column=1, columnspan=3, sticky="w")
        self.cb_fs = ttk.Combobox(f2, width=8, values=["16000", "48000", "32000"], state="readonly",
                                  textvariable=self.v_fs)
        self.cb_fs.pack(side="left")
        ttk.Label(f2, text="   块大小 ms").pack(side="left")
        ttk.Combobox(f2, width=6, values=["8", "16", "32"], state="readonly",
                     textvariable=self.v_blk).pack(side="left", padx=4)
        r += 1

        ttk.Label(self, text="输出增益", width=10, anchor="e").grid(row=r, column=0, sticky="e", padx=(0, 8), pady=3)
        f3 = ttk.Frame(self)
        f3.grid(row=r, column=1, columnspan=3, sticky="w")
        self.v_gain = tk.DoubleVar(value=6.0)
        ttk.Scale(f3, from_=-12, to=18, variable=self.v_gain, length=220,
                  command=self._on_gain).pack(side="left")
        self.lbl_gain = ttk.Label(f3, text="+6.0 dB（×2.0）", width=18)
        self.lbl_gain.pack(side="left", padx=8)
        ttk.Label(f3, text="模型输出是无量纲掩码乘出来的，通常偏小", foreground="#777").pack(side="left")
        r += 1

        bar = ttk.Frame(self)
        bar.grid(row=r, column=0, columnspan=4, sticky="we", pady=10); r += 1
        self.btn_start = ttk.Button(bar, text="启动", width=10, command=self.start)
        self.btn_start.pack(side="left")
        self.btn_stop = ttk.Button(bar, text="停止", width=10, command=self.stop, state="disabled")
        self.btn_stop.pack(side="left", padx=6)
        self.v_bypass = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="旁路（A/B 对比）", variable=self.v_bypass,
                        command=self.toggle_bypass).pack(side="left", padx=12)
        self.lbl = ttk.Label(bar, text="就绪", foreground="#888")
        self.lbl.pack(side="left", padx=8)

        self.m_in = self._meter(r, "输入"); r += 1
        self.m_out = self._meter(r, "输出"); r += 1
        self.log = LogPane(self, height=9)
        self.log.grid(row=r, column=0, columnspan=4, sticky="nsew", pady=6)
        self.rowconfigure(r, weight=1)
        self.columnconfigure(1, weight=1)
        self.after(50, self._tick)

    def _meter(self, row, label):
        f = ttk.Frame(self)
        f.grid(row=row, column=0, columnspan=4, sticky="we", pady=2)
        ttk.Label(f, text=label, width=4).pack(side="left")
        c = tk.Canvas(f, height=12, width=560, bg="#222", highlightthickness=0)
        c.pack(side="left", padx=6)
        return c

    def refresh(self):
        try:
            self.ins, self.outs = scan_devices()
        except Exception as e:                      # noqa: BLE001
            messagebox.showerror("音频设备", str(e))
            return
        self.cb_in["values"] = [n for _, n in self.ins]
        self.cb_out["values"] = [n for _, n in self.outs]

    def _dev_id(self, cb, table):
        s = cb.get()
        return next((i for i, n in table if n == s), None)

    def pick_ckpt(self):
        p = filedialog.askopenfilename(filetypes=[("PyTorch", "*.pt"), ("全部", "*.*")])
        if p:
            self.v_ckpt.set(p)

    def _draw(self, canvas, v):
        canvas.delete("all")
        w = int(canvas["width"])
        frac = float(np.clip((20 * np.log10(max(v, 1e-6)) + 60) / 60, 0, 1))
        col = "#3fb950" if frac < 0.75 else ("#d29922" if frac < 0.92 else "#f85149")
        canvas.create_rectangle(0, 0, int(w * frac), 12, fill=col, width=0)

    def _tick(self):
        e = self.engine
        if e:
            self._draw(self.m_in, e.level_in)
            self._draw(self.m_out, e.level_out)
            if e.err:
                self.log.write(f"!! 音频线程异常：{e.err}\n")
                e.err = None
                self.stop()
            elif e.running:
                vad = e.st.last_vad
                self.lbl.config(
                    text=f"运行中  延迟≈{e.latency_ms:.0f} ms  目标VAD {vad:.2f} "
                         f"{'（说话）' if vad > 0.5 else '（静音）'}",
                    foreground="#3fb950" if vad > 0.5 else "#888")
        self.after(50, self._tick)

    def toggle_bypass(self):
        if self.engine:
            self.engine.bypass = self.v_bypass.get()

    def _on_gain(self, _=None):
        db = float(self.v_gain.get())
        self.lbl_gain.config(text=f"{db:+.1f} dB（×{10 ** (db / 20):.1f}）")
        if self.engine:
            self.engine.gain_db = db

    def load(self):
        import torch
        from config import Cfg
        from model import TSE
        p = self.v_ckpt.get()
        if os.path.isdir(p):
            p = os.path.join(p, "best.pt")
            if not os.path.exists(p):
                p = os.path.join(os.path.dirname(p), "last.pt")
        if not os.path.exists(p):
            raise FileNotFoundError(f"找不到 {p}")
        ck = torch.load(p, map_location="cpu", weights_only=False)
        cfg = Cfg(**{k: v for k, v in ck.get("cfg", {}).items() if k in Cfg.__dataclass_fields__})
        model = TSE(cfg)
        model.load_state_dict(ck["model"])
        model.eval()
        self.cfg = cfg
        return model, cfg, p

    def start(self):
        d_in = self._dev_id(self.cb_in, self.ins)
        d_out = self._dev_id(self.cb_out, self.outs)
        if d_in is None or d_out is None:
            messagebox.showwarning("设备", "输入和输出都必须选")
            return
        try:
            model, cfg, p = self.load()
        except Exception as e:                      # noqa: BLE001
            messagebox.showerror("模型", str(e))
            return
        try:
            eng = Engine(model, cfg, d_in, d_out, int(self.v_fs.get()), int(self.v_blk.get()))
            eng.gain_db = float(self.v_gain.get())
        except Exception as e:                      # noqa: BLE001
            messagebox.showerror("音频初始化", f"{e}\n\n请把采样率改成 16000 或 48000。")
            return

        eng.start()
        self.engine = eng
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.log.write(f"启动 {os.path.basename(p)}  输入#{d_in} 输出#{d_out}  "
                       f"{self.v_fs.get()} Hz  延迟≈{eng.latency_ms:.0f} ms\n")
        self.log.write("提示：输出选 VB-Cable / VoiceMeeter，再在 Zoom/Discord 里把麦克风设成同一路，即可接管通话。\n")

    def stop(self):
        if self.engine:
            self.engine.stop()
            time.sleep(0.1)
            self.engine = None
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.lbl.config(text="已停止", foreground="#888")
        self._draw(self.m_in, 0)
        self._draw(self.m_out, 0)


# ---------------------------------------------------------------- 主窗口

class App:
    def __init__(self, root: tk.Tk):
        root.title("目标说话人提取 — 训练 / 实时推理")
        root.geometry("1000x760")
        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True, padx=8, pady=8)
        self.train_tab = TrainTab(nb)
        self.infer_tab = InferTab(nb)
        nb.add(self.train_tab, text="   训练   ")
        nb.add(self.infer_tab, text="   实时推理   ")


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:                               # noqa: BLE001
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
