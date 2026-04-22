"""
Loona 唤醒模块 Demo - Tkinter GUI
=================================

在 wake_demo.py 的核心引擎上封装一个可视化界面：
- 选择麦克风设备、触发模式（multimodal/vad/kws）、唤醒词、基础阈值
- 开始/停止监听
- 实时显示 KWS 分数 / 动态阈值 / SNR / VAD / 状态
- 多模态模式下显示：注视/唇动/DOA/近场四硬门限状态灯 + 摄像头预览
- 唤醒事件高亮日志 + 生成 wav 列表（双击可播放）
- 模拟 "TTS 播报中"（方案 §5.3）以验证阈值抬升

运行：
    python wake_gui.py
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, scrolledtext
import wave

import sounddevice as sd

# 复用核心引擎
from wake_demo import WakeDemo, SAMPLE_RATE

try:
    import cv2
    from PIL import Image, ImageTk
    HAVE_PREVIEW = True
except ImportError:
    HAVE_PREVIEW = False

AVAILABLE_WAKEWORDS = ["hey_jarvis", "alexa", "hey_mycroft", "hey_rhasspy", "timer", "weather"]


class WakeGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Loona 唤醒模块 Demo")
        self.geometry("1080x640")
        self.minsize(960, 560)

        self.engine: WakeDemo | None = None
        self.engine_thread: threading.Thread | None = None
        self.event_q: queue.Queue = queue.Queue()
        self._last_gates = {}
        self._preview_photo = None  # 防止被 GC

        self._build_ui()
        self._refresh_devices()
        self.after(80, self._drain_events)
        self.after(100, self._refresh_preview)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------ UI ------------------------ #
    def _build_ui(self):
        pad = {"padx": 4, "pady": 2}
        small = ("Segoe UI", 9)

        # ============ 上部：左(参数+实时监测) + 右(控制栏+摄像头) ============
        upper = ttk.Frame(self)
        upper.pack(fill="both", expand=False, padx=6, pady=(4, 2))
        upper.columnconfigure(0, weight=0, minsize=520)
        upper.columnconfigure(1, weight=1, minsize=280)

        # ------- 左列：参数 -------
        top = ttk.LabelFrame(upper, text="参数")
        top.grid(row=0, column=0, sticky="nsew", padx=(0, 4))

        # 行 0：设备 + 基础阈值
        ttk.Label(top, text="设备:", font=small).grid(row=0, column=0, sticky="e", **pad)
        self.cmb_device = ttk.Combobox(top, width=28, state="readonly", font=small)
        self.cmb_device.grid(row=0, column=1, columnspan=2, sticky="we", **pad)
        ttk.Button(top, text="↻", command=self._refresh_devices, width=2).grid(row=0, column=3, **pad)

        ttk.Label(top, text="阈值:", font=small).grid(row=0, column=4, sticky="e", **pad)
        self.var_thresh = tk.DoubleVar(value=0.35)
        self.scl_thresh = ttk.Scale(top, from_=0.2, to=0.9, variable=self.var_thresh,
                                    orient="horizontal", length=90,
                                    command=self._on_thresh_change)
        self.scl_thresh.grid(row=0, column=5, sticky="we", **pad)
        self.lbl_thresh_val = ttk.Label(top, text="0.35", width=4, font=small)
        self.lbl_thresh_val.grid(row=0, column=6, sticky="w", **pad)

        # 行 1：触发模式（单选）+ 调试复选
        ttk.Label(top, text="模式:", font=small).grid(row=1, column=0, sticky="e", **pad)
        self.var_mode = tk.StringVar(value="multimodal")
        mode_frame = ttk.Frame(top)
        mode_frame.grid(row=1, column=1, columnspan=6, sticky="w", **pad)
        ttk.Radiobutton(mode_frame, text="多模态", variable=self.var_mode,
                        value="multimodal", command=self._on_mode_change).pack(side="left")
        ttk.Radiobutton(mode_frame, text="VAD", variable=self.var_mode,
                        value="vad", command=self._on_mode_change).pack(side="left", padx=6)
        ttk.Radiobutton(mode_frame, text="KWS", variable=self.var_mode,
                        value="kws", command=self._on_mode_change).pack(side="left")
        self.var_no_vad = tk.BooleanVar(value=False)
        ttk.Checkbutton(mode_frame, text="关VAD门控",
                        variable=self.var_no_vad).pack(side="left", padx=10)
        self.var_tts = tk.BooleanVar(value=False)
        ttk.Checkbutton(mode_frame, text="模拟TTS",
                        variable=self.var_tts, command=self._on_tts_toggle).pack(side="left")

        # 唤醒词（隐藏 UI，仍保留变量用于 KWS 模式启动）
        self.cmb_wake = ttk.Combobox(self, values=AVAILABLE_WAKEWORDS,
                                     state="readonly", width=14)
        self.cmb_wake.set("hey_jarvis")
        # 不 pack/grid 到窗口 → 不显示

        # 行 2：VAD 子参数
        self.vad_frame = ttk.Frame(top)
        self.vad_frame.grid(row=2, column=0, columnspan=7, sticky="we", **pad)
        ttk.Label(self.vad_frame, text="语音≥(ms):", font=small).pack(side="left")
        self.var_min_speech = tk.IntVar(value=400)
        ttk.Spinbox(self.vad_frame, from_=150, to=2000, increment=50,
                    textvariable=self.var_min_speech, width=5,
                    font=small).pack(side="left", padx=2)
        ttk.Label(self.vad_frame, text="SNR≥(dB):", font=small).pack(side="left", padx=(8, 0))
        self.var_min_snr = tk.DoubleVar(value=4.0)
        ttk.Spinbox(self.vad_frame, from_=0.0, to=25.0, increment=0.5,
                    textvariable=self.var_min_snr, width=5, font=small,
                    format="%.1f").pack(side="left", padx=2)

        # KWS 子面板（与 vad_frame 共用一行）
        self.kws_frame = ttk.Frame(top)
        self.kws_frame.grid(row=3, column=0, columnspan=7, sticky="we", **pad)
        ttk.Label(self.kws_frame, text="(KWS 模式使用默认唤醒词 hey_jarvis)",
                  font=small, foreground="#888").pack(side="left")

        # 行 4：多模态专属参数
        self.mm_frame = ttk.Frame(top)
        self.mm_frame.grid(row=4, column=0, columnspan=7, sticky="we", **pad)
        ttk.Label(self.mm_frame, text="近场RMS:", font=small).pack(side="left")
        self.var_nf_rms = tk.IntVar(value=400)
        ttk.Spinbox(self.mm_frame, from_=200, to=5000, increment=50,
                    textvariable=self.var_nf_rms, width=6,
                    font=small).pack(side="left", padx=2)
        ttk.Label(self.mm_frame, text="DOA±°:", font=small).pack(side="left", padx=(6, 0))
        self.var_doa = tk.IntVar(value=30)
        ttk.Spinbox(self.mm_frame, from_=5, to=90, increment=5,
                    textvariable=self.var_doa, width=4,
                    font=small).pack(side="left", padx=2)
        ttk.Label(self.mm_frame, text="声道:", font=small).pack(side="left", padx=(6, 0))
        self.var_ch = tk.IntVar(value=2)
        ttk.Spinbox(self.mm_frame, from_=1, to=4, increment=1,
                    textvariable=self.var_ch, width=3,
                    font=small).pack(side="left", padx=2)
        ttk.Label(self.mm_frame, text="摄像头:", font=small).pack(side="left", padx=(6, 0))
        self.var_cam = tk.IntVar(value=0)
        ttk.Spinbox(self.mm_frame, from_=0, to=4, increment=1,
                    textvariable=self.var_cam, width=3,
                    font=small).pack(side="left", padx=2)
        self.var_no_cam = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.mm_frame, text="禁用摄像头",
                        variable=self.var_no_cam).pack(side="left", padx=6)

        top.columnconfigure(1, weight=1)
        top.columnconfigure(2, weight=1)
        self._update_mode_panels()

        # ------- 右列：控制栏 + 摄像头预览 -------
        right = ttk.Frame(upper)
        right.grid(row=0, column=1, rowspan=2, sticky="nsew")

        ctrl = ttk.LabelFrame(right, text="控制")
        ctrl.pack(fill="x", pady=(0, 2))
        ctrl_row = ttk.Frame(ctrl)
        ctrl_row.pack(fill="x", padx=4, pady=2)
        self.btn_start = ttk.Button(ctrl_row, text="▶ 开始", command=self.start_engine, width=8)
        self.btn_start.pack(side="left", padx=2)
        self.btn_stop = ttk.Button(ctrl_row, text="■ 停止", command=self.stop_engine,
                                   width=7, state="disabled")
        self.btn_stop.pack(side="left", padx=2)
        self.lbl_state = ttk.Label(ctrl_row, text="● IDLE", foreground="gray",
                                   font=("Consolas", 9, "bold"))
        self.lbl_state.pack(side="left", padx=6)

        count_row = ttk.Frame(ctrl)
        count_row.pack(fill="x", padx=4, pady=(0, 2))
        ttk.Label(count_row, text="唤醒次数:", font=small).pack(side="left")
        self.wake_count = 0
        self.lbl_wake_count = ttk.Label(count_row, text="0", foreground="#d62828",
                                        font=("Consolas", 13, "bold"), width=4)
        self.lbl_wake_count.pack(side="left", padx=2)
        self.btn_reset_count = ttk.Button(count_row, text="清零", width=5,
                                          command=self._reset_wake_count)
        self.btn_reset_count.pack(side="left", padx=2)

        cam_frame = ttk.LabelFrame(right, text="摄像头预览")
        cam_frame.pack(fill="both", expand=True)
        self.lbl_preview = ttk.Label(cam_frame, text="(未启动)", font=small,
                                     width=30, anchor="center")
        self.lbl_preview.pack(fill="both", expand=True, padx=2, pady=2)

        # ============ 左列：实时监测（紧凑） ============
        mon = ttk.LabelFrame(upper, text="实时监测")
        mon.grid(row=1, column=0, sticky="nsew", padx=(0, 4), pady=(2, 0))
        upper.rowconfigure(1, weight=0)

        self._add_bar(mon, "KWS 分数", "score", 0)
        self._add_bar(mon, "动态阈值", "thresh", 1, maximum=1.0)
        self._add_bar(mon, "SNR (dB)", "snr", 2, maximum=60)
        self._bar_snr_offset = 30

        status = ttk.Frame(mon)
        status.grid(row=3, column=0, columnspan=3, sticky="we", padx=4, pady=1)
        self.lbl_vad = ttk.Label(status, text="VAD: ---", width=11, font=small)
        self.lbl_vad.pack(side="left")
        self.lbl_word = ttk.Label(status, text="Word: ---", width=16, font=small,
                                  anchor="w")
        self.lbl_word.pack(side="left", padx=4)
        self.lbl_reason = ttk.Label(status, text="Reason: ---", width=22, font=small,
                                    anchor="w")
        self.lbl_reason.pack(side="left", padx=4)

        self.lbl_audio_diag = ttk.Label(mon,
            text="RMS=---  centroid=---Hz  pitch=---Hz  DOA=---",
            font=("Consolas", 8), foreground="#555")
        self.lbl_audio_diag.grid(row=4, column=0, columnspan=3, sticky="w", padx=4, pady=(0, 2))
        mon.columnconfigure(1, weight=1)

        # ============ 硬门限灯排（全宽，紧凑） ============
        gates_frame = ttk.LabelFrame(self, text="硬门限 (多模态模式下全部命中才触发)")
        gates_frame.pack(fill="x", padx=6, pady=2)
        self.gate_labels = {}
        self.gate_names = {
            "voiced": "浊音", "snr": "SNR", "speech_run": "语音时长",
            "near_field": "近场", "doa": "DOA",
            "face": "人脸", "gaze": "注视", "lip": "唇动",
            "av_sync": "唇声同步",
        }
        for i, (k, label) in enumerate(self.gate_names.items()):
            lbl = ttk.Label(gates_frame, text=f"○ {label}", width=10,
                            font=small, foreground="#888")
            lbl.grid(row=0, column=i, padx=3, pady=2)
            self.gate_labels[k] = lbl

        # ============ 底部：日志 + ASR ============
        bot = ttk.Panedwindow(self, orient="horizontal")
        bot.pack(fill="both", expand=True, padx=6, pady=(2, 4))

        log_frame = ttk.LabelFrame(bot, text="事件日志")
        self.txt_log = scrolledtext.ScrolledText(log_frame, height=6, font=("Consolas", 8))
        self.txt_log.pack(fill="both", expand=True)
        self.txt_log.tag_config("wake", foreground="#d62828", font=("Consolas", 8, "bold"))
        self.txt_log.tag_config("info", foreground="#444")
        self.txt_log.tag_config("warn", foreground="#b58105")
        bot.add(log_frame, weight=3)

        asr_frame = ttk.LabelFrame(bot, text="ASR 转写结果")
        self.txt_asr = scrolledtext.ScrolledText(asr_frame, height=6,
                                                 font=("Microsoft YaHei", 9), wrap="word")
        self.txt_asr.pack(fill="both", expand=True)
        self.txt_asr.tag_config("idx", foreground="#888", font=("Consolas", 8))
        self.txt_asr.tag_config("text", foreground="#111",
                                font=("Microsoft YaHei", 10, "bold"))
        self.txt_asr.tag_config("meta", foreground="#666", font=("Consolas", 7))
        self.txt_asr.tag_config("pending", foreground="#b58105",
                                font=("Microsoft YaHei", 9, "italic"))
        self.txt_asr.tag_config("reject", foreground="#b02a37",
                                font=("Microsoft YaHei", 9, "italic"))
        bot.add(asr_frame, weight=3)

    def _add_bar(self, parent, label, key, row, maximum=1.0):
        ttk.Label(parent, text=label, width=14,
                  font=("Segoe UI", 9)).grid(row=row, column=0, sticky="w", padx=4, pady=1)
        bar = ttk.Progressbar(parent, orient="horizontal", maximum=maximum, length=280)
        bar.grid(row=row, column=1, sticky="we", padx=4, pady=1)
        val = ttk.Label(parent, text="0.00", width=10, font=("Consolas", 8))
        val.grid(row=row, column=2, sticky="w", padx=4)
        setattr(self, f"bar_{key}", bar)
        setattr(self, f"lbl_{key}", val)

    # ------------------------ 控制 ------------------------ #
    def _refresh_devices(self):
        try:
            devs = sd.query_devices()
        except Exception as e:
            self._log(f"查询设备失败: {e}", "warn")
            return
        self.devices = []
        names = []
        for i, d in enumerate(devs):
            if d.get("max_input_channels", 0) > 0:
                self.devices.append(i)
                names.append(f"[{i}] {d['name']} ({d['hostapi']} in={d['max_input_channels']})")
        self.cmb_device["values"] = names
        if names and not self.cmb_device.get():
            # 选默认输入
            try:
                default_in = sd.default.device[0]
            except Exception:
                default_in = self.devices[0]
            for idx, di in enumerate(self.devices):
                if di == default_in:
                    self.cmb_device.current(idx)
                    break
            else:
                self.cmb_device.current(0)

    def _selected_device(self):
        idx = self.cmb_device.current()
        if idx < 0 or idx >= len(self.devices):
            return None
        return self.devices[idx]

    def _on_thresh_change(self, _):
        t = self.var_thresh.get()
        self.lbl_thresh_val.config(text=f"{t:.2f}")
        if self.engine is not None:
            self.engine.arbiter.base_thresh = float(t)

    def _on_tts_toggle(self):
        if self.engine is not None:
            self.engine.arbiter.tts_playing = bool(self.var_tts.get())
        self._log(f"模拟 TTS 播报中 = {self.var_tts.get()}", "info")

    def _on_mode_change(self):
        self._update_mode_panels()
        self._log(f"触发模式切换为: {self.var_mode.get()}", "info")

    def _update_mode_panels(self):
        mode = self.var_mode.get()
        state_kws = "normal" if mode == "kws" else "disabled"
        state_vad = "normal" if mode in ("vad", "multimodal") else "disabled"
        state_mm = "normal" if mode == "multimodal" else "disabled"
        try:
            for w in self.kws_frame.winfo_children():
                w.configure(state=state_kws)
            for w in self.vad_frame.winfo_children():
                w.configure(state=state_vad)
            for w in self.mm_frame.winfo_children():
                w.configure(state=state_mm)
        except Exception:
            pass

    def start_engine(self):
        if self.engine is not None:
            return
        device = self._selected_device()
        mode = self.var_mode.get()
        wake = self.cmb_wake.get() or "hey_jarvis"
        thresh = float(self.var_thresh.get())
        args = argparse.Namespace(
            device=device,
            list_devices=False,
            mode=mode,
            wakewords=[wake],
            base_thresh=thresh,
            min_speech_ms=float(self.var_min_speech.get()),
            min_snr_db=float(self.var_min_snr.get()),
            near_field_rms=float(self.var_nf_rms.get()),
            audio_channels=int(self.var_ch.get()),
            doa_cone_deg=float(self.var_doa.get()),
            camera_index=int(self.var_cam.get()),
            no_camera=bool(self.var_no_cam.get()),
            no_strict_multimodal=False,
            no_vad_gate=bool(self.var_no_vad.get()),
            asr_model="tiny",
            verbose=False,
        )
        self._log(f"初始化引擎: mode={mode} device={device} thresh={thresh:.2f}", "info")
        try:
            self.engine = WakeDemo(args)
        except Exception as e:
            self._log(f"引擎初始化失败: {e}", "warn")
            self.engine = None
            return
        self.engine.on_event = self._engine_cb
        self.engine.arbiter.tts_playing = bool(self.var_tts.get())
        self.engine_thread = threading.Thread(target=self.engine.run, daemon=True)
        self.engine_thread.start()
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")

    def stop_engine(self):
        if self.engine is None:
            return
        self.engine.stop_flag.set()
        self._log("停止中...", "info")
        self.btn_stop.config(state="disabled")

    def _on_close(self):
        self.stop_engine()
        self.after(300, self.destroy)

    # ------------------------ 事件桥接 ------------------------ #
    def _engine_cb(self, event_type, payload):
        # 来自后台线程，塞队列由主线程消费
        self.event_q.put((event_type, payload))

    def _drain_events(self):
        try:
            while True:
                ev, pl = self.event_q.get_nowait()
                self._handle_event(ev, pl)
        except queue.Empty:
            pass
        self.after(60, self._drain_events)

    def _handle_event(self, ev, pl):
        if ev == "tick":
            score = pl["score"]
            thresh = pl["threshold"]
            snr = pl["snr"]
            self.bar_score["value"] = max(0.0, min(1.0, score))
            self.lbl_score.config(text=f"{score:.2f}")
            self.bar_thresh["value"] = thresh
            self.lbl_thresh.config(text=f"{thresh:.2f}")
            self.bar_snr["value"] = max(0.0, min(60.0, snr + self._bar_snr_offset))
            self.lbl_snr.config(text=f"{snr:+.1f} dB")
            self.lbl_vad.config(text=f"VAD: {'●' if pl['vad'] else '○'} {pl['vad']}")
            word_txt = str(pl['word'])
            if len(word_txt) > 10:
                word_txt = word_txt[:9] + "…"
            self.lbl_word.config(text=f"Word: {word_txt}")
            reason_txt = str(pl['reason'])
            if len(reason_txt) > 16:
                reason_txt = reason_txt[:15] + "…"
            self.lbl_reason.config(text=f"Reason: {reason_txt}")
            # 音频诊断
            rms = pl.get("rms")
            centroid = pl.get("centroid")
            pitch = pl.get("pitch")
            doa = pl.get("doa")
            if rms is not None:
                doa_str = f"{doa:+.1f}°" if doa is not None else "n/a"
                self.lbl_audio_diag.config(
                    text=(f"RMS={rms:7.1f}  centroid={centroid:6.0f}Hz  "
                          f"pitch={(pitch or 0):5.1f}Hz  DOA={doa_str}")
                )
            # 硬门限状态灯
            gates = pl.get("gates") or {}
            self._last_gates = gates
            for k, lbl in self.gate_labels.items():
                name = self.gate_names.get(k, k)
                v = gates.get(k)
                if v is True:
                    lbl.config(text=f"● {name}", foreground="#2b9348")
                elif v is False:
                    lbl.config(text=f"✗ {name}", foreground="#c0392b")
                else:
                    lbl.config(text=f"○ {name}", foreground="#888")
        elif ev == "wake":
            cnt = pl.get("count", self.wake_count + 1)
            self.wake_count = cnt
            self.lbl_wake_count.config(text=str(cnt))
            self._log(
                f">>> WAKE #{cnt}! word={pl['word']} score={pl['score']:.2f} | 语音 {pl['duration']:.2f}s → ASR...",
                "wake",
            )
            # 占位一行等待 ASR 回填
            self.txt_asr.insert("end", f"#{cnt}  ", "idx")
            self.txt_asr.insert("end", f"[{pl['word']} {pl['score']:.2f}] ", "meta")
            self.txt_asr.insert("end", "识别中...\n", "pending")
            self.txt_asr.see("end")
            self._pending_asr_idx = cnt
        elif ev == "asr":
            cnt = pl.get("count", self.wake_count)
            text = pl.get("text") or ""
            err = pl.get("error")
            rejected = bool(pl.get("rejected"))
            reject_reason = pl.get("reject_reason") or ""
            if rejected:
                # 回退 UI 计数（引擎已自减 wake_count）
                if self.engine is not None:
                    self.wake_count = self.engine.wake_count
                else:
                    self.wake_count = max(0, self.wake_count - 1)
                self.lbl_wake_count.config(text=str(self.wake_count))
                self._log(
                    f"✗ 撤销唤醒 #{cnt}: {reject_reason}"
                    + (f" | 文本={text!r}" if text else ""),
                    "warn",
                )
                display = f"✗ 撤销 ({reject_reason})"
                if text:
                    display += f" 文本={text!r}"
                display += "\n"
                try:
                    self.txt_asr.insert("end", f"    → ", "meta")
                    self.txt_asr.insert("end", display, "reject")
                    self.txt_asr.see("end")
                except Exception:
                    self.txt_asr.insert("end", display, "reject")
                    self.txt_asr.see("end")
                return
            if err:
                line = f"(ASR 失败: {err})\n"
                self._log(f"ASR #{cnt} 失败: {err}", "warn")
            else:
                line = (text if text else "(未识别出文字)") + "\n"
                self._log(f"ASR #{cnt}: {text!r}", "info")
            # 替换最后一行的 "识别中..."
            try:
                end_idx = self.txt_asr.index("end-1c")
                line_start = self.txt_asr.index(f"{end_idx} linestart")
                # 找到本行中 "识别中..." 的位置并替换
                cur_line = self.txt_asr.get(f"{end_idx} -1l linestart", f"{end_idx} -1l lineend")
                # 简化：直接在新行追加
                self.txt_asr.insert("end", f"    → ", "meta")
                self.txt_asr.insert("end", line, "text")
                self.txt_asr.see("end")
            except Exception:
                self.txt_asr.insert("end", line, "text")
                self.txt_asr.see("end")
        elif ev == "state":
            st = pl["state"]
            color = "#2b9348" if st == "LISTENING" else "gray"
            self.lbl_state.config(text=f"● {st}", foreground=color)
            self._log(f"状态切换 -> {st} ({pl.get('reason', '')})", "info")
        elif ev == "started":
            self.lbl_state.config(text="● IDLE", foreground="gray")
            extra = ""
            if "audio_channels" in pl:
                extra = f" channels={pl['audio_channels']} visual={pl.get('visual_on')}"
            self._log(f"监听启动 mode={pl.get('mode')}{extra}", "info")
        elif ev == "stopped":
            self.lbl_state.config(text="● STOPPED", foreground="gray")
            self._log("监听已停止", "info")
            self.engine = None
            self.btn_start.config(state="normal")
            self.btn_stop.config(state="disabled")
        elif ev == "interrupt_hint":
            self._log(f"(LISTENING 中再触发，按方案交给打断模块) {pl['word']}={pl['score']:.2f}", "warn")

    def _refresh_preview(self):
        self.after(100, self._refresh_preview)
        if not HAVE_PREVIEW or self.engine is None or self.engine.visual is None:
            return
        frame = self.engine.visual.preview()
        if frame is None:
            return
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 限制显示尺寸
            h, w = rgb.shape[:2]
            target_w = 260
            if w > target_w:
                scale = target_w / w
                rgb = cv2.resize(rgb, (target_w, int(h * scale)))
            img = Image.fromarray(rgb)
            self._preview_photo = ImageTk.PhotoImage(img)
            self.lbl_preview.config(image=self._preview_photo, text="")
        except Exception as e:
            self.lbl_preview.config(text=f"preview error: {e}")

    # ------------------------ 日志 & 播放 ------------------------ #
    def _log(self, msg, tag="info"):
        ts = time.strftime("%H:%M:%S")
        self.txt_log.insert("end", f"[{ts}] {msg}\n", tag)
        self.txt_log.see("end")

    def _reset_wake_count(self):
        self.wake_count = 0
        self.lbl_wake_count.config(text="0")
        if self.engine is not None:
            self.engine.wake_count = 0
        self._log("唤醒计数已清零", "info")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    app = WakeGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
