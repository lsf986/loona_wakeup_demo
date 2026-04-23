"""
Loona 唤醒模块 Demo
====================

对应《唤醒模块实现方案-v1.0.md》的最小可运行骨架，支持三种触发模式：
  - kws        : 预训练唤醒词（openWakeWord）+ 仲裁器
  - vad        : 无唤醒词（纯音频意图）
  - multimodal : 方案 §5.2 路径 B —— 注视+唇动+DOA+近场四硬门限全命中才触发
                 + 增强音频前端（高通、浊音、频谱重心、GCC-PHAT DOA）

运行:
    python wake_demo.py                     # 默认 multimodal (若摄像头可用)
    python wake_demo.py --mode vad
    python wake_demo.py --mode kws
    python wake_demo.py --list-devices
"""

from __future__ import annotations

import argparse
import collections
import os
import queue
import sys
import threading
import time
import wave
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from perception_audio import AudioFrontend, AudioFeatures
from perception_visual import VisualFrontend, VisualGates, HAVE_CV2

SAMPLE_RATE = 16000
VAD_FRAME_MS = 20
VAD_FRAME_LEN = SAMPLE_RATE * VAD_FRAME_MS // 1000  # 320 samples
KWS_CHUNK_LEN = 1280  # openWakeWord 要求 80 ms (=1280 samples @16k)


# ASR 后处理：过滤空白/无意义识别结果（多为环境噪声或同事远场语音被 whisper
# 用常见口头禅/语气词"幻觉"出来的短文本）
_MEANINGLESS_CHARS = set("嗯啊哦呃唉哈呀哎嘿噢哇呵咳喂哼唔呣阿噫唷唉~哦嗯啊")
_PUNCT_CHARS = set(" \t\n\r，。！？、.,!?~～·…—-“”\"'`()（）[]【】《》:;：；")


def _asr_is_meaningless(text: str) -> tuple[bool, str]:
    """判断 ASR 文本是否应被判为"无效唤醒"。
    返回 (是否无效, 原因描述)。
    """
    if not text:
        return True, "未识别出文字"
    stripped = "".join(ch for ch in text if ch not in _PUNCT_CHARS)
    if not stripped:
        return True, "仅含标点/空白"
    # 全是拟声/语气词（"嗯"、"啊"、"哦哦哦" 等）
    if all(ch in _MEANINGLESS_CHARS for ch in stripped):
        return True, f"仅含语气词: {stripped[:8]}"
    # 长度过短（1 字符中文一般是识别噪声，如单独"嗯"/"啊"/"的"）
    if len(stripped) <= 1:
        return True, f"过短无意义: {stripped!r}"
    return False, ""


def _lazy_oww():
    try:
        from openwakeword.model import Model as OWWModel
        return OWWModel
    except ImportError:
        print("[FATAL] openwakeword 未安装，请先: pip install -r requirements.txt", file=sys.stderr)
        raise


class EnergyVAD:
    """轻量能量 + 过零率 VAD（替代 webrtcvad，避免 C 扩展依赖）。

    原理：自适应跟踪噪声能量底，当前帧 RMS 超过噪声底 × 阈值倍数且过零率在人声区间
    内，则判为语音。对桌面办公场景的稳态噪声（风扇、空调）自适应效果较好；对瞬态
    噪声（键盘敲击）会偶发误报，此时 KWS 得分一般仍低，不会导致唤醒——和仲裁器联合
    使用即可（方案 §3）。
    """

    def __init__(self, energy_ratio_db: float = 8.0):
        self.energy_ratio_db = energy_ratio_db
        self.noise_floor = 50.0  # int16 RMS
        self.alpha = 0.02

    def is_speech(self, frame: np.ndarray) -> bool:
        f = frame.astype(np.float32)
        rms = float(np.sqrt(np.mean(f * f) + 1e-9))
        # 过零率（归一化到 [0,1]）
        zc = float(np.mean(np.abs(np.diff(np.signbit(f).astype(np.int8)))))
        # 阈值：高于噪声底 N dB
        thresh = self.noise_floor * (10 ** (self.energy_ratio_db / 20.0))
        speech = (rms > thresh) and (0.01 < zc < 0.35)
        if not speech:
            # 只在非语音时更新噪声底（保护跟踪器不被语音拉偏）
            self.noise_floor = (1 - self.alpha) * self.noise_floor + self.alpha * rms
        return speech
RING_SECONDS = 8          # 环形缓冲：8s 历史，避免长句头被截
REPLAY_SECONDS = 3.5      # 唪醒命中时已累积的历史时长（向前看）
POSTROLL_SECONDS = 2.0    # 唪醒后继续录 2s 才送 ASR（向后看，保全句尾）
LISTEN_TIMEOUT_S = 8.0


# ------------------------------ Ring Buffer ------------------------------ #
class RingBuffer:
    """对应方案 §3.2：2s 环形缓冲，用于唤醒命中后回灌首句。"""

    def __init__(self, seconds: int, sr: int = SAMPLE_RATE):
        self.cap = seconds * sr
        self.buf = np.zeros(self.cap, dtype=np.int16)
        self.wpos = 0
        self.filled = 0
        self.lock = threading.Lock()

    def write(self, data: np.ndarray) -> None:
        with self.lock:
            n = len(data)
            if n >= self.cap:
                self.buf[:] = data[-self.cap:]
                self.wpos = 0
                self.filled = self.cap
                return
            end = self.wpos + n
            if end <= self.cap:
                self.buf[self.wpos:end] = data
            else:
                first = self.cap - self.wpos
                self.buf[self.wpos:] = data[:first]
                self.buf[:n - first] = data[first:]
            self.wpos = end % self.cap
            self.filled = min(self.cap, self.filled + n)

    def read_last(self, seconds: float) -> np.ndarray:
        with self.lock:
            n = min(int(seconds * SAMPLE_RATE), self.filled)
            if n == 0:
                return np.zeros(0, dtype=np.int16)
            start = (self.wpos - n) % self.cap
            if start + n <= self.cap:
                return self.buf[start:start + n].copy()
            tail = self.cap - start
            return np.concatenate([self.buf[start:], self.buf[:n - tail]])


# ------------------------------ Noise / SNR ------------------------------ #
class NoiseTracker:
    """用非语音帧跟踪噪声底，估计当前 SNR，为动态阈值提供依据。"""

    def __init__(self):
        self.noise_rms = 50.0  # 初始噪声底（int16 RMS）
        self.alpha_update = 0.02  # 噪声跟随速度

    @staticmethod
    def rms(frame: np.ndarray) -> float:
        f = frame.astype(np.float32)
        return float(np.sqrt(np.mean(f * f) + 1e-9))

    def update(self, frame_rms: float, is_speech: bool) -> None:
        if not is_speech:
            self.noise_rms = (1 - self.alpha_update) * self.noise_rms + self.alpha_update * frame_rms

    def snr_db(self, frame_rms: float) -> float:
        return 20.0 * np.log10((frame_rms + 1e-6) / (self.noise_rms + 1e-6))


# ------------------------------ Wake Arbiter ------------------------------ #
@dataclass
class WakeDecision:
    fire: bool
    threshold: float
    reason: str


class WakeArbiter:
    """对应方案 §5 仲裁器（简化版：仅唤醒词路径 A）。

    - 动态阈值随 SNR / TTS 状态 / 冷却加成调整
    - 硬否决：极端静默下的高分视为异常；非 VAD 语音帧下的边缘分数丢弃
    - 连续命中进入冷却，提高阈值，防止"乱唤醒"
    """

    def __init__(self, base_thresh: float = 0.5):
        self.base_thresh = base_thresh
        self.hit_times: collections.deque[float] = collections.deque(maxlen=8)
        self.cooldown_bonus = 0.0
        self.cooldown_until = 0.0
        self.tts_playing = False  # 真实系统应由 TTS 播放线程回调置位

    def dynamic_threshold(self, snr_db: float) -> float:
        if snr_db >= 20:
            t = self.base_thresh
        elif snr_db >= 10:
            t = self.base_thresh + 0.15
        else:
            t = self.base_thresh + 0.30
        if self.tts_playing:
            t += 0.10
        t += self.cooldown_bonus
        return float(min(t, 0.95))

    def decide(self, score: float, snr_db: float, vad_active: bool) -> WakeDecision:
        now = time.time()
        if now > self.cooldown_until and self.cooldown_bonus > 0:
            self.cooldown_bonus = 0.0

        thresh = self.dynamic_threshold(snr_db)

        # 硬否决：整体很安静（噪声底本身低 + 帧能量小）却命中高分 → 异常
        if snr_db < -3 and score >= thresh:
            return WakeDecision(False, thresh, "veto:anomaly-silent")

        # VAD 未激活时要求显著更高的分数，抑制瞬态/爆音误触
        if not vad_active and score < thresh + 0.10:
            return WakeDecision(False, thresh, "no-vad")

        if score >= thresh:
            return WakeDecision(True, thresh, "ok")
        return WakeDecision(False, thresh, "below")

    def register_hit(self, t: float) -> None:
        self.hit_times.append(t)
        recent = [x for x in self.hit_times if t - x <= 5.0]
        if len(recent) >= 2:
            self.cooldown_bonus = 0.15
            self.cooldown_until = t + 60.0
            print(f"[ARBITER] 进入冷却，阈值 +0.15，持续 60s")


# ------------------------------ Demo Runner ------------------------------ #
class WakeDemo:
    def __init__(self, args):
        self.args = args
        self.mode = getattr(args, "mode", "multimodal")  # "vad" | "kws" | "multimodal"
        self.vad = EnergyVAD()
        self.kws = None
        if self.mode == "kws":
            print(f"[INIT] KWS 模式，加载模型: {args.wakewords}")
            OWWModel = _lazy_oww()
            self.kws = OWWModel(wakeword_models=list(args.wakewords), inference_framework="onnx")
        elif self.mode == "multimodal":
            print(f"[INIT] MULTIMODAL 模式: 注视+唇动+DOA+近场四硬门限 + 增强音频前端")
        else:
            print(f"[INIT] VAD 模式（无唤醒词）: min_speech={getattr(args, 'min_speech_ms', 400)}ms, "
                  f"min_snr={getattr(args, 'min_snr_db', 6.0):.1f}dB")

        self.ring = RingBuffer(RING_SECONDS)
        self.noise = NoiseTracker()
        self.arbiter = WakeArbiter(base_thresh=args.base_thresh)
        self.audio_q: queue.Queue = queue.Queue(maxsize=100)
        self.state = "IDLE"
        self.listen_start = 0.0
        self.kws_accum = np.zeros(0, dtype=np.int16)

        # VAD/multimodal 模式共用的语音计时
        self.speech_run_ms = 0.0
        self.silence_run_ms = 0.0
        self.last_fire = 0.0
        # 最近 1000ms 的 (voiced, lip_moving) 对，用于唇-声同步门
        self._av_window = collections.deque(maxlen=50)  # 50 * 20ms = 1000ms
        # 最近 10 帧（200ms）DOA 是否在锥内，用于做时间迟滞，抗 GCC-PHAT 抖动
        self._doa_window = collections.deque(maxlen=10)
        # 持续命中计数：要求硬门限连续 N 帧全绿才触发，抑制瞬时误报
        self._sustain_pass_frames = 0

        # 增强音频前端（多模态模式主力，其他模式也可用于近场诊断）
        self.audio_fe = AudioFrontend(
            near_field_rms=float(getattr(args, "near_field_rms", 900.0)),
            voicing_min=0.28,
        )

        # 视觉前端（按需启动）
        self.visual = None
        self.use_visual = (self.mode == "multimodal") and not getattr(args, "no_camera", False)
        if self.use_visual:
            if not HAVE_CV2:
                print("[WARN] 未找到 opencv-python，视觉门限将被跳过")
            else:
                self.visual = VisualFrontend(
                    camera_index=int(getattr(args, "camera_index", 0)),
                )
        # 音频输入通道数（若 device 支持 ≥2，用于 GCC-PHAT DOA）
        self.audio_channels = int(getattr(args, "audio_channels", 1))
        self.doa_cone_deg = float(getattr(args, "doa_cone_deg", 30.0))

        # GUI / 外部订阅
        self.on_event = None
        self.stop_flag = threading.Event()

        # ASR（按需懒加载；启动时会在后台线程预加载一次）
        self.wake_count = 0
        self._asr_model = None
        self._asr_backend = None
        self._asr_error = None         # 最近一次加载失败的可读原因
        self._asr_loaded_event = threading.Event()  # 预加载完成（无论成功失败）
        self._asr_lock = threading.Lock()
        self._asr_pool = []  # 简单的后台线程列表（用完丢弃）

    def _emit(self, event_type: str, **payload):
        cb = self.on_event
        if cb is not None:
            try:
                cb(event_type, payload)
            except Exception as e:
                print(f"[EVENT-CB-ERR] {e}", file=sys.stderr)

    # sounddevice 回调（在独立线程）
    def _audio_cb(self, indata, frames, time_info, status):
        if status:
            print(f"[AUDIO] {status}", file=sys.stderr)
        # indata: (frames, channels)
        if indata.shape[1] >= 2:
            pair = (indata[:, 0].copy(), indata[:, 1].copy())
        else:
            pair = (indata[:, 0].copy(), None)
        try:
            self.audio_q.put_nowait(pair)
        except queue.Full:
            # 丢最老的一帧，保持实时性，避免阻塞音频回调
            try:
                self.audio_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.audio_q.put_nowait(pair)
            except queue.Full:
                pass

    def _get_asr(self):
        """懒加载 ASR 模型。
        优先用 sherpa-onnx + SenseVoiceSmall（中文识别准确率显著优于 whisper tiny/base），
        如未安装或模型文件缺失则回退到 faster-whisper。
        返回 (model, backend_name) 二元组，backend_name ∈ {'sensevoice','whisper',None}。
        加载失败时会把可读的错误原因写到 self._asr_error。
        """
        with self._asr_lock:
            if self._asr_model is not None:
                return self._asr_model, self._asr_backend
            reasons: list[str] = []
            # 首选 SenseVoice
            sv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "models", "sherpa-onnx-sense-voice")
            sv_model = os.path.join(sv_dir, "model.int8.onnx")
            sv_tokens = os.path.join(sv_dir, "tokens.txt")
            sv_files_ok = os.path.isfile(sv_model) and os.path.isfile(sv_tokens)
            if not sv_files_ok:
                reasons.append(
                    f"SenseVoice 模型文件缺失: 需要 {sv_model} 与 {sv_tokens}"
                )
            else:
                try:
                    import sherpa_onnx  # type: ignore
                except ImportError:
                    reasons.append(
                        "未安装 sherpa-onnx（执行: pip install sherpa-onnx）"
                    )
                else:
                    try:
                        print("[ASR] 加载 sherpa-onnx SenseVoiceSmall ...")
                        self._asr_model = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                            model=sv_model,
                            tokens=sv_tokens,
                            num_threads=2,
                            use_itn=True,
                            language="zh",
                        )
                        self._asr_backend = "sensevoice"
                        self._asr_error = None
                        print("[ASR] SenseVoice 加载完成")
                        return self._asr_model, self._asr_backend
                    except Exception as e:
                        reasons.append(f"SenseVoice 加载失败: {e}")
                        print(f"[ASR] SenseVoice 加载失败，尝试回退 whisper: {e}",
                              file=sys.stderr)
            # 回退：faster-whisper
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                reasons.append(
                    "未安装 faster-whisper（执行: pip install faster-whisper）"
                )
                print("[ASR] faster-whisper 未安装，跳过转写", file=sys.stderr)
                self._asr_error = "；".join(reasons)
                return None, None
            model_size = getattr(self.args, "asr_model", "base")
            try:
                print(f"[ASR] 加载 faster-whisper({model_size}) ...")
                self._asr_model = WhisperModel(
                    model_size, device="cpu", compute_type="int8"
                )
                self._asr_backend = "whisper"
                self._asr_error = None
                print("[ASR] Whisper 加载完成")
            except Exception as e:
                reasons.append(f"Whisper 加载失败: {e}")
                print(f"[ASR] 加载失败: {e}", file=sys.stderr)
                self._asr_model = None
                self._asr_backend = None
                self._asr_error = "；".join(reasons)
            return self._asr_model, self._asr_backend

    def _preload_asr_async(self):
        """在后台线程预加载 ASR，避免首个唤醒事件阻塞等待模型加载，
        同时让不可用错误尽早暴露在日志里。"""
        def _job():
            try:
                model, backend = self._get_asr()
                if model is None:
                    reason = self._asr_error or "未知原因"
                    print(f"[ASR] 预加载失败: {reason}", file=sys.stderr)
                    self._emit("asr_status", ok=False, reason=reason)
                else:
                    self._emit("asr_status", ok=True, backend=backend)
            finally:
                self._asr_loaded_event.set()
        t = threading.Thread(target=_job, daemon=True, name="asr-preload")
        t.start()

    def _transcribe_async(self, pcm_i16: np.ndarray, name: str, score: float,
                          wake_count: int | None = None):
        """在后台线程做语音转文字，完成后 emit 'asr' 事件。
        优先用 SenseVoice，失败时回退 whisper；
        若结果为空或无实际意义，调用 _reject_wake 回滚本次唤醒并在事件里带 rejected=True。
        """
        wake_count_snapshot = self.wake_count if wake_count is None else wake_count

        def _job():
            pair = self._get_asr()
            model, backend = pair if pair else (None, None)
            if model is None:
                self._reject_wake(wake_count_snapshot)
                reason_detail = self._asr_error or "ASR 模型不可用"
                self._emit("asr", word=name, score=score, text="",
                           count=wake_count_snapshot,
                           rejected=True,
                           reject_reason=f"ASR 不可用: {reason_detail}",
                           error="model_unavailable")
                return
            try:
                audio_f = pcm_i16.astype(np.float32) / 32768.0
                # 峰值归一化，提升低音量下的识别率
                peak = float(np.max(np.abs(audio_f))) if audio_f.size else 0.0
                if 0 < peak < 0.5:
                    audio_f = audio_f * min(3.0, 0.8 / peak)

                if backend == "sensevoice":
                    s = model.create_stream()
                    s.accept_waveform(SAMPLE_RATE, audio_f)
                    model.decode_stream(s)
                    text = (s.result.text or "").strip()
                    lang_detected = getattr(s.result, "lang", None)
                else:  # whisper
                    segments, info = model.transcribe(
                        audio_f,
                        language="zh",
                        beam_size=5,
                        best_of=5,
                        temperature=[0.0, 0.2, 0.4],
                        vad_filter=False,
                        condition_on_previous_text=False,
                        no_speech_threshold=0.45,
                        initial_prompt="以下是用户对智能助手的中文指令。",
                    )
                    text = "".join(seg.text for seg in segments).strip()
                    lang_detected = getattr(info, "language", None)

                print(f"[ASR/{backend}] 识别结果: {text!r}")
                rejected, reject_reason = _asr_is_meaningless(text)
                if rejected:
                    self._reject_wake(wake_count_snapshot)
                    print(f"[ASR] 判定为无效唤醒: {reject_reason}")
                self._emit("asr", word=name, score=score, text=text,
                           count=wake_count_snapshot, backend=backend,
                           rejected=rejected, reject_reason=reject_reason,
                           language=lang_detected)
            except Exception as e:
                print(f"[ASR] 转写失败: {e}", file=sys.stderr)
                self._reject_wake(wake_count_snapshot)
                self._emit("asr", word=name, score=score, text="",
                           count=wake_count_snapshot,
                           rejected=True, reject_reason=f"转写异常: {e}",
                           error=str(e))

        t = threading.Thread(target=_job, daemon=True)
        t.start()
        self._asr_pool.append(t)

    def _reject_wake(self, cnt: int):
        """回退一次唤醒计数（仅当仍为最新那次时才回退，避免并发冲突）。"""
        if self.wake_count == cnt and cnt > 0:
            self.wake_count = cnt - 1
        # 立即回到 IDLE，让下一轮可以正常识别
        if self.state == "LISTENING":
            self.state = "IDLE"
            self._emit("state", state=self.state, reason="asr_rejected")

    def _on_wake(self, name: str, score: float):
        """唪醒命中。立即 emit wake 事件；ASR 延迟 POSTROLL 秒再启动，
        等到环形缓冲把唪醒之后的语音一起录完，保证句尾完整。"""
        self.wake_count += 1
        print(f">>> WAKE #{self.wake_count}! word={name} score={score:.2f} "
              f"| 等待 {POSTROLL_SECONDS:.1f}s postroll 后送 ASR")
        self._emit("wake", word=name, score=score,
                   duration=REPLAY_SECONDS + POSTROLL_SECONDS,
                   count=self.wake_count)

        wake_count_snapshot = self.wake_count

        def _delayed():
            time.sleep(POSTROLL_SECONDS)
            history = self.ring.read_last(REPLAY_SECONDS + POSTROLL_SECONDS)
            dur = len(history) / SAMPLE_RATE
            print(f"[ASR ] 取 {dur:.2f}s 音频送转写")
            self._transcribe_async(history.copy(), name, score,
                                   wake_count=wake_count_snapshot)

        threading.Thread(target=_delayed, daemon=True).start()

    def run(self):
        try:
            # 后台预加载 ASR：避免首次唤醒阻塞等待模型，也让不可用错误提前暴露
            self._preload_asr_async()
            # 尝试按配置打开多通道；若失败退回单通道
            ch = max(1, self.audio_channels)
            try:
                stream = sd.InputStream(
                    samplerate=SAMPLE_RATE, channels=ch, dtype="int16",
                    blocksize=VAD_FRAME_LEN, device=self.args.device,
                    latency="high",
                    callback=self._audio_cb,
                )
                stream.start()
            except Exception as e:
                if ch > 1:
                    print(f"[WARN] {ch} 通道打开失败 ({e})，回退单通道，DOA 门限将被旁路")
                    ch = 1
                    stream = sd.InputStream(
                        samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                        blocksize=VAD_FRAME_LEN, device=self.args.device,
                        latency="high",
                        callback=self._audio_cb,
                    )
                    stream.start()
                else:
                    raise
            self.audio_channels = ch

            # 启动视觉前端
            if self.visual is not None:
                ok = self.visual.start()
                if not ok:
                    print("[WARN] 摄像头开启失败，视觉门限将被跳过")
                    self.visual = None

            try:
                if self.mode == "kws":
                    hint = f"请说唤醒词: {' / '.join(self.args.wakewords)}"
                elif self.mode == "multimodal":
                    hint = ("请正对摄像头说话（需 注视+唇动+近场 同时命中，"
                            f"{'含 DOA ' if self.audio_channels>=2 else ''}仲裁）")
                else:
                    hint = "请直接对着麦克风说话（无需唤醒词）"
                print(f"[RUN ] 正在持续监听... {hint}")
                print("[RUN ] Ctrl+C 退出")
                self._emit("started", mode=self.mode,
                           audio_channels=self.audio_channels,
                           visual_on=(self.visual is not None),
                           wakewords=list(getattr(self.args, "wakewords", []) or []))
                self._loop()
            finally:
                try:
                    stream.stop(); stream.close()
                except Exception:
                    pass
        except KeyboardInterrupt:
            print("\n[EXIT] 用户终止")
        finally:
            if self.visual is not None:
                self.visual.stop()
            self._emit("stopped")

    def _loop(self):
        while not self.stop_flag.is_set():
            try:
                item = self.audio_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if not isinstance(item, tuple):
                continue
            frame, ref = item
            if len(frame) != VAD_FRAME_LEN:
                continue
            self.ring.write(frame)

            # 轻量 VAD（做状态保留/超时）
            is_speech = self.vad.is_speech(frame)

            # 噪声底/SNR
            rms = NoiseTracker.rms(frame)
            self.noise.update(rms, is_speech)
            snr = self.noise.snr_db(rms)

            # 增强音频特征（HPF、浊音、频谱重心、DOA、近场）
            feats = self.audio_fe.process(frame, ref)

            # LISTENING 超时
            if self.state == "LISTENING" and time.time() - self.listen_start > LISTEN_TIMEOUT_S:
                print("<<< LISTENING 超时 8s，回到 IDLE")
                self.state = "IDLE"
                self._emit("state", state=self.state, reason="timeout")

            if self.mode == "kws":
                self._process_kws(frame, is_speech, snr, feats)
            elif self.mode == "multimodal":
                self._process_multimodal(is_speech, snr, feats)
            else:
                self._process_vad(is_speech, snr)

    # -------- 模式 A：唤醒词路径 -------- #
    def _process_kws(self, frame, is_speech, snr, feats: AudioFeatures):
        self.kws_accum = np.concatenate([self.kws_accum, frame])
        while len(self.kws_accum) >= KWS_CHUNK_LEN:
            chunk = self.kws_accum[:KWS_CHUNK_LEN]
            self.kws_accum = self.kws_accum[KWS_CHUNK_LEN:]
            preds = self.kws.predict(chunk)
            best_name, best_score = max(preds.items(), key=lambda x: x[1])
            vad_flag = is_speech or self.args.no_vad_gate
            decision = self.arbiter.decide(best_score, snr, vad_flag)

            self._emit(
                "tick", word=best_name, score=float(best_score),
                threshold=decision.threshold, snr=snr, vad=is_speech,
                reason=decision.reason, state=self.state,
                rms=feats.rms, centroid=feats.spec_centroid,
                near_field=feats.near_field_ok,
            )
            if best_score >= 0.3 or self.args.verbose:
                print(f"[KWS ] {best_name}={best_score:.2f} "
                      f"thresh={decision.threshold:.2f} snr={snr:+.1f}dB "
                      f"vad={is_speech} -> {decision.reason}")
            if not decision.fire:
                continue
            if self.state == "IDLE":
                self._fire_wake(best_name, float(best_score))
            else:
                print(f"[INFO] LISTENING 中再次触发 ({best_name}={best_score:.2f})")
                self._emit("interrupt_hint", word=best_name, score=float(best_score))

    # -------- 模式 C：多模态硬门限路径（方案 §5.2 路径 B 正式实现） -------- #
    def _process_multimodal(self, is_speech, snr, feats: AudioFeatures):
        """注视 + 唇动 + DOA + 近场 四硬门限全命中才触发。

        近场有两条口径（只要一条满足即可，以应对 demo 1m 使用场景）：
          - 视觉近场：人脸占画面比 ≥ 阈值
          - 音频近场：HPF 后 RMS ≥ 校准阈值 且 频谱重心在人声区
        DOA 可选：单通道时跳过（真实产品必须启用）
        """
        # 1) 音频"意图证据"必须首先在场：浊音 + SNR
        # 浊音 hangover：每次出现浊音后保留 N 帧"近期浊音"状态，抗短暂停顿
        # 灵敏度调整：内部门限更宽松（SNR 1.0 而不是 3.0），hangover 加长到 400ms，
        # 长静音清零门限放宽到 900ms，避免自然说话停顿把累计置零。
        if feats.is_voiced and snr >= 1.0:
            self.speech_run_ms += VAD_FRAME_MS
            self.silence_run_ms = 0.0
            self._voiced_hangover = 20  # 20*20ms = 400ms 尾拖
        else:
            if getattr(self, "_voiced_hangover", 0) > 0:
                # hangover 内仍推进 speech_run（按 0.75 速率，尽量保留语流）
                self._voiced_hangover -= 1
                self.speech_run_ms += VAD_FRAME_MS * 0.75
                self.silence_run_ms = 0.0
            else:
                self.silence_run_ms += VAD_FRAME_MS
                if self.silence_run_ms >= 900:  # 长静音才清零
                    self.speech_run_ms = 0.0

        min_speech_ms = float(getattr(self.args, "min_speech_ms", 300))
        min_snr_db = float(getattr(self.args, "min_snr_db", 6.0))
        pseudo_score = min(1.0, self.speech_run_ms / max(min_speech_ms, 1.0))

        # 2) 视觉门限
        if self.visual is not None:
            vg = self.visual.snapshot()
        else:
            vg = VisualGates(available=False, face_present=False, gaze_aligned=False,
                              near_field=False, lip_moving=False)

        # 3) DOA 门限（有第二通道时生效，且做 200ms 迟滞投票，抗相位抖动）
        if feats.doa_deg is not None:
            self._doa_window.append(abs(feats.doa_deg) <= self.doa_cone_deg)
            # 最近 10 帧中至少 6 帧在锥内才算 pass
            if len(self._doa_window) >= 6:
                doa_ok = sum(self._doa_window) >= 6
            else:
                doa_ok = False
        else:
            doa_ok = True  # 单通道时依赖视觉 gaze（已代表方向）

        # 4) 近场门限（视觉或音频至少一条通过）
        near_ok = feats.near_field_ok or (vg.available and vg.near_field)

        # 5) 唇-声同步门：最近 1000ms 内
        #    - “嘴动且发声”的共现帧占音频语音帧 ≥ 40%
        #    - “发声但嘴没动”的反证帧数 ≤ 1.5 × 共现帧数  ← 抗同事远场语音的核心
        # 灵敏度调整：阈值从 50% 放到 40%，反证系数从 1.0 放到 1.5，
        # voiced/lip 最小出现次数从 6 放到 4，历史窗口从 300ms 放到 240ms。
        lip_now = bool(vg.available and vg.lip_moving)
        voiced_now = bool(feats.is_voiced)
        self._av_window.append((voiced_now, lip_now))
        if len(self._av_window) >= 12:  # 至少 240ms 历史
            coincide = sum(1 for v, l in self._av_window if v and l)
            voiced_cnt = sum(1 for v, _ in self._av_window if v)
            lip_cnt = sum(1 for _, l in self._av_window if l)
            voiced_only = sum(1 for v, l in self._av_window if v and not l)  # 有声无唇 ≈ 同事语音
            av_sync_ok = (
                voiced_cnt >= 4 and lip_cnt >= 4
                and coincide >= max(3, int(0.40 * voiced_cnt))
                and voiced_only <= int(1.5 * coincide) + 1  # 反证帧放宽，允许短暂嘴型遮挡
            )
        else:
            av_sync_ok = False

        # 硬门限集合
        gates = {
            "voiced": feats.is_voiced,
            "snr": snr >= min_snr_db,
            "speech_run": self.speech_run_ms >= min_speech_ms,
            "near_field": near_ok,
            "doa": doa_ok,
            # 视觉硬门限（若不可用则 False，严格模式强制要求视觉）
            "face": vg.available and vg.face_present,
            "gaze": vg.available and vg.gaze_aligned,
            "lip": vg.available and vg.lip_moving,
            # 唇-声同步
            "av_sync": av_sync_ok,
        }

        # 如果视觉不可用，允许降级但记录日志（提示用户：不满足需求的严格多模态）
        strict = not bool(getattr(self.args, "no_strict_multimodal", False))
        if not vg.available and not strict:
            # 非严格：视觉门限自动放行（此时近似退化为 VAD 模式+强噪声抑制）
            gates["face"] = gates["gaze"] = gates["lip"] = gates["av_sync"] = True

        all_pass = all(gates.values())

        # 持续命中门限：要求硬门限连续 N 帧保持全绿（抗瞬时误报）
        # 灵敏度调整：从 200ms(10 帧) 下调到 100ms(5 帧)，并允许少量抖动（缓慢衰减而非清零）
        SUSTAIN_FRAMES = 5  # 5 * 20ms = 100ms
        if all_pass:
            self._sustain_pass_frames += 1
        else:
            # 不直接清零：允许单帧抖动（摄像头帧间人脸丢失、DOA 抖）不重置累计
            self._sustain_pass_frames = max(0, self._sustain_pass_frames - 2)
        sustained = self._sustain_pass_frames >= SUSTAIN_FRAMES

        # 防抖 + 冷却 + TTS 抑制
        now = time.time()
        if all_pass and not sustained:
            reason = f"hold {self._sustain_pass_frames}/{SUSTAIN_FRAMES}"
        elif sustained and (now - self.last_fire) < 1.5:
            sustained = False
            reason = "veto:debounce"
        elif self.arbiter.tts_playing and sustained:
            # TTS 播报中 + SNR 仍处理为疑似回环，抬高要求
            if snr < min_snr_db + 3:
                sustained = False
                reason = "veto:tts-playing"
            else:
                reason = "ok"
        else:
            # 找第一个未通过的门
            failed = [k for k, v in gates.items() if not v]
            reason = "ok" if sustained else f"veto:{','.join(failed)}"

        # 诊断：当有浊音时每 1.5s 打印一次门限状态，帮助定位哪个门在挡
        if is_speech:
            last_diag = getattr(self, "_last_diag_t", 0.0)
            if now - last_diag > 1.5:
                self._last_diag_t = now
                gate_str = " ".join(f"{k}={'Y' if v else 'N'}" for k, v in gates.items())
                print(f"[GATES] rms={feats.rms:6.0f} snr={snr:+5.1f} "
                      f"cent={feats.spec_centroid:5.0f} pitch={(feats.pitch_hz or 0):5.1f} "
                      f"face_area={vg.face_area_ratio:.3f} lip_std={vg.lip_motion_std:.2f} | "
                      f"{gate_str} -> {reason}")

        # UI tick
        self._emit(
            "tick", word="(multimodal)", score=pseudo_score,
            threshold=1.0, snr=snr, vad=is_speech, reason=reason, state=self.state,
            rms=feats.rms, centroid=feats.spec_centroid,
            pitch=feats.pitch_hz, doa=feats.doa_deg,
            near_field=feats.near_field_ok,
            visual=dict(available=vg.available, face=vg.face_present,
                        gaze=vg.gaze_aligned, near=vg.near_field,
                        lip=vg.lip_moving,
                        face_area=vg.face_area_ratio,
                        offset=vg.face_center_offset,
                        lip_std=vg.lip_motion_std, fps=vg.fps),
            gates=gates,
        )

        if not all_pass:
            return
        if not sustained:
            return
        if self.state != "IDLE":
            self._emit("interrupt_hint", word="(multimodal)", score=pseudo_score)
            return
        self.last_fire = now
        self.speech_run_ms = 0.0
        self._sustain_pass_frames = 0
        self._fire_wake("(multimodal)", pseudo_score)

    # -------- 模式 B：无唤醒词 VAD 路径 -------- #
    def _process_vad(self, is_speech, snr):
        """方案 §8.1 路径 B 的简化版：以"连续语音活动时长 + SNR"为触发证据。

        真实产品中，这个打分应融合：注视方向、唇动、DOA、距离、上下文状态。
        Demo 无摄像头，故以 VAD 持续时长 + SNR 作为近似，并保留仲裁器的 TTS 抑制 / 冷却 /
        硬否决（极端静默）。
        """
        min_speech_ms = float(getattr(self.args, "min_speech_ms", 400))
        min_snr_db = float(getattr(self.args, "min_snr_db", 6.0))
        hangover_ms = 300.0  # 连续静音多久才重置计数
        debounce_s = 1.5     # 每次触发后至少间隔 1.5s 才能再次触发

        if is_speech:
            self.speech_run_ms += VAD_FRAME_MS
            self.silence_run_ms = 0.0
        else:
            self.silence_run_ms += VAD_FRAME_MS
            if self.silence_run_ms >= hangover_ms:
                self.speech_run_ms = 0.0

        # 归一化 "score" 用于 UI 显示：当前累计语音时长 / 触发门限
        pseudo_score = min(1.0, self.speech_run_ms / max(min_speech_ms, 1.0))

        # 仲裁：把 pseudo_score 当成分数跑仲裁器，复用 TTS 抑制/冷却/动态阈值
        # 动态阈值对无唤醒词路径语义上是"证据强度门限"
        vad_flag = is_speech
        decision = self.arbiter.decide(pseudo_score, snr, vad_flag)

        # 额外的 VAD 路径硬门限（方案 §5.2 / §5.3 的近似）
        reason = decision.reason
        fire = decision.fire
        now = time.time()
        if fire and snr < min_snr_db:
            fire = False
            reason = f"veto:snr<{min_snr_db:.0f}dB"
        if fire and (now - self.last_fire) < debounce_s:
            fire = False
            reason = "veto:debounce"
        if fire and self.speech_run_ms < min_speech_ms:
            fire = False
            reason = "below"

        self._emit(
            "tick", word="(vad-intent)", score=pseudo_score,
            threshold=decision.threshold, snr=snr, vad=is_speech,
            reason=reason, state=self.state,
        )

        if not fire:
            return
        if self.state != "IDLE":
            self._emit("interrupt_hint", word="(vad-intent)", score=pseudo_score)
            return

        self.last_fire = now
        self.speech_run_ms = 0.0
        self._fire_wake("(vad-intent)", pseudo_score)

    def _fire_wake(self, name: str, score: float):
        self.state = "LISTENING"
        self.listen_start = time.time()
        self.arbiter.register_hit(self.listen_start)
        self._emit("state", state=self.state, reason="wake")
        self._on_wake(name, score)


# ------------------------------ CLI ------------------------------ #
def parse_args():
    p = argparse.ArgumentParser(description="Loona 唤醒模块 Demo")
    p.add_argument("--device", type=int, default=None, help="sounddevice 输入设备索引")
    p.add_argument("--list-devices", action="store_true", help="列出音频设备并退出")
    p.add_argument("--mode", choices=["vad", "kws", "multimodal"], default="multimodal",
                   help="触发模式: multimodal=四硬门限(默认) / vad=无唤醒词 / kws=唤醒词")
    p.add_argument(
        "--wakewords", nargs="+", default=["hey_jarvis"],
        help="[kws 模式] openWakeWord 预训练唤醒词",
    )
    p.add_argument("--base-thresh", type=float, default=0.5,
                   help="[kws] KWS 基础阈值 / [vad] 证据强度基础门限")
    p.add_argument("--min-speech-ms", type=float, default=300.0,
                   help="[vad/multimodal] 触发所需的最小连续语音时长(ms)")
    p.add_argument("--min-snr-db", type=float, default=6.0,
                   help="[vad/multimodal] 触发所需的最小 SNR(dB)")
    p.add_argument("--near-field-rms", type=float, default=500.0,
                   help="[multimodal] 音频近场 RMS 绝对门限（同事远场一般 <300）")
    p.add_argument("--audio-channels", type=int, default=2,
                   help="[multimodal] 音频通道数，≥2 时启用 GCC-PHAT DOA")
    p.add_argument("--doa-cone-deg", type=float, default=30.0,
                   help="[multimodal] DOA 允许的正对角锥度(±deg)")
    p.add_argument("--camera-index", type=int, default=0, help="[multimodal] OpenCV 摄像头索引")
    p.add_argument("--no-camera", action="store_true", help="[multimodal] 禁用摄像头")
    p.add_argument("--no-strict-multimodal", action="store_true",
                   help="[multimodal] 若视觉不可用，允许退化为纯音频严格模式")
    p.add_argument("--no-vad-gate", action="store_true", help="[kws] 关闭 VAD 门控（调试）")
    p.add_argument("--asr-model", type=str, default="tiny",
                   help="faster-whisper 模型: tiny/base/small/medium/large-v3 (默认 tiny)")
    p.add_argument("--verbose", action="store_true", help="打印所有分数")
    return p.parse_args()


def main():
    args = parse_args()
    # Windows 控制台默认 GBK 编码，强制 stdout/stderr 用 utf-8 避免设备名/中文报错
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if args.list_devices:
        print(sd.query_devices())
        return
    WakeDemo(args).run()


if __name__ == "__main__":
    main()
