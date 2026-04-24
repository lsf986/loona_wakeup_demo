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

from perception_audio import AudioFrontend, AudioFeatures, Audio3A
from perception_visual import VisualFrontend, VisualGates, HAVE_CV2

SAMPLE_RATE = 16000
VAD_FRAME_MS = 20
VAD_FRAME_LEN = SAMPLE_RATE * VAD_FRAME_MS // 1000  # 320 samples
KWS_CHUNK_LEN = 1280  # openWakeWord 要求 80 ms (=1280 samples @16k)


# ASR 后处理：过滤空白/无意义识别结果（多为环境噪声或同事远场语音被 whisper
# 用常见口头禅/语气词"幻觉"出来的短文本）
_MEANINGLESS_CHARS = set("嗯啊哦呃唉哈呀哎嘿噢哇呵咳喂哼唔呣阿噫唷唉~哦嗯啊")
_PUNCT_CHARS = set(" \t\n\r，。！？、.,!?~～·…—-“”\"'`()（）[]【】《》:;：；")


def _strip_punct(text: str) -> str:
    """去掉 ASR 结果里的标点符号。保留中英文字母与数字之间的单个空格，
    确保英文短语可读。
    """
    if not text:
        return text
    # 去除 _PUNCT_CHARS 中定义的所有标点（含中英文常见标点）
    chars = [ch for ch in text if ch not in _PUNCT_CHARS or ch == " "]
    # 合并连续空格 & 去首尾空格
    out = "".join(chars)
    out = " ".join(out.split())
    return out


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

    def __init__(self, energy_ratio_db: float = 5.0):
        self.energy_ratio_db = energy_ratio_db
        self.noise_floor = 40.0  # int16 RMS
        self.alpha = 0.02

    def is_speech(self, frame: np.ndarray) -> bool:
        f = frame.astype(np.float32)
        rms = float(np.sqrt(np.mean(f * f) + 1e-9))
        # 过零率（归一化到 [0,1]）
        zc = float(np.mean(np.abs(np.diff(np.signbit(f).astype(np.int8)))))
        # 阈值：高于噪声底 N dB
        thresh = self.noise_floor * (10 ** (self.energy_ratio_db / 20.0))
        speech = (rms > thresh) and (0.005 < zc < 0.40)
        if not speech:
            # 只在非语音时更新噪声底（保护跟踪器不被语音拉偏）
            self.noise_floor = (1 - self.alpha) * self.noise_floor + self.alpha * rms
        return speech
RING_SECONDS = 12         # 环形缓冲：12s 历史，兼顾长句头和长尾
REPLAY_SECONDS = 2.5      # 唪醒命中时已累积的历史时长（向前看）
POSTROLL_SECONDS = 6.0    # 唪醒后最多再录 6s 才送 ASR（硬上限；实际由尾点静音决定）
ENDPOINT_SILENCE_MS = 500 # 尾点检测：末尾连续静音 ≥ 500ms 即认为说完
ENDPOINT_MIN_WAIT_MS = 250 # 最少等 250ms，避免把短词末尾切掉
LISTEN_TIMEOUT_S = 10.0


# ------------------------------ Ring Buffer ------------------------------ #
class RingBuffer:
    """对应方案 §3.2：2s 环形缓冲，用于唤醒命中后回灌首句。"""

    def __init__(self, seconds: int, sr: int = SAMPLE_RATE):
        self.cap = seconds * sr
        self.buf = np.zeros(self.cap, dtype=np.int16)
        self.wpos = 0
        self.filled = 0
        self.total_written = 0  # 累计写入样本数，用于跨次 ASR 的边界隔离
        self.lock = threading.Lock()

    def write(self, data: np.ndarray) -> None:
        with self.lock:
            n = len(data)
            self.total_written += n
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

    def read_last(self, seconds: float, max_samples: int | None = None) -> np.ndarray:
        """读最近 seconds 的样本；若提供 max_samples 则再收紧到该样本数。"""
        with self.lock:
            n = min(int(seconds * SAMPLE_RATE), self.filled)
            if max_samples is not None:
                n = min(n, max(0, int(max_samples)))
            if n == 0:
                return np.zeros(0, dtype=np.int16)
            start = (self.wpos - n) % self.cap
            if start + n <= self.cap:
                return self.buf[start:start + n].copy()
            tail = self.cap - start
            return np.concatenate([self.buf[start:], self.buf[:n - tail]])

    def snapshot_total(self) -> int:
        with self.lock:
            return self.total_written


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


# ------------------------------ Speaker Tracker ------------------------------ #
class SpeakerTracker:
    """轻量声纹追踪：按基频/频谱重心对最近若干"语句"做粗分群。

    实现动机：真实声纹模型（ECAPA/resemblyzer 等）体积大、推理开销高，
    本 demo 只需要在"过去几秒内听到几位不同的说话人"这个粒度做判定，
    用于驱动多人场景（strict profile）。因此采用一组对人耳频谱差异敏感、
    完全基于 numpy 的启发式特征：

        - voiced 帧的 pitch_hz（中位数 & IQR）
        - voiced 帧的 spectral_centroid（中位数 & IQR）

    两句声音被判定为"同一说话人"的条件：
        | pitch_semitones | < pitch_tol 且 | centroid | < centroid_tol
    否则视为新说话人并加入 roster。

    保留窗口 retention_s：超过该窗口未再出现的说话人会被移除。
    """

    PITCH_TOL_ST = 2.5        # semitones
    CENT_TOL_HZ = 500.0
    MIN_UTT_FRAMES = 6        # 至少 120ms voiced 才算一句
    SILENCE_FRAMES = 20       # 连续 400ms 静音视为一句说完

    def __init__(self, retention_s: float = 8.0):
        self.retention_s = retention_s
        self._buf_pitch: list[float] = []
        self._buf_cent: list[float] = []
        self._silence_run = 0
        # roster: list[dict(id, ts, pitch, cent)]
        self._roster: list[dict] = []
        self._next_id = 0

    def _flush_utterance(self) -> int | None:
        """把当前累计帧收成一句，返回 speaker_id（或 None 如果帧数不足）。"""
        if len(self._buf_pitch) < self.MIN_UTT_FRAMES:
            self._buf_pitch.clear(); self._buf_cent.clear()
            return None
        p_med = float(np.median(self._buf_pitch))
        c_med = float(np.median(self._buf_cent))
        self._buf_pitch.clear(); self._buf_cent.clear()

        now = time.time()
        # 回收过期说话人
        self._roster = [r for r in self._roster if now - r["ts"] <= self.retention_s]

        best_id: int | None = None
        best_d = float("inf")
        for r in self._roster:
            if r["pitch"] <= 0 or p_med <= 0:
                continue
            semitones = abs(12.0 * np.log2(p_med / r["pitch"]))
            cent_d = abs(c_med - r["cent"])
            if semitones < self.PITCH_TOL_ST and cent_d < self.CENT_TOL_HZ:
                score = semitones / self.PITCH_TOL_ST + cent_d / self.CENT_TOL_HZ
                if score < best_d:
                    best_d = score
                    best_id = r["id"]

        if best_id is None:
            best_id = self._next_id
            self._next_id += 1
            self._roster.append(dict(id=best_id, ts=now, pitch=p_med, cent=c_med))
        else:
            for r in self._roster:
                if r["id"] == best_id:
                    # EMA 更新模板，适应音调漂移
                    r["ts"] = now
                    r["pitch"] = 0.7 * r["pitch"] + 0.3 * p_med
                    r["cent"] = 0.7 * r["cent"] + 0.3 * c_med
                    break
        return best_id

    def feed(self, is_voiced: bool, pitch_hz: float, spec_centroid: float) -> int | None:
        """每帧调用。返回 speaker_id 仅在刚好完成一句时非 None。"""
        if is_voiced and pitch_hz > 0:
            self._buf_pitch.append(float(pitch_hz))
            self._buf_cent.append(float(spec_centroid))
            self._silence_run = 0
            return None
        # silence
        if self._buf_pitch:
            self._silence_run += 1
            if self._silence_run >= self.SILENCE_FRAMES:
                sid = self._flush_utterance()
                self._silence_run = 0
                return sid
        return None

    def distinct_recent(self, window_s: float = 6.0) -> int:
        now = time.time()
        return len({r["id"] for r in self._roster if now - r["ts"] <= window_s})

    def snapshot(self, window_s: float = 6.0) -> dict:
        now = time.time()
        active = [r for r in self._roster if now - r["ts"] <= window_s]
        return dict(distinct=len(active), ids=[r["id"] for r in active])


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
def _load_tuned_config() -> dict:
    """读取 auto_tune 生成的阈值配置（可选）。
    文件位置: demo/config/thresholds.json。不存在则返回空字典。
    """
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "config", "thresholds.json")
    if not os.path.isfile(cfg_path):
        return {}
    try:
        import json
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if isinstance(cfg, dict):
            print(f"[CFG ] 载入自适应阈值 {cfg_path}: {cfg}")
            return cfg
    except Exception as e:
        print(f"[CFG ] 读取 {cfg_path} 失败: {e}", file=sys.stderr)
    return {}


class WakeDemo:
    # 三档 profile 的默认策略表：必需门 / 可选门权重 / 可选分阈值 / 持续帧 / 冷却秒数
    # auto_tune 可以通过 config/thresholds.json 的 "profiles" 字段覆盖
    DEFAULT_PROFILES: dict = {
        "relaxed": {
            # 单人安静环境：只要求声/人脸在场，持续帧与冷却进一步降低
            "required": ["voiced", "face"],
            "weights": {"gaze": 0.9, "lip": 0.8, "near_field": 0.7,
                         "av_sync": 0.6, "snr": 0.5, "doa": 0.3,
                         "speech_run": 0.6},
            "optional_thresh": 0.9,
            "sustain_frames": 2,
            "debounce_s": 0.6,
        },
        "normal": {
            "required": ["voiced", "snr", "speech_run", "face", "gaze",
                         "lip", "near_field", "av_sync"],
            "weights": {},
            "optional_thresh": 0.0,
            "sustain_frames": 5,
            "debounce_s": 1.5,
        },
        "strict": {
            "required": ["voiced", "snr", "speech_run", "face", "gaze",
                         "lip", "near_field", "av_sync"],
            "weights": {"gaze": 1.8, "doa": 1.2, "av_sync": 1.2},
            "optional_thresh": 0.0,
            "sustain_frames": 7,
            "debounce_s": 2.0,
        },
    }

    def __init__(self, args):
        # 先载入自适应阈值，按 key 覆盖 args 里对应字段（仅覆盖标量字段；
        # 结构字段如 profiles 保留在 self._tuned_cfg 里供 _resolve_profile_cfg 使用）
        tuned = _load_tuned_config()
        for k, v in tuned.items():
            if k == "profiles":
                continue
            if hasattr(args, k) and not isinstance(v, (dict, list)):
                setattr(args, k, v)
        self.args = args
        self._tuned_cfg = tuned
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
        # 记录上一次送入 ASR 的音频截止样本，避免下一次识别把旧句的尾巴也读进来。
        self._last_asr_end_sample = 0
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
        # 环境画像：噪声等级 + 人数 → 宽/严判决策略
        self._env_face_hist: collections.deque = collections.deque(maxlen=50)  # ~1s@50fps audio tick
        self._env_voiced_hist: collections.deque = collections.deque(maxlen=50)
        self._env_noise_hist: collections.deque = collections.deque(maxlen=100)  # 2s
        self._env_profile = "normal"   # 'relaxed' | 'normal' | 'strict'
        self._env_noise_label = "normal"  # 'quiet' | 'normal' | 'noisy'
        self._env_crowd_label = "none"    # 'none' | 'single' | 'multi'
        self._last_env_emit = 0.0
        # 声纹追踪：基于 pitch+centroid 的轻量说话人聚类，驱动"多人场景"判定
        self.speakers = SpeakerTracker(retention_s=8.0)
        self._speakers_distinct = 0
        # 最近 1000ms 的 (voiced, lip_moving) 对，用于唇-声同步门
        self._av_window = collections.deque(maxlen=50)  # 50 * 20ms = 1000ms
        # 最近 10 帧（200ms）DOA 是否在锥内，用于做时间迟滞，抗 GCC-PHAT 抖动
        self._doa_window = collections.deque(maxlen=10)
        # 持续命中计数：要求硬门限连续 N 帧全绿才触发，抑制瞬时误报
        self._sustain_pass_frames = 0

        # 增强音频前端（多模态模式主力，其他模式也可用于近场诊断）
        self.audio_fe = AudioFrontend(
            near_field_rms=float(getattr(args, "near_field_rms", 400.0)),
            voicing_min=0.20,
        )

        # 3A 预处理（HPF + webrtc 风格 ANS + AGC），默认温和强度
        self.use_3a = not bool(getattr(args, "no_3a", False))
        apm_strength = float(getattr(args, "apm_strength", 0.3))
        self.audio_3a = (
            Audio3A(sr=SAMPLE_RATE, frame_len=VAD_FRAME_LEN,
                     strength=apm_strength)
            if self.use_3a else None
        )
        if self.use_3a and self.audio_3a is not None:
            print(f"[3A  ] backend={self.audio_3a.backend} strength={apm_strength:.2f}")
        elif not self.use_3a:
            print("[3A  ] disabled")

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

        # ===== 数据采集：供 GUI 标注正/负样本，用于 auto_tune 迭代 =====
        # 采集时 _loop 会把 PCM、摄像头帧、每帧特征写入 _collector；stop_collect 时落盘
        self._collector = None  # type: dict | None
        self._collector_lock = threading.Lock()

    def _emit(self, event_type: str, **payload):
        cb = self.on_event
        if cb is not None:
            try:
                cb(event_type, payload)
            except Exception as e:
                print(f"[EVENT-CB-ERR] {e}", file=sys.stderr)

    def _resolve_profile_cfg(self, profile: str) -> dict:
        """返回当前 profile 的判决策略（必需门/权重/阈值/持续帧/冷却）。
        默认值来自 DEFAULT_PROFILES；若 tuned 配置里有 "profiles" 字段，
        则逐字段覆盖（允许部分覆盖，例如只改 weights）。
        """
        base = self.DEFAULT_PROFILES.get(profile, self.DEFAULT_PROFILES["normal"])
        # 深拷贝一份避免意外修改类属性
        cfg = {
            "required": list(base["required"]),
            "weights": dict(base["weights"]),
            "optional_thresh": float(base["optional_thresh"]),
            "sustain_frames": int(base["sustain_frames"]),
            "debounce_s": float(base["debounce_s"]),
        }
        tuned_profiles = (self._tuned_cfg or {}).get("profiles") or {}
        override = tuned_profiles.get(profile) or {}
        if "required" in override and isinstance(override["required"], list):
            cfg["required"] = list(override["required"])
        if "weights" in override and isinstance(override["weights"], dict):
            # 按键合并：允许 auto_tune 只写部分权重
            merged = dict(cfg["weights"])
            for k, v in override["weights"].items():
                try:
                    merged[k] = float(v)
                except (TypeError, ValueError):
                    continue
            cfg["weights"] = merged
        for k in ("optional_thresh", "sustain_frames", "debounce_s"):
            if k in override:
                try:
                    cfg[k] = type(cfg[k])(override[k])
                except (TypeError, ValueError):
                    pass
        return cfg

    # ------------------- 数据采集（标注正/负样本） ------------------- #
    def start_collect(self, label: str) -> str | None:
        """开始采集样本。label ∈ {'positive','negative'}。返回保存目录或 None。

        采集策略（精简版）：
          - 不保存原始音频 wav、不保存摄像头帧 jpg
          - 仅在内存中累计每帧特征（注视、唇动、近场、SNR、浊音、DOA 等）
          - stop_collect 时聚合出 summary.json（均值/占比/分位数），体积极小
        """
        if label not in ("positive", "negative"):
            raise ValueError("label must be 'positive' or 'negative'")
        with self._collector_lock:
            if self._collector is not None:
                return None
            root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "dataset", label, time.strftime("%Y%m%d_%H%M%S"))
            os.makedirs(root, exist_ok=True)
            self._collector = dict(
                label=label, root=root,
                features=[],           # list of dict (每帧特征)
                start_t=time.time(),
            )
        self._emit("collect_start", label=label, root=root)
        print(f"[COLLECT] 开始采集 {label} 样本 -> {root}")
        return root

    def stop_collect(self) -> dict | None:
        """停止采集并落盘。只保存一份聚合后的 summary.json（包含各门命中率、
        关键特征的均值/分位数），供 auto_tune 使用。
        """
        with self._collector_lock:
            c = self._collector
            self._collector = None
        if c is None:
            return None
        import json as _json
        duration = time.time() - c["start_t"]
        rows = c["features"]
        summary = self._aggregate_collect(rows)
        summary.update(dict(
            label=c["label"],
            duration_s=round(duration, 3),
            frames=len(rows),
            sample_rate_hz=50,  # 20ms / 帧 = 50Hz
            root=c["root"],
        ))
        try:
            with open(os.path.join(c["root"], "summary.json"), "w",
                      encoding="utf-8") as f:
                _json.dump(summary, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[COLLECT] 写 summary 失败: {e}", file=sys.stderr)
        print(f"[COLLECT] 保存完成: "
              f"label={summary['label']} duration={summary['duration_s']}s "
              f"frames={summary['frames']} "
              f"gaze_rate={summary.get('gaze_rate', 0):.2f} "
              f"lip_rate={summary.get('lip_rate', 0):.2f}")
        meta = dict(label=summary["label"],
                    duration_s=summary["duration_s"],
                    frames=summary["frames"],
                    root=summary["root"])
        self._emit("collect_done", **meta)
        return meta

    def _aggregate_collect(self, rows: list[dict]) -> dict:
        """把每帧特征聚合成一个小的统计对象（比存全量省几个数量级空间）。
        字段命名与 auto_tune 所用一致。
        """
        n = len(rows)
        if n == 0:
            return dict(empty=True)
        arr = lambda k: np.array([r.get(k, 0.0) for r in rows], dtype=np.float32)
        rate = lambda k: float(np.mean([1.0 if r.get(k) else 0.0 for r in rows]))

        def stats(vals: np.ndarray) -> dict:
            if vals.size == 0:
                return dict(mean=0.0, p15=0.0, p50=0.0, p85=0.0, p90=0.0)
            return dict(
                mean=round(float(np.mean(vals)), 3),
                p15=round(float(np.percentile(vals, 15)), 3),
                p50=round(float(np.percentile(vals, 50)), 3),
                p85=round(float(np.percentile(vals, 85)), 3),
                p90=round(float(np.percentile(vals, 90)), 3),
            )

        # 只对"有语音"的帧统计语音相关特征，避免静音段拉偏
        active = [r for r in rows if r.get("voiced") or r.get("is_speech")]
        active_rms = np.array([r.get("rms", 0.0) for r in active], dtype=np.float32)
        active_snr = np.array([r.get("snr", 0.0) for r in active], dtype=np.float32)
        # 面部相关：只在"有脸"的帧统计
        face_rows = [r for r in rows if r.get("face")]
        face_area_vals = np.array([r.get("face_area", 0.0) for r in face_rows], dtype=np.float32)
        lip_std_vals = np.array([r.get("lip_std", 0.0) for r in face_rows], dtype=np.float32)

        # av_sync 帧：voiced 且 lip
        av_sync_rate = float(np.mean([1.0 if (r.get("voiced") and r.get("lip")) else 0.0
                                      for r in rows]))

        return dict(
            # 帧级命中率（权重调参的核心信号）
            voiced_rate=rate("voiced"),
            is_speech_rate=rate("is_speech"),
            face_rate=rate("face"),
            multi_face_rate=float(np.mean([1.0 if r.get("face_count", 0) >= 2 else 0.0
                                           for r in rows])),
            gaze_rate=rate("gaze"),
            lip_rate=rate("lip"),
            near_field_audio_rate=rate("near_field_audio"),
            near_field_visual_rate=rate("near_field_visual"),
            av_sync_rate=av_sync_rate,
            # 关键连续量分位数（仅语音/仅有脸段）
            active_frames=len(active),
            rms=stats(active_rms),
            snr=stats(active_snr),
            face_area=stats(face_area_vals),
            lip_std=stats(lip_std_vals),
        )

    def _collect_tick(self, frame_i16: np.ndarray, is_speech: bool, snr: float,
                       feats: "AudioFeatures", vg: "VisualGates | None"):
        """在 _loop 每 20ms 调用一次；只有采集器激活时才真正工作。
        只提取判决所需要的语音+注视+唇动特征，不保留原始波形/图像。
        """
        c = self._collector
        if c is None:
            return
        row = dict(
            is_speech=bool(is_speech),
            rms=float(feats.rms),
            voiced=bool(feats.is_voiced),
            pitch=float(feats.pitch_hz or 0.0),
            centroid=float(feats.spec_centroid),
            snr=float(snr),
            near_field_audio=bool(feats.near_field_ok),
            doa=float(feats.doa_deg) if feats.doa_deg is not None else None,
        )
        if vg is not None and vg.available:
            row.update(dict(
                face=bool(vg.face_present),
                face_count=int(vg.face_count),
                gaze=bool(vg.gaze_aligned),
                near_field_visual=bool(vg.near_field),
                lip=bool(vg.lip_moving),
                face_area=float(vg.face_area_ratio),
                face_offset=float(vg.face_center_offset),
                lip_std=float(vg.lip_motion_std),
            ))
        else:
            row.update(dict(face=False, face_count=0, gaze=False,
                             near_field_visual=False, lip=False,
                             face_area=0.0, face_offset=0.0, lip_std=0.0))
        c["features"].append(row)

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
                            num_threads=4,   # i5/i7 上 4 线程解码更快
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

    # ---- ASR 前处理：去静音 + RMS 归一化（不做预加重，SenseVoice 对原始波形更鲁棒） ---- #
    def _prep_asr_audio(self, pcm_i16: np.ndarray) -> np.ndarray:
        """把送入 ASR 的 int16 PCM 转为 float32，并做两步增强：

        1) VAD 裁剪：按 20ms 帧计算 RMS，动态阈值取 max(静态底, 0.4·中位数)，
           容忍 ≤ 160ms 的短暂静音（不因换气断句），首尾各保留 250ms padding。
        2) RMS 归一化到 ≈ -20 dBFS，配合软限幅。

        注：不做 pre-emphasis。SenseVoice 训练数据为原始 16k PCM，预加重会
        明显降低中文短词识别率（实测"停"/"开灯"等经常被识成空）。
        """
        if pcm_i16.size == 0:
            return np.zeros(0, dtype=np.float32)
        audio = pcm_i16.astype(np.float32) / 32768.0

        fl = int(0.02 * SAMPLE_RATE)
        pad = int(0.25 * SAMPLE_RATE)          # 250ms padding
        n_frames = len(audio) // fl
        if n_frames >= 20:
            energies = np.array([
                np.sqrt(np.mean(audio[i * fl:(i + 1) * fl] ** 2) + 1e-9)
                for i in range(n_frames)
            ], dtype=np.float32)
            static_floor = 0.005                # ≈ -46 dBFS
            dyn_floor = 0.4 * float(np.median(energies))
            thr = max(static_floor, dyn_floor)
            voiced = energies > thr
            if voiced.any():
                gap = 0
                best_s = best_e = -1
                cur_s = -1
                best_len = 0
                for i, v in enumerate(voiced):
                    if v:
                        if cur_s < 0:
                            cur_s = i
                        gap = 0
                        cur_e = i
                        cur_len = cur_e - cur_s + 1
                        if cur_len > best_len:
                            best_s, best_e, best_len = cur_s, cur_e, cur_len
                    else:
                        gap += 1
                        if gap > 25:           # 容忍 500ms 换气/停顿，避免长句被切成两段
                            cur_s = -1
                            gap = 0
                if best_s >= 0:
                    s_smp = max(0, best_s * fl - pad)
                    e_smp = min(len(audio), (best_e + 1) * fl + pad)
                    audio = audio[s_smp:e_smp]

        # RMS 归一到 -20 dBFS
        rms = float(np.sqrt(np.mean(audio ** 2) + 1e-9))
        if rms > 1e-4:
            target = 0.10
            gain = min(5.0, target / rms)
            audio = audio * gain
            np.clip(audio, -0.99, 0.99, out=audio)
        return audio.astype(np.float32)

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
                # 统一的 ASR 前处理：裁掉静音段 + 预加重 + RMS 归一化
                audio_f = self._prep_asr_audio(pcm_i16)
                if audio_f.size < int(0.2 * SAMPLE_RATE):
                    # 语音段过短（<200ms），直接判为无效
                    self._reject_wake(wake_count_snapshot)
                    self._emit("asr", word=name, score=score, text="",
                               count=wake_count_snapshot,
                               rejected=True,
                               reject_reason="语音段过短（<200ms）",
                               backend=backend)
                    return

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
                        no_speech_threshold=0.5,
                        log_prob_threshold=-1.0,
                        compression_ratio_threshold=2.4,
                        initial_prompt="以下是用户对智能助手的中文指令，请用简体中文转写。",
                    )
                    text = "".join(seg.text for seg in segments).strip()
                    lang_detected = getattr(info, "language", None)

                print(f"[ASR/{backend}] 识别结果: {text!r}")
                rejected, reject_reason = _asr_is_meaningless(text)
                if rejected:
                    self._reject_wake(wake_count_snapshot)
                    print(f"[ASR] 判定为无效唤醒: {reject_reason}")
                # 去除标点后再对外暴露（GUI 展示 / 下游使用更干净）
                clean_text = _strip_punct(text)
                self._emit("asr", word=name, score=score, text=clean_text,
                           count=wake_count_snapshot, backend=backend,
                           rejected=rejected, reject_reason=reject_reason,
                           language=lang_detected)
                # ASR 成功完成后立即回到 IDLE，避免等待 8s LISTENING 超时
                # 导致"一次识别后要间隔几秒才能再次识别"。
                if not rejected and self.state == "LISTENING":
                    self.state = "IDLE"
                    self._emit("state", state=self.state, reason="asr_done")
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
        """唪醒命中。立即 emit wake 事件；随后对尾部做 endpointing，
        一旦检测到说完（末尾连续静音 ≥ ENDPOINT_SILENCE_MS）立刻送 ASR，
        最多等 POSTROLL_SECONDS。"""
        self.wake_count += 1
        print(f">>> WAKE #{self.wake_count}  {name} {score:.2f}")
        self._emit("wake", word=name, score=score,
                   duration=REPLAY_SECONDS + POSTROLL_SECONDS,
                   count=self.wake_count)

        wake_count_snapshot = self.wake_count

        def _delayed():
            # 尾点检测：每 50ms 轮询末尾 300ms 音频的 RMS，统计连续静音时长
            fl = int(0.02 * SAMPLE_RATE)        # 20ms
            sil_floor = max(80.0, self.noise.noise_rms * 2.5)  # 小声说话也不被误切
            min_wait = ENDPOINT_MIN_WAIT_MS / 1000.0
            max_wait = POSTROLL_SECONDS
            sil_need = ENDPOINT_SILENCE_MS / 1000.0
            t0 = time.time()
            sil_run = 0.0
            while True:
                time.sleep(0.05)
                elapsed = time.time() - t0
                if elapsed >= max_wait:
                    reason = "timeout"
                    break
                if elapsed < min_wait:
                    continue
                tail = self.ring.read_last(0.20)  # 最近 200ms
                if tail.size < fl:
                    continue
                # 末段 RMS（int16 域）
                tail_f = tail.astype(np.float32)
                rms_tail = float(np.sqrt(np.mean(tail_f * tail_f) + 1e-6))
                if rms_tail < sil_floor:
                    sil_run += 0.05
                    if sil_run >= sil_need:
                        reason = "endpoint"
                        break
                else:
                    sil_run = 0.0

            history_seconds = REPLAY_SECONDS + (time.time() - t0) + 0.3  # 多留 300ms 裕量
            history = self.ring.read_last(
                history_seconds,
                max_samples=max(0, self.ring.snapshot_total() - self._last_asr_end_sample),
            )
            # 记录本次读取的结束位置，避免下次唤醒又把这段音频当成"上一句"重复识别
            self._last_asr_end_sample = self.ring.snapshot_total()
            waited_ms = int((time.time() - t0) * 1000)
            if getattr(self.args, "verbose", False):
                dur = len(history) / SAMPLE_RATE
                print(f"[ASR ] endpoint={reason} wait={waited_ms}ms audio={dur:.2f}s")
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

            # 环形缓冲保存"原始"音频——ASR 直接从 ring 读，避免 3A 的 ANS/AGC
            # 污染模型输入。3A 处理只用于下游 VAD/特征/门限判决。
            self.ring.write(frame)

            # 3A 预处理（仅影响 VAD/特征，不影响 ASR 输入）：
            # 使用上一帧的 is_speech 作为噪声估计 hint，避免讲话时更新噪声模型。
            if self.audio_3a is not None:
                frame = self.audio_3a.process(
                    frame, ref_i16=ref,
                    is_speech_hint=bool(getattr(self, "_prev_is_speech", False)),
                )

            # 轻量 VAD（做状态保留/超时）
            is_speech = self.vad.is_speech(frame)
            self._prev_is_speech = is_speech

            # 噪声底/SNR
            rms = NoiseTracker.rms(frame)
            self.noise.update(rms, is_speech)
            snr = self.noise.snr_db(rms)

            # 增强音频特征（HPF、浊音、频谱重心、DOA、近场）
            feats = self.audio_fe.process(frame, ref)

            # 声纹追踪：把 voiced 帧喂给 SpeakerTracker，按句聚类
            self.speakers.feed(bool(feats.is_voiced), float(feats.pitch_hz or 0.0),
                               float(feats.spec_centroid or 0.0))
            self._speakers_distinct = self.speakers.distinct_recent(6.0)

            # 如果采集器开启：无论当前模式是否是 multimodal，都抓一次视觉快照
            # （采集始终记录音视频+特征，方便后续 auto_tune）
            if self._collector is not None:
                vg_snap = self.visual.snapshot() if self.visual is not None else None
                self._collect_tick(frame, is_speech, snr, feats, vg_snap)

            # LISTENING 超时
            if self.state == "LISTENING" and time.time() - self.listen_start > LISTEN_TIMEOUT_S:
                print("<<< LISTENING 超时 8s，回到 IDLE")
                self.state = "IDLE"
                self._emit("state", state=self.state, reason="timeout")

            if self.mode == "kws":
                try:
                    self._process_kws(frame, is_speech, snr, feats)
                except Exception as e:
                    print(f"[LOOP-ERR/kws] {e}", file=sys.stderr)
            elif self.mode == "multimodal":
                try:
                    self._process_multimodal(is_speech, snr, feats)
                except Exception as e:
                    print(f"[LOOP-ERR/multimodal] {e}", file=sys.stderr)
            else:
                try:
                    self._process_vad(is_speech, snr)
                except Exception as e:
                    print(f"[LOOP-ERR/vad] {e}", file=sys.stderr)

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
                self._emit("interrupt_hint", word=best_name, score=float(best_score))

    # -------- 模式 C：多模态硬门限路径（方案 §5.2 路径 B 正式实现） -------- #
    def _update_env_profile(self, vg: "VisualGates", voiced_now: bool, snr_db: float):
        """环境画像：根据噪声底和人脸数量动态选择宽/严策略。

        返回 (profile, snapshot_dict)。profile ∈ {'relaxed','normal','strict'}。

        关键修正：为避免单人安静环境下用户自己说话被当成"嘈杂"，
        噪声评估只用"voiced 且 用户无唇动"的帧（近似视为外部人声），
        从而把用户本人的当前语音排除在外。同时把嘈杂阈值抬高。

        判定规则（灵敏度偏好：更容易进入 relaxed）：
        1) 噪声：
           - quiet:  noise_rms 2s 均值 < 400 且 other_voice_ratio < 0.25
           - noisy:  noise_rms > 900 或 other_voice_ratio > 0.60
           - normal: 其他
        2) 人数：
           - single: 近 1s 内 face_count == 1 的帧占比 ≥ 55%
           - multi:  同窗口 face_count ≥ 2 的帧占比 ≥ 30%
           - none:   无脸
        3) profile 合成：
           - relaxed = quiet + single
           - strict  = noisy 或 multi
           - normal  = 其它
        """
        # 采样：噪声底、人脸数
        self._env_noise_hist.append(self.noise.noise_rms)
        self._env_face_hist.append(int(vg.face_count) if vg.available else 0)
        # 外部人声证据：voiced 且 用户本人没有在说话（无唇动）
        # 视觉不可用时无法区分，回退为 voiced
        if vg.available:
            other_voice_now = bool(voiced_now and not vg.lip_moving)
        else:
            other_voice_now = bool(voiced_now)
        self._env_voiced_hist.append(1 if other_voice_now else 0)

        # 样本不足时给出中性画像
        if len(self._env_noise_hist) < 25:  # 0.5s
            profile = "normal"
            snap = dict(noise="warmup", crowd="warmup", profile=profile,
                        noise_rms=float(self.noise.noise_rms),
                        voiced_ratio=0.0, face_count=0,
                        single_ratio=0.0, multi_ratio=0.0)
            return profile, snap

        noise_mean = float(np.mean(self._env_noise_hist))
        other_voice_ratio = float(np.mean(self._env_voiced_hist))
        face_arr = np.array(self._env_face_hist)
        single_ratio = float(np.mean(face_arr == 1))
        multi_ratio = float(np.mean(face_arr >= 2))

        if noise_mean < 400.0 and other_voice_ratio < 0.25:
            noise_label = "quiet"
        elif noise_mean > 900.0 or other_voice_ratio > 0.60:
            noise_label = "noisy"
        else:
            noise_label = "normal"

        if not vg.available:
            crowd_label = "none"
        elif multi_ratio >= 0.30:
            crowd_label = "multi"
        elif single_ratio >= 0.55:
            crowd_label = "single"
        else:
            crowd_label = "none"

        # 声纹：短时间内听到 ≥2 位不同说话人 → 强制标记多人
        speakers_distinct = int(self._speakers_distinct)
        if speakers_distinct >= 2:
            crowd_label = "multi"

        if noise_label == "quiet" and crowd_label == "single":
            profile = "relaxed"
        elif noise_label == "noisy" or crowd_label == "multi":
            profile = "strict"
        else:
            profile = "normal"

        self._env_noise_label = noise_label
        self._env_crowd_label = crowd_label
        self._env_profile = profile

        # 变化或节流 1.5s emit 一次，给 GUI 显示
        now = time.time()
        if (profile != getattr(self, "_env_last_profile", None)
                or now - self._last_env_emit > 1.5):
            self._env_last_profile = profile
            self._last_env_emit = now
            self._emit("env", profile=profile, noise=noise_label, crowd=crowd_label,
                       noise_rms=noise_mean, voiced_ratio=other_voice_ratio,
                       single_ratio=single_ratio, multi_ratio=multi_ratio,
                       speakers_distinct=speakers_distinct)

        return profile, dict(noise=noise_label, crowd=crowd_label, profile=profile,
                             noise_rms=noise_mean, voiced_ratio=other_voice_ratio,
                             single_ratio=single_ratio, multi_ratio=multi_ratio,
                             speakers_distinct=speakers_distinct)

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
        if feats.is_voiced and snr >= 0.5:
            self.speech_run_ms += VAD_FRAME_MS
            self.silence_run_ms = 0.0
            self._voiced_hangover = 25  # 25*20ms = 500ms 尾拖
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
                voiced_cnt >= 3 and lip_cnt >= 3
                and coincide >= max(2, int(0.35 * voiced_cnt))
                and voiced_only <= int(1.8 * coincide) + 2  # 反证帧放宽，允许短暂嘴型遮挡
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

        # ===== 环境画像：根据噪声/人数选择宽/严策略（方案"添加权重"） ===== #
        profile, env_snap = self._update_env_profile(vg, voiced_now, snr)

        # 多人场景（视觉 ≥2 张脸 或 声纹 ≥2 位说话人）下收紧 av_sync：
        #   - 使用 500ms 窗，共现 ≥ 45%、反证帧 ≤ 共现+2
        #   - VisualFrontend 只对主脸做唇动检测，所以只有主脸真在说话才会通过
        if vg.available and (vg.face_count >= 2
                             or env_snap.get("speakers_distinct", 0) >= 2):
            if len(self._av_window) >= 25:  # 500ms
                _co = sum(1 for v, l in self._av_window if v and l)
                _vc = sum(1 for v, _ in self._av_window if v)
                _vo = sum(1 for v, l in self._av_window if v and not l)
                gates["av_sync"] = (
                    _vc >= 5
                    and _co >= max(3, int(0.45 * _vc))
                    and _vo <= _co + 2
                )
            else:
                gates["av_sync"] = False

        # 每个 profile 的权重表：默认值 + 可被 auto_tune 配置覆盖。
        # 配置来源: self._tuned_cfg['profiles'][profile]，若不存在则用内置默认。
        prof_cfg = self._resolve_profile_cfg(profile)
        required = list(prof_cfg["required"])
        weights = dict(prof_cfg["weights"])
        optional_thresh = float(prof_cfg["optional_thresh"])
        sustain_frames = int(prof_cfg["sustain_frames"])
        debounce_s = float(prof_cfg["debounce_s"])

        # ===== 多人即时加固 =====
        # 只要当前画面里出现 ≥2 张脸，或声纹窗口内出现 ≥2 位说话人，
        # 就立即（不等 env profile 积累）强制把 gaze/lip/av_sync/face 纳入必需门，
        # 并抬高 sustain_frames / debounce，避免旁观者或他人说话蹭唤醒。
        multi_face_now = bool(vg.available and vg.face_count >= 2)
        multi_speaker_now = int(env_snap.get("speakers_distinct", 0)) >= 2
        if multi_face_now or multi_speaker_now:
            for k in ("voiced", "snr", "speech_run", "face",
                      "gaze", "lip", "near_field", "av_sync"):
                if k not in required:
                    required.append(k)
            sustain_frames = max(sustain_frames, 7)
            debounce_s = max(debounce_s, 2.0)

        if profile == "strict":
            # SNR 在 strict 下再抬高 2dB，对应方案 §5.3 的嘈杂抬阈
            min_snr_eff = min_snr_db + 2.0
            gates["snr"] = snr >= min_snr_eff

        # 必需门是否全通过
        required_ok = all(gates.get(k, False) for k in required)
        # 可选门的加权得分：仅计算不在 required 里的门
        optional_score = sum(w for k, w in weights.items()
                             if k not in required and gates.get(k, False))
        # 综合判决
        evidence_ok = required_ok and (optional_score >= optional_thresh)
        # 兼容原有 "all_pass" 语义：normal/strict 下其实就是全通过
        all_pass = evidence_ok

        # 持续命中门限：按 profile 定长；不直接清零，允许单帧抖动
        SUSTAIN_FRAMES = sustain_frames
        if all_pass:
            self._sustain_pass_frames += 1
        else:
            self._sustain_pass_frames = max(0, self._sustain_pass_frames - 2)
        sustained = self._sustain_pass_frames >= SUSTAIN_FRAMES

        # 防抖 + 冷却 + TTS 抑制
        now = time.time()
        if all_pass and not sustained:
            reason = f"hold {self._sustain_pass_frames}/{SUSTAIN_FRAMES}({profile})"
        elif sustained and (now - self.last_fire) < debounce_s:
            sustained = False
            reason = "veto:debounce"
        elif self.arbiter.tts_playing and sustained:
            # TTS 播报中 + SNR 仍处理为疑似回环，抬高要求
            if snr < min_snr_db + 3:
                sustained = False
                reason = "veto:tts-playing"
            else:
                reason = f"ok({profile})"
        else:
            # 找第一个未通过的必需门或诊断出缺少可选分
            failed = [k for k in required if not gates.get(k, False)]
            if not failed and not evidence_ok:
                reason = (f"veto:opt<{optional_thresh:.1f}"
                          f"(got={optional_score:.1f},{profile})")
            else:
                reason = (f"ok({profile})" if sustained
                          else f"veto:{','.join(failed)}({profile})")

        # 诊断：仅 --verbose 时才打印门限状态（常规运行不再刷屏）
        if is_speech and getattr(self.args, "verbose", False):
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
                        lip_std=vg.lip_motion_std, fps=vg.fps,
                        face_count=vg.face_count),
            gates=gates,
            env=env_snap,
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
    p.add_argument("--min-speech-ms", type=float, default=200.0,
                   help="[vad/multimodal] 触发所需的最小连续语音时长(ms)")
    p.add_argument("--min-snr-db", type=float, default=3.0,
                   help="[vad/multimodal] 触发所需的最小 SNR(dB)")
    p.add_argument("--near-field-rms", type=float, default=350.0,
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
    p.add_argument("--no-3a", action="store_true",
                   help="关闭 3A 预处理(HPF+ANS+AGC)，用于对比/排查")
    p.add_argument("--apm-strength", type=float, default=0.3,
                   help="ANS 过滤强度 0~1，默认 0.3（值大 = 压噪更狠，但可能剥剥声），0 等同关闭 ANS")
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
