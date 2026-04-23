"""
音频前端增强 - 针对办公室远场人声/稳态噪声的屏蔽
---------------------------------------------------
提供：
  - 1 阶高通滤波（80Hz，去风扇/空调低频轰鸣）
  - 浊音检测（短时自相关，过滤瞬态与无意义能量）
  - 频谱重心（远场高频被空气吸收，区间外判为非近讲）
  - GCC-PHAT DOA（仅当输入为多通道时启用）
  - 近场绝对能量门限（核心：同事远场语音 RMS 显著低于用户近讲）
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

SR = 16000


@dataclass
class AudioFeatures:
    rms: float
    is_voiced: bool
    spec_centroid: float   # Hz
    pitch_hz: float
    doa_deg: float | None  # None 表示不可用（单通道）
    near_field_ok: bool    # 绝对能量近场门限


class HighPass1:
    """1 阶高通: y[n] = a*(y[n-1] + x[n] - x[n-1])。"""

    def __init__(self, fc: float = 80.0, sr: int = SR):
        import math
        rc = 1.0 / (2 * math.pi * fc)
        dt = 1.0 / sr
        self.a = rc / (rc + dt)
        self.prev_x = 0.0
        self.prev_y = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x, dtype=np.float32)
        a = self.a
        px, py = self.prev_x, self.prev_y
        xf = x.astype(np.float32)
        for i in range(len(xf)):
            py = a * (py + xf[i] - px)
            px = xf[i]
            out[i] = py
        self.prev_x, self.prev_y = px, py
        return out


def spectral_centroid(x: np.ndarray, sr: int = SR) -> float:
    if len(x) < 64:
        return 0.0
    w = np.hanning(len(x))
    mag = np.abs(np.fft.rfft(x * w))
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    s = mag.sum() + 1e-9
    return float((freqs * mag).sum() / s)


def autocorr_pitch(x: np.ndarray, sr: int = SR, fmin: float = 80.0, fmax: float = 400.0) -> tuple[float, float]:
    """返回 (pitch_hz, voiced_strength 0~1)。"""
    if len(x) < int(sr / fmin):
        return 0.0, 0.0
    x = x - x.mean()
    if np.std(x) < 50:
        return 0.0, 0.0
    lag_min = int(sr / fmax)
    lag_max = int(sr / fmin)
    # 归一化自相关
    ref = np.dot(x, x) + 1e-9
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac = ac[: lag_max + 1] / ref
    if lag_max <= lag_min:
        return 0.0, 0.0
    seg = ac[lag_min: lag_max + 1]
    peak = int(np.argmax(seg))
    strength = float(seg[peak])
    lag = lag_min + peak
    if lag <= 0:
        return 0.0, 0.0
    return float(sr / lag), max(0.0, min(1.0, strength))


def gcc_phat(sig: np.ndarray, ref: np.ndarray, sr: int = SR, max_tau: float | None = None,
             interp: int = 4) -> tuple[float, float]:
    """广义互相关-相位变换，返回 (tau_sec, 相关峰强度)。"""
    n = len(sig) + len(ref)
    SIG = np.fft.rfft(sig, n=n)
    REF = np.fft.rfft(ref, n=n)
    R = SIG * np.conj(REF)
    R /= (np.abs(R) + 1e-12)
    cc = np.fft.irfft(R, n=n * interp)
    max_shift = int(interp * n / 2)
    if max_tau is not None:
        max_shift = min(int(interp * sr * max_tau), max_shift)
    cc = np.concatenate([cc[-max_shift:], cc[: max_shift + 1]])
    peak = int(np.argmax(np.abs(cc))) - max_shift
    strength = float(np.abs(cc[peak + max_shift]) / (np.abs(cc).max() + 1e-9))
    return float(peak) / (interp * sr), strength


class Audio3A:
    """轻量 3A 预处理：HPF + 温和 ANS（基于 webrtc 风格的频谱门控）+ 简易 AGC。

    背景：webrtc 官方 APM 的 Python 绑定（webrtc-audio-processing / webrtc-noise-gain
    / webrtcvad）在 Python 3.14 上均无预编译 wheel 且无法从源码构建；此处采用
    `noisereduce` 作为纯 Python 替代——它实现的"非平稳噪声谱门控"算法与
    webrtc NS 思路一致（先建立噪声画像，再按频点做软衰减），且默认参数更温和。
    若未安装 noisereduce，会自动退化为 HPF + AGC（不做 ANS，避免抖音）。

    关键参数：
    - `strength`（0~1）：ANS 强度。默认 0.3，远低于之前手写谱减法的 1.4 过减。
    - `agc_target_rms`：AGC 目标 RMS（int16 域），默认 1500（≈ -23 dBFS）。
    - `enable_aec`：是否启用 NLMS AEC；demo 无扬声器回采接口，默认关闭。

    处理顺序：HPF → (AEC 可选) → ANS → AGC，全部在 float32 域进行。
    """

    def __init__(self, sr: int = SR, frame_len: int = 320,
                 strength: float = 0.3,
                 agc_target_rms: float = 1500.0,
                 agc_max_gain: float = 4.0,
                 enable_aec: bool = False,
                 noise_profile_ms: int = 500):
        self.sr = sr
        self.frame_len = frame_len
        self.strength = float(max(0.0, min(1.0, strength)))
        self.agc_target_rms = float(agc_target_rms)
        self.agc_max_gain = float(agc_max_gain)
        self.enable_aec = bool(enable_aec)

        # 尝试加载 noisereduce（webrtc 风格 NS 的纯 Python 替代）
        try:
            import noisereduce as _nr  # type: ignore
            self._nr = _nr
            self._ns_backend = "noisereduce"
        except ImportError:
            self._nr = None
            self._ns_backend = "off"

        # --- HPF (80Hz, 1 阶高通) 状态 ---
        self._hpf_prev_x = 0.0
        self._hpf_prev_y = 0.0
        RC = 1.0 / (2.0 * np.pi * 80.0)
        dt = 1.0 / sr
        self._hpf_alpha = RC / (RC + dt)

        # --- 噪声画像（前 noise_profile_ms 内的非语音帧） ---
        self._noise_profile_ready = False
        self._noise_profile: np.ndarray | None = None
        self._noise_buf: list[np.ndarray] = []
        self._noise_target_frames = max(10, int(noise_profile_ms / (1000.0 * frame_len / sr)))

        # --- AGC 状态 ---
        self._agc_gain = 1.0
        self._agc_alpha = 0.1

        # --- AEC (NLMS) 状态 ---
        self._aec_w = np.zeros(512, dtype=np.float32)
        self._aec_xbuf = np.zeros(512, dtype=np.float32)

    @property
    def backend(self) -> str:
        return self._ns_backend

    def _hpf(self, x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x)
        px, py = self._hpf_prev_x, self._hpf_prev_y
        a = self._hpf_alpha
        for i in range(len(x)):
            py = a * (py + x[i] - px)
            px = x[i]
            y[i] = py
        self._hpf_prev_x, self._hpf_prev_y = float(px), float(py)
        return y

    def _aec(self, ref: np.ndarray, d: np.ndarray) -> np.ndarray:
        taps = self._aec_w.size
        out = np.empty_like(d)
        w = self._aec_w; xbuf = self._aec_xbuf
        mu = 0.1; eps = 1e-3
        for n in range(len(d)):
            xbuf[1:] = xbuf[:-1]; xbuf[0] = ref[n]
            y_hat = float(np.dot(w, xbuf))
            e = float(d[n] - y_hat)
            out[n] = e
            norm = float(np.dot(xbuf, xbuf)) + eps
            if norm > 1e3:
                w += (mu * e / norm) * xbuf
        return out

    def _ns(self, x: np.ndarray, is_speech_hint: bool) -> np.ndarray:
        """webrtc 风格的频谱门控噪声抑制。仅当 noisereduce 可用且
        噪声画像已建立时生效。帧长 20ms 时使用小 FFT（n_fft=256）。"""
        if self._nr is None or self.strength <= 0.01:
            return x
        # 未建立噪声画像：只在非语音帧积累
        if not self._noise_profile_ready:
            if not is_speech_hint:
                self._noise_buf.append(x.copy())
                if len(self._noise_buf) >= self._noise_target_frames:
                    self._noise_profile = np.concatenate(self._noise_buf)
                    self._noise_profile_ready = True
                    self._noise_buf.clear()
            return x
        try:
            y = self._nr.reduce_noise(
                y=x, sr=self.sr,
                y_noise=self._noise_profile,
                stationary=True,                    # 稳态噪声模式（风扇/空调）
                prop_decrease=self.strength,        # 0.3 → 只衰减 30% 的噪声能量
                n_fft=256, win_length=256, hop_length=128,
            )
            return y.astype(np.float32)
        except Exception:
            return x

    def process(self, frame_i16: np.ndarray,
                ref_i16: np.ndarray | None = None,
                is_speech_hint: bool = False) -> np.ndarray:
        """输入/输出都是 int16。ref_i16 为扬声器参考（可选，AEC 用）。"""
        x = frame_i16.astype(np.float32)
        # 1) HPF
        x = self._hpf(x)
        # 2) AEC（可选）
        if self.enable_aec and ref_i16 is not None and len(ref_i16) == len(frame_i16):
            x = self._aec(ref_i16.astype(np.float32), x)
        # 3) ANS（webrtc 风格频谱门控，强度默认 0.3）
        x = self._ns(x, is_speech_hint=is_speech_hint)
        # 4) AGC（保守，避免小声放大噪声）
        rms = float(np.sqrt(np.mean(x * x) + 1e-6))
        if rms < 30.0:
            tg = 1.0
        else:
            tg = self.agc_target_rms / rms
            tg = min(self.agc_max_gain, max(0.5, tg))
        self._agc_gain = (1 - self._agc_alpha) * self._agc_gain + self._agc_alpha * tg
        x = x * self._agc_gain
        # clip
        np.clip(x, -32768.0, 32767.0, out=x)
        return x.astype(np.int16)


class AudioFrontend:
    """把整流、特征提取、近场判定收口。"""

    def __init__(self, sr: int = SR, mic_spacing_m: float = 0.04,
                 near_field_rms: float = 900.0,
                 centroid_min: float = 300.0, centroid_max: float = 3500.0,
                 voicing_min: float = 0.22):
        self.sr = sr
        self.hp = HighPass1(80.0, sr)
        self.mic_spacing = mic_spacing_m
        self.near_field_rms = near_field_rms
        self.centroid_min = centroid_min
        self.centroid_max = centroid_max
        self.voicing_min = voicing_min

    def calibrate_near_field(self, samples_int16: np.ndarray):
        """用用户"近讲校准"录下的音频估计阈值，取 RMS 中位数的 0.5。"""
        x = samples_int16.astype(np.float32)
        # 分帧
        fl = int(0.02 * self.sr)
        rms_list = []
        for i in range(0, len(x) - fl, fl):
            seg = x[i: i + fl]
            rms_list.append(float(np.sqrt(np.mean(seg * seg) + 1e-9)))
        if not rms_list:
            return
        med = float(np.median(rms_list))
        self.near_field_rms = max(200.0, med * 0.5)

    def process(self, mono: np.ndarray, second_ch: np.ndarray | None = None) -> AudioFeatures:
        # 高通
        y = self.hp.process(mono)
        rms = float(np.sqrt(np.mean(y * y) + 1e-9))
        pitch, voicing = autocorr_pitch(y, self.sr)
        centroid = spectral_centroid(y, self.sr)
        is_voiced = (voicing >= self.voicing_min) and (pitch > 0) and (rms > 50)

        near = (rms >= self.near_field_rms) and (self.centroid_min <= centroid <= self.centroid_max)

        doa = None
        if second_ch is not None and len(second_ch) == len(mono):
            ref = self.hp.process(second_ch.astype(np.float32))
            # 注意: 为避免滤波器状态串扰，真实场景应用独立滤波器；此处仅做 demo 近似
            max_tau = self.mic_spacing / 343.0
            tau, _strength = gcc_phat(y, ref, self.sr, max_tau=max_tau)
            # 角度：假设两 mic 水平对齐，θ = asin(c·τ / d)
            ratio = max(-1.0, min(1.0, 343.0 * tau / self.mic_spacing))
            doa = float(np.degrees(np.arcsin(ratio)))

        return AudioFeatures(
            rms=rms,
            is_voiced=is_voiced,
            spec_centroid=centroid,
            pitch_hz=pitch,
            doa_deg=doa,
            near_field_ok=near,
        )
