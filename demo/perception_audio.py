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
        is_voiced = (voicing >= self.voicing_min) and (pitch > 0) and (rms > 80)

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
