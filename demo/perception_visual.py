"""
视觉前端 - 注视 / 近场 / 唇动
-----------------------------
使用 OpenCV（不依赖 mediapipe/dlib），基于 Haar cascade + 口部 ROI 像素方差：

  - 人脸检测（opencv haarcascade_frontalface_default）
  - 注视：正面人脸被检出 + 脸框中心在画面水平中轴附近（demo 近似，
          真实产品应用 head pose estimation + gaze）
  - 近场：人脸框面积 / 画面面积 ≥ 阈值（脸越大越近）
  - 唇动：脸下半部 ROI 帧间差方差在 0.8~1.2s 窗口内显著波动

线程：独立 grab 线程持续采集摄像头帧，主线程查 snapshot 的最新门限状态，
避免阻塞音频链路。
"""

from __future__ import annotations

import threading
import time
import sys
from collections import deque
from dataclasses import dataclass

import numpy as np

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False


@dataclass
class VisualGates:
    available: bool
    face_present: bool
    gaze_aligned: bool
    near_field: bool
    lip_moving: bool
    # 诊断
    face_area_ratio: float = 0.0
    face_center_offset: float = 0.0  # [-1,1]，0=居中
    lip_motion_std: float = 0.0
    fps: float = 0.0
    # 环境感知：人数
    face_count: int = 0


class VisualFrontend:
    def __init__(self, camera_index: int = 0,
                 face_area_min: float = 0.02,          # 脸占画面 ≥2%
                 gaze_offset_max: float = 0.32,        # 脸中心偏画面中轴 ≤32%
                 lip_motion_thresh: float = 1.3,       # 基线扣除后的差分均值阈值
                 lip_hangover_frames: int = 8):        # 命中后尾拖 N 帧（~400ms@20fps）
        self.camera_index = camera_index
        self.face_area_min = face_area_min
        self.gaze_offset_max = gaze_offset_max
        self.lip_motion_thresh = lip_motion_thresh
        self.lip_hangover_frames = int(lip_hangover_frames)
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest_gates = VisualGates(available=False, face_present=False,
                                         gaze_aligned=False, near_field=False,
                                         lip_moving=False)
        self._latest_preview: np.ndarray | None = None
        self._cap = None
        self._cascade = None
        # 唇动差分历史：存"基线扣除后"的数值
        self._mouth_history: deque = deque(maxlen=25)  # ~1s@25fps
        # 原始差分历史：用来估计静止基线
        self._mouth_raw: deque = deque(maxlen=25)
        self._prev_mouth_gray: np.ndarray | None = None
        # 人脸 bbox EMA（降低 Haar 抖动对 ROI 的影响）
        self._face_ema: tuple[float, float, float, float] | None = None
        self._face_ema_alpha = 0.4
        # 命中尾拖
        self._lip_hangover_left = 0
        self._t0 = time.time()
        self._frames = 0

    def start(self):
        if not HAVE_CV2:
            return False
        try:
            self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(self.camera_index)
            if not self._cap.isOpened():
                return False
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            self._cap.set(cv2.CAP_PROP_FPS, 20)
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._cascade = cv2.CascadeClassifier(path)
            if self._cascade.empty():
                return False
        except Exception:
            return False
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def snapshot(self) -> VisualGates:
        with self._lock:
            return self._latest_gates

    def preview(self) -> np.ndarray | None:
        with self._lock:
            return None if self._latest_preview is None else self._latest_preview.copy()

    def _loop(self):
        last_faces = None  # 缓存上一次检测结果（None = 尚未检测过）
        while self._running and self._cap is not None:
            try:
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    time.sleep(0.03)
                    continue
                self._frames += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                H, W = gray.shape
                # 每 2 帧跑一次 Haar（CPU 大头），其余帧复用上次结果
                need_detect = (last_faces is None) or (self._frames % 2 == 0)
                if need_detect:
                    faces = self._cascade.detectMultiScale(
                        gray, scaleFactor=1.15, minNeighbors=3,
                        minSize=(int(W * 0.06), int(W * 0.06)),
                    )
                    last_faces = faces
                else:
                    faces = last_faces
                face_present = len(faces) > 0
            except Exception as e:
                print(f"[VISUAL-LOOP-ERR] {e}", file=sys.stderr)
                time.sleep(0.05)
                continue
            gaze_aligned = False
            near_field = False
            lip_moving = False
            face_area_ratio = 0.0
            center_offset = 0.0
            lip_std = 0.0

            if face_present:
                # 选最大的脸
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                # bbox EMA 平滑：降低 Haar 逐帧抖动带来的 ROI 错位
                if self._face_ema is None:
                    self._face_ema = (float(x), float(y), float(w), float(h))
                else:
                    a = self._face_ema_alpha
                    ex, ey, ew, eh = self._face_ema
                    self._face_ema = (
                        (1 - a) * ex + a * float(x),
                        (1 - a) * ey + a * float(y),
                        (1 - a) * ew + a * float(w),
                        (1 - a) * eh + a * float(h),
                    )
                sx, sy, sw, sh = self._face_ema
                x, y, w, h = int(sx), int(sy), int(sw), int(sh)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
                face_area_ratio = (w * h) / float(W * H)
                near_field = face_area_ratio >= self.face_area_min
                center_offset = ((x + w / 2) - W / 2) / (W / 2)
                gaze_aligned = abs(center_offset) <= self.gaze_offset_max
                # 口部 ROI: 脸下 58%~90%、左右 22%~78%
                my1 = y + int(0.58 * h)
                my2 = y + int(0.90 * h)
                mx1 = x + int(0.22 * w)
                mx2 = x + int(0.78 * w)
                mx1, my1 = max(0, mx1), max(0, my1)
                mx2, my2 = min(W, mx2), min(H, my2)
                if mx2 > mx1 and my2 > my1:
                    mouth = gray[my1:my2, mx1:mx2]
                    mouth = cv2.resize(mouth, (60, 30))
                    # 轻微模糊降低传感器噪声对 absdiff 的影响
                    mouth_f = cv2.GaussianBlur(mouth, (3, 3), 0)
                    if (self._prev_mouth_gray is not None
                            and self._prev_mouth_gray.shape == mouth_f.shape):
                        diff = cv2.absdiff(mouth_f, self._prev_mouth_gray)
                        raw = float(diff.mean())
                        self._mouth_raw.append(raw)
                        # 静止基线：最近 1s 内的 20% 分位数（只在没说话时接近底噪）
                        if len(self._mouth_raw) >= 10:
                            baseline = float(np.percentile(self._mouth_raw, 20))
                        else:
                            baseline = 0.0
                        # 基线扣除后的"真正唇动能量"
                        eff = max(0.0, raw - baseline)
                        self._mouth_history.append(eff)
                    self._prev_mouth_gray = mouth_f.copy()
                    cv2.rectangle(frame, (mx1, my1), (mx2, my2), (0, 120, 255), 1)
                    if len(self._mouth_history) >= 8:
                        arr = np.array(self._mouth_history)
                        # 使用三路证据的 OR（任一满足即判唇动）：
                        #  1) 近 160ms 峰值 ≥ 阈值（捕捉张嘴/闭嘴的瞬时）
                        #  2) 近 500ms 均值 ≥ 阈值 × 0.7（持续说话）
                        #  3) 整体 std ≥ 阈值 × 0.8（反复起伏）
                        recent_peak = float(arr[-4:].max())
                        recent_mean = float(arr[-10:].mean())
                        lip_std = float(arr.std())
                        th = float(self.lip_motion_thresh)
                        hit = (
                            recent_peak >= th
                            or recent_mean >= th * 0.7
                            or lip_std >= th * 0.8
                        )
                        # hangover：命中后尾拖 N 帧，避免开合瞬间掉帧抖动
                        if hit:
                            self._lip_hangover_left = self.lip_hangover_frames
                        if self._lip_hangover_left > 0:
                            lip_moving = True
                            self._lip_hangover_left -= 1
            else:
                self._mouth_history.clear()
                self._mouth_raw.clear()
                self._prev_mouth_gray = None
                self._lip_hangover_left = 0
                self._face_ema = None

            # 叠加状态
            def tag(ok):
                return (0, 200, 0) if ok else (80, 80, 200)
            cv2.putText(frame, f"face={face_present}", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, tag(face_present), 1)
            cv2.putText(frame, f"gaze={gaze_aligned} ({center_offset:+.2f})",
                        (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tag(gaze_aligned), 1)
            cv2.putText(frame, f"near={near_field} ({face_area_ratio:.3f})",
                        (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tag(near_field), 1)
            cv2.putText(frame, f"lip={lip_moving} ({lip_std:.2f})",
                        (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tag(lip_moving), 1)

            now = time.time()
            fps = self._frames / max(0.1, (now - self._t0))

            gates = VisualGates(
                available=True,
                face_present=face_present,
                gaze_aligned=gaze_aligned,
                near_field=near_field,
                lip_moving=lip_moving,
                face_area_ratio=face_area_ratio,
                face_center_offset=center_offset,
                lip_motion_std=lip_std,
                fps=fps,
                face_count=int(len(faces)) if face_present else 0,
            )
            with self._lock:
                self._latest_gates = gates
                self._latest_preview = frame
