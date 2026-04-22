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


class VisualFrontend:
    def __init__(self, camera_index: int = 0,
                 face_area_min: float = 0.02,          # 脸占画面 ≥2%
                 gaze_offset_max: float = 0.32,        # 脸中心偏画面中轴 ≤32%
                 lip_motion_thresh: float = 2.0):      # 像素差的帧间标准差
        self.camera_index = camera_index
        self.face_area_min = face_area_min
        self.gaze_offset_max = gaze_offset_max
        self.lip_motion_thresh = lip_motion_thresh
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest_gates = VisualGates(available=False, face_present=False,
                                         gaze_aligned=False, near_field=False,
                                         lip_moving=False)
        self._latest_preview: np.ndarray | None = None
        self._cap = None
        self._cascade = None
        self._mouth_history: deque = deque(maxlen=25)  # ~1s@25fps
        self._prev_mouth_gray: np.ndarray | None = None
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
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
                face_area_ratio = (w * h) / float(W * H)
                near_field = face_area_ratio >= self.face_area_min
                center_offset = ((x + w / 2) - W / 2) / (W / 2)
                gaze_aligned = abs(center_offset) <= self.gaze_offset_max
                # 口部 ROI: 脸下 40%~85%、左右 25%~75%
                my1 = y + int(0.58 * h)
                my2 = y + int(0.90 * h)
                mx1 = x + int(0.22 * w)
                mx2 = x + int(0.78 * w)
                mx1, my1 = max(0, mx1), max(0, my1)
                mx2, my2 = min(W, mx2), min(H, my2)
                if mx2 > mx1 and my2 > my1:
                    mouth = gray[my1:my2, mx1:mx2]
                    mouth = cv2.resize(mouth, (60, 30))
                    if self._prev_mouth_gray is not None and self._prev_mouth_gray.shape == mouth.shape:
                        diff = cv2.absdiff(mouth, self._prev_mouth_gray)
                        self._mouth_history.append(float(diff.mean()))
                    self._prev_mouth_gray = mouth.copy()
                    cv2.rectangle(frame, (mx1, my1), (mx2, my2), (0, 120, 255), 1)
                    if len(self._mouth_history) >= 6:
                        arr = np.array(self._mouth_history)
                        # 使用 std 或 max 两者之一达标即认为嘴在动，
                        # 兼顾说话幅度较小但存在瞬时变化的情况
                        lip_std = float(arr.std())
                        lip_max = float(arr.max())
                        lip_moving = (lip_std >= self.lip_motion_thresh
                                      or lip_max >= self.lip_motion_thresh * 1.8)
            else:
                self._mouth_history.clear()
                self._prev_mouth_gray = None

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
            )
            with self._lock:
                self._latest_gates = gates
                self._latest_preview = frame
