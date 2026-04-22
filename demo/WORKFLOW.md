# Loona 唤醒模块 — 工作流程详述

> 本文档描述 `demo/` 目录下当前实现的唤醒模块，从**音频/视频输入进入程序**开始，一直到**触发唤醒 → ASR 转写 → 状态复位**的完整数据流、线程模型、各级判决与回压机制。阅读对象：后续接手的工程师、做集成/调参的同学、做评审的产品/算法方。

---

## 0. 模块总览

```
 麦克风 ──► sd.InputStream ──► audio_q ─┐
                                        │
 摄像头 ──► VisualFrontend._loop ──► VisualGates(最新快照) 
                                        │
                                        ▼
                          WakeDemo._loop (主线程)
                                        │
       ┌────────────────────────────────┼────────────────────────────────┐
       ▼                                ▼                                ▼
  [模式 A: KWS]                  [模式 C: 多模态]                    [模式 B: VAD]
  openWakeWord 分数               9 条硬门限全部命中                  语音时长+SNR
  + 动态阈值仲裁                   + SUSTAIN 7 帧                    + 仲裁器
                                        │
                                        ▼
                                 fire wake  →  _on_wake()
                                        │
                          RingBuffer.read_last(2.5s)
                                        │
                                        ▼
                      faster-whisper 后台线程转写 → emit "asr"
                                        │
                                        ▼
                             GUI 更新唤醒次数 / ASR 文本
```

三条并行线程：

| 线程 | 来源 | 职责 |
|---|---|---|
| **音频回调线程** | `sd.InputStream` 内部 | 20 ms/帧把 PCM 塞入 `audio_q` |
| **视觉采集线程** | `VisualFrontend._thread` | 抓摄像头帧，跑 Haar + 口部差分，更新 `VisualGates` |
| **主处理线程** | `WakeDemo.run → _loop` | 从 `audio_q` 取帧，查视觉快照，跑特征、门限、仲裁，触发 wake |
| **ASR 后台线程**（短生命） | `_transcribe_async._job` | 每次唤醒起一条，用 faster-whisper 转写，完成后 emit，线程退出 |
| **GUI 主循环** | Tkinter `mainloop` | 每 60 ms 从 `event_q` 取事件刷新 UI，每 100 ms 抓一次预览帧 |

---

## 1. 音频输入路径（从麦克风到特征）

### 1.1 采集与上抛
- 设备由 GUI 选择（`sd.query_devices()` 过滤 `max_input_channels>0`）。
- `sd.InputStream` 配置：`samplerate=16000`、`dtype=int16`、`blocksize=320`（= 20 ms）、`latency="high"`、`channels=2`（失败回退到 1）。
- 回调 `_audio_cb(indata, frames, ...)` 在 **sounddevice 内部线程** 触发：
  - 把双通道拆成 `(mono, ref)` 两个 `int16` 数组（单通道时 `ref=None`）。
  - `audio_q.put_nowait((mono, ref))`；**队列满时丢最老帧**（保实时，不阻塞回调）。

### 1.2 环形缓冲（RingBuffer）
- 主线程每次从 `audio_q` 取到帧，立刻 `self.ring.write(frame)`。
- `RingBuffer` 容量 `RING_SECONDS=3 秒 × 16kHz = 48000 samples`，用于唤醒命中后回灌最后 `REPLAY_SECONDS=2.5 秒` 给 ASR。

### 1.3 轻量 EnergyVAD
- `EnergyVAD.is_speech(frame)`：自适应噪声底（只在非语音时用 `alpha=0.02` 更新）+ 过零率约束 `0.01<zc<0.35`；判决式 `rms > noise_floor × 10^(8dB/20)`。
- 主要用于：LISTENING 状态超时判断、噪声底更新、多模态模式的诊断打印触发。

### 1.4 噪声底 & SNR 跟踪（NoiseTracker）
- 非语音帧时 `noise_rms ← (1-α)·noise_rms + α·frame_rms`，`α=0.02`。
- `snr_db(frame_rms) = 20·log10(frame_rms / noise_rms)`。
- SNR 既用于 KWS 动态阈值，也作为 VAD/多模态的硬门之一。

### 1.5 音频前端（AudioFrontend.process）

`perception_audio.py`，每 20 ms 执行一次，产出 `AudioFeatures`：

| 步骤 | 产出 | 作用 |
|---|---|---|
| **1 阶高通 80 Hz** | HPF 后波形 `y` | 砍风扇/空调低频轰鸣，避免把稳态噪声误判为人声 |
| **RMS** | `rms` | 近场能量门限、VAD、诊断 |
| **自相关基频** | `pitch_hz`, `voicing∈[0,1]` | 在 80–400 Hz 内找自相关峰，人声区间；`voicing≥0.22 且 pitch>0 且 rms>80` → `is_voiced=True` |
| **频谱重心** | `spec_centroid`（Hz） | 远场高频被空气吸收 → 重心偏低；近讲一般在 300–3500 Hz |
| **GCC-PHAT DOA**（需双通道） | `doa_deg` 或 `None` | 相位互相关估计到达角，用于方向硬门（`|doa|≤30°`） |
| **近场判据** | `near_field_ok = rms≥NF_RMS 且 centroid∈[300,3500]` | 用户近讲 vs 同事远讲的一次性区分 |

参数默认：`near_field_rms=400`（GUI 可调），`voicing_min=0.22`，`centroid∈[300,3500]`，`mic_spacing=0.04 m`。

---

## 2. 视频输入路径（摄像头到视觉门）

`perception_visual.py`，完全 OpenCV + Haar，**不**依赖 mediapipe/dlib。

### 2.1 采集
- `VideoCapture(index, CAP_DSHOW)` 优先，失败再降级到默认后端。
- 参数：`480×360 @ 20fps`。**不**设 `CAP_PROP_BUFFERSIZE`（在 Windows DShow 下会挂住采集线程）。
- 独立后台线程 `_loop`，死循环抓帧：
  - `ok, frame = cap.read()` 失败则 `sleep(30ms)` 继续；任何异常捕获后不崩溃。

### 2.2 人脸检测（跳帧）
- Haar cascade `haarcascade_frontalface_default.xml`，`scaleFactor=1.15, minNeighbors=3, minSize=6% W`。
- **每 2 帧跑一次**（`self._frames % 2 == 0`），其余帧复用 `last_faces`，把 CPU 压到约一半。
- 选面积最大的人脸作为主目标。

### 2.3 四项视觉子门

| 子门 | 计算 | 阈值 |
|---|---|---|
| `face_present` | 是否检测到脸 | 有脸即 True |
| `near_field` | 脸框面积 / 画面面积 | `≥0.02` (≈2%) |
| `gaze_aligned` | 脸中心水平偏画面中轴的归一偏差 | `|offset| ≤ 0.32` |
| `lip_moving` | 口部 ROI(脸下 58–90%, 左右 22–78%) 缩放到 60×30 后的**帧间差均值**序列 | 最近 ~1s 内 `std ≥ 2.0` **或** `max ≥ 3.6` |

### 2.4 快照与预览
- `_latest_gates: VisualGates` 持锁更新；`snapshot()` 返回最新一份拷贝。
- `preview()` 返回带状态叠加（face/gaze/near/lip 文字+矩形）的 BGR 帧，给 GUI 预览使用。
- 主线程每 20 ms 拿一次快照，和音频特征对齐到同一帧。

---

## 3. 主循环与融合（WakeDemo._loop）

每取到一个 20 ms 音频帧，顺序执行：

1. `self.ring.write(frame)`
2. `is_speech = self.vad.is_speech(frame)`
3. 更新噪声底 → `snr = self.noise.snr_db(rms)`
4. `feats = self.audio_fe.process(frame, ref)` — 音频特征
5. 若处于 `LISTENING` 且超时 8s → 回 `IDLE`
6. 按 `self.mode` 分发到 `_process_kws / _process_multimodal / _process_vad`

下面展开三条路径。

---

## 4. 模式 A：KWS（唤醒词）路径

走的是 openWakeWord（ONNX），典型用法。

### 4.1 累积 80ms chunk
- openWakeWord 要求 1280 samples（80 ms）窗口。
- 把 20 ms 的帧追加到 `kws_accum`，每凑够 80 ms 跑一次 `self.kws.predict(chunk)`，取分数最高的词 `(name, score)`。

### 4.2 WakeArbiter 仲裁
- **动态阈值**：基础阈值 `base_thresh`（GUI 可调，默认 0.35）
  - `snr ≥ 20dB`：不加
  - `10 ≤ snr < 20`：+0.15
  - `snr < 10`：+0.30
  - TTS 播报中：+0.10
  - 冷却期：+0.10（短时间内被触发 3 次后进入，持续 5 min）
  - 封顶 0.95
- **硬否决**：
  - 静默异常：`snr < -3 且 score ≥ thresh` → 判 `veto:anomaly-silent`
  - VAD 未激活：`score < thresh + 0.10` → 判 `no-vad`
- 通过后 `fire=True`，调 `_fire_wake`。

---

## 5. 模式 C：多模态（默认路径，方案 §5.2 路径 B 的正式实现）

这是真正打业务价值的路径。**9 条硬门限全部命中** **并且** 持续 **7 帧（140 ms）** 才触发。

### 5.1 音频意图证据（前 3 条门）

- `voiced`：`feats.is_voiced`
- `snr`：`snr ≥ min_snr_db`（默认 6 dB，GUI 可调）
- `speech_run`：累计连续语音时长 `≥ min_speech_ms`（默认 400 ms）
  - 累计规则带 **hangover 240 ms**：出现浊音后即使短暂静音，继续以 **半速率** 累加 12 帧；长静音 ≥700 ms 才清零。这是为了避免"嗯…你好"这种开头抖动把计数打断。

### 5.2 空间定位（第 4–5 条门）

- `near_field = feats.near_field_ok OR (视觉 face_area ≥ 2%)` — 音频/视觉任一通过即可
- `doa`：双通道下 `|feats.doa_deg| ≤ doa_cone_deg`（默认 30°）；单通道时该门强制 True（真实产品应强制双通道）

### 5.3 视觉生理证据（第 6–8 条门）

- `face`：`vg.face_present`
- `gaze`：`vg.gaze_aligned`
- `lip`：`vg.lip_moving`

（若 `--no-strict-multimodal` 且视觉完全不可用，这三门自动放行——退化为 VAD+严格音频模式。默认开启 strict。）

### 5.4 唇-声同步（第 9 条门，关键反同事语音）

滑动窗口 `_av_window` 最多保留最近 40 帧（800 ms）的 `(voiced_now, lip_now)` 对：

```
coincide = 同时 voiced 且 lip_moving 的帧数
voiced_cnt = 音频语音帧数
lip_cnt = 唇动帧数
av_sync = (voiced_cnt≥4) AND (lip_cnt≥4) AND (coincide ≥ max(3, 0.35·voiced_cnt))
```

**这是区分"用户自己说话"和"同事说话+我只是望着屏幕"的核心**——同事说话时麦克风有声音（voiced），但用户嘴没动（lip=0），`coincide=0` 无法达标。

### 5.5 持续判定（SUSTAIN）

```
all_pass = 以上 9 条全部 True
if all_pass: _sustain_pass_frames += 1 else: = 0
sustained  = _sustain_pass_frames ≥ 7   (140 ms)
```

### 5.6 防抖 / 冷却 / TTS 抑制
- 自上次触发 `last_fire` 不足 **1.5 s** → `veto:debounce`
- `arbiter.tts_playing = True` 且 `snr < min_snr + 3dB` → `veto:tts-playing`（防回环自激）

### 5.7 诊断与事件
- 每 1.5 s（在有音频活动时）打印一行 `[GATES]`，列出每条门的 Y/N 以及关键特征值。
- 每 20 ms emit 一条 `tick` 事件给 GUI，内容包括：score/threshold/snr/vad/reason/rms/centroid/pitch/doa/gates/visual 等。GUI 据此渲染进度条、9 盏硬门限灯、音频诊断行。

### 5.8 触发
全部通过后：
```python
self.last_fire = now
self.speech_run_ms = 0.0
self._sustain_pass_frames = 0
self._fire_wake("(multimodal)", pseudo_score)
```

---

## 6. 模式 B：纯 VAD（无唤醒词）路径

简化版 §8.1 路径 B。只看"连续语音时长 + SNR"：

1. `is_speech` 时 `speech_run_ms += 20`，`silence` 累计 ≥ 300 ms 则清零。
2. `pseudo_score = min(1.0, speech_run_ms / min_speech_ms)`
3. 丢给 `WakeArbiter.decide(pseudo_score, snr, vad)` 借用同一套动态阈值/TTS/冷却逻辑。
4. 额外硬否决：
   - `snr < min_snr_db` → `veto:snr<XdB`
   - `now - last_fire < 1.5s` → `veto:debounce`
   - `speech_run_ms < min_speech_ms` → `below`

这个模式在 demo 里主要用来"无视觉对照组"，不是业务默认路径。

---

## 7. 触发之后：_fire_wake → _on_wake → ASR

### 7.1 状态切换
```python
self.state = "LISTENING"
self.listen_start = time.time()
self.arbiter.register_hit(self.listen_start)
self._emit("state", state="LISTENING", reason="wake")
```
- `arbiter.register_hit`：记录命中时刻；5 s 内 ≥3 次 → 进入冷却，`cooldown_bonus=0.10` 持续 5 min。

### 7.2 回灌首句
```python
history = self.ring.read_last(REPLAY_SECONDS=2.5s)
dur = len(history) / 16000
self.wake_count += 1
self._emit("wake", word=..., score=..., duration=dur, count=self.wake_count)
```
注意：**不落地 wav 文件**，直接把 numpy 数组传给 ASR。

### 7.3 faster-whisper 转写（后台线程）
- 首次调用懒加载 `WhisperModel("tiny", device="cpu", compute_type="int8")`，有约 2–4 s 冷启动。
- 调用参数：
  ```python
  model.transcribe(audio_f32, language="zh", beam_size=1,
                   vad_filter=True, condition_on_previous_text=False)
  ```
- 成功：`emit("asr", word=..., score=..., text=transcript)`
- 失败：`emit("asr", ..., text="", error=str(e))`

### 7.4 LISTENING 超时
- 主循环每 20 ms 检查：`time.time() - listen_start > LISTEN_TIMEOUT_S=8s` → 回 `IDLE`，emit `state=IDLE reason=timeout`。
- 真实产品会在这个窗口里把首句送 ASR 后等后端对话完成再回 IDLE；demo 仅做定时回退。

---

## 8. GUI 事件链（wake_gui.py）

所有来自引擎的事件都通过 `on_event(ev, payload)` 回调 → `event_q.put` 放入线程安全队列。GUI 主线程每 60 ms 调一次 `_drain_events` 消费：

| 事件 | GUI 反应 |
|---|---|
| `started` | 状态灯置 IDLE，日志打印模式/通道数/视觉开关 |
| `tick` | 刷新 KWS 分数/动态阈值/SNR 进度条、VAD/Word/Reason、RMS/centroid/pitch/DOA 诊断行、9 盏硬门限灯颜色 |
| `wake` | 唤醒计数 +1，日志红色 `>>> WAKE #N!`，ASR 面板追加 "识别中…" |
| `asr` | 替换 ASR 面板最后一行为识别文本（或错误提示） |
| `state` | 状态灯切 `LISTENING`（绿）/`IDLE`（灰），日志记录切换原因 |
| `interrupt_hint` | 日志黄字提示"LISTENING 中再触发 → 交给打断模块" |
| `stopped` | 状态灯 STOPPED，`btn_start` 恢复可点 |

GUI 另有一个独立的 100 ms 定时器 `_refresh_preview`：
- 若引擎的 `visual` 存在，调 `visual.preview()` 拿到带标注的 BGR 帧。
- `cv2.cvtColor → BGR2RGB → resize(宽=260) → PIL.Image → ImageTk.PhotoImage`，塞到右上角摄像头预览 Label。

---

## 9. 关键时间常量一览

| 常量 | 值 | 位置 | 作用 |
|---|---|---|---|
| `VAD_FRAME_MS` | 20 ms | `wake_demo.py` | 音频主处理粒度 |
| `KWS_CHUNK_LEN` | 1280 samples (80 ms) | 同上 | openWakeWord 推理窗 |
| `RING_SECONDS` | 3 s | 同上 | 回灌缓冲容量 |
| `REPLAY_SECONDS` | 2.5 s | 同上 | 唤醒后送 ASR 的历史片段长度 |
| `LISTEN_TIMEOUT_S` | 8 s | 同上 | LISTENING 自动回 IDLE |
| `SUSTAIN_FRAMES` | 7 帧 (140 ms) | `_process_multimodal` | 多模态连续全绿要求 |
| hangover | 12 帧 (240 ms) | 同上 | 浊音尾拖 |
| silence_reset | 700 ms | 同上 | 多模态 speech_run 清零阈值 |
| silence_reset (vad) | 300 ms | `_process_vad` | VAD 模式 speech_run 清零阈值 |
| debounce | 1.5 s | 多模态/vad | 两次触发最小间隔 |
| cooldown 窗口 | 5 s / ≥3 次 → +0.10 持续 5 min | `WakeArbiter` | 防乱唤醒 |
| AV 同步窗 | 800 ms (40×20ms) | `_av_window` | 唇-声共现统计 |
| 视觉跳帧 | 每 2 帧 | `VisualFrontend._loop` | Haar CPU 优化 |

---

## 10. 一次"用户说话"事件的完整时间线（多模态，理想情况）

```
 t=0ms    用户张嘴，嘴部像素开始变化
 t~50ms   摄像头线程：lip_std 首次越阈
 t~60ms   用户发出第一个浊音；音频帧 is_voiced=True
          speech_run_ms 开始累计
 t~100ms  GCC-PHAT 算出 DOA≈+5°，gates.doa=Y
          face/gaze/lip/near_field/voiced/snr 全 Y
          _sustain_pass_frames = 1
 t~240ms  speech_run_ms ≥ 400? 尚未（累计 ≈180ms，含 hangover）
 t~400ms  speech_run_ms 达 400ms，gates.speech_run=Y
          av_sync 窗口内 coincide≥4，gates.av_sync=Y → all_pass=True
          _sustain_pass_frames 开始递增
 t~540ms  _sustain_pass_frames=7 → sustained=True → FIRE
          state=LISTENING，emit wake(count=N, duration≈2.5s)
          启动 ASR 后台线程
 t~540ms  ring.read_last(2.5s) 取 t-1960ms ~ t+540ms 这段 PCM
 t~1~4s   faster-whisper 后台转写完成 → emit asr(text="...")
          GUI 替换"识别中..."为真正文本
 t=8.54s  无新触发 → LISTENING 超时回 IDLE
```

---

## 11. 主要失败模式与对应防线

| 可能误触发场景 | 防线 |
|---|---|
| 同事在背后讲话 | `av_sync`（用户嘴没动）、`near_field`（同事远场 RMS 低）、`doa`（方向偏）、`gaze`（用户没看屏幕）|
| 键盘/爆音瞬态 | `voiced`（非浊音）、`speech_run`（<400 ms）、`SUSTAIN` 140 ms |
| 空调/风扇稳态 | HPF 80 Hz + `noise_tracker` 自适应底 + SNR 门 |
| TTS 自激回环 | `arbiter.tts_playing` 时阈值 +0.10，多模态下 SNR 加严 3 dB |
| 用户多次说话误连触 | `debounce 1.5s` + `cooldown`（5 s 内 3 次后阈值 +0.10 持续 5 min）|
| 视觉丢失 | strict 模式下直接否决；非 strict 下放行视觉门但记录日志 |

---

## 12. 可调参数速查（GUI / CLI）

| GUI 控件 | CLI | 默认 | 说明 |
|---|---|---|---|
| 设备 | `--device N` | 系统默认输入 | sounddevice 输入索引 |
| 阈值 | `--base-thresh` | 0.35 | KWS 基础阈值 / VAD 证据门限 |
| 语音≥(ms) | `--min-speech-ms` | 400 | 多模态/VAD 最小连续语音时长 |
| SNR≥(dB) | `--min-snr-db` | 6.0 | 最小触发 SNR |
| 近场 RMS | `--near-field-rms` | 400 | 音频近场能量门限 |
| DOA ±° | `--doa-cone-deg` | 30 | 到达角容忍锥 |
| 声道 | `--audio-channels` | 2 | ≥2 启用 DOA |
| 摄像头索引 | `--camera-index` | 0 | OpenCV 设备号 |
| 禁用摄像头 | `--no-camera` | False | 关摄像头 |
| 关VAD门控 | `--no-vad-gate` | False | 仅 KWS 调试 |
| 模拟 TTS | （GUI 勾选） | False | 测试回环抑制 |
| — | `--asr-model` | tiny | faster-whisper 模型档位 |

---

## 13. 文件与职责

- `wake_demo.py` — 引擎核心：RingBuffer、EnergyVAD、NoiseTracker、WakeArbiter、WakeDemo (三种模式、ASR、事件分发)。
- `perception_audio.py` — HPF/基频/重心/GCC-PHAT/近场判据。
- `perception_visual.py` — 摄像头线程、Haar、四视觉子门、预览帧渲染。
- `wake_gui.py` — Tkinter 封装；订阅引擎事件；参数面板、实时监测、9 灯硬门限、预览、日志、ASR 面板、唤醒计数。
- `requirements.txt` — 依赖清单。

---
