# Wake Demo (Windows)

基于 `唤醒模块实现方案-v1.0.md` 的可运行 Python 原型，演示：

- 持续麦克风采集（16 kHz 单声道）
- 2 s 环形缓冲（命中后回灌，解决首句截断）
- WebRTC VAD 门控
- openWakeWord 预训练 KWS（`hey_jarvis` / `alexa` / `hey_mycroft`）
- 仲裁器：动态阈值（随 SNR）、TTS 播报期抑制、连续误触发冷却
- 唤醒事件后将回灌音频保存为 `wake_*.wav`（模拟送入 ASR）

## 环境要求
- Windows 10/11
- Python 3.9 ~ 3.11（openWakeWord 对 3.12 的 onnxruntime 支持偶有问题，建议 3.10/3.11）
- 一个可用麦克风

## 安装
```powershell
cd e:\1_keyi\wakeup\demo
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -c "import openwakeword; openwakeword.utils.download_models()"
```

## 运行
```powershell
python wake_demo.py
```

启动后对着麦克风说 **"Hey Jarvis"**（或 "Alexa"），命令行会打印：
```
[KWS] hey_jarvis=0.87 thresh=0.50 snr=24.3dB vad=True -> ok
>>> WAKE! word=hey_jarvis score=0.87 | replay 1.50s audio from ring buffer
    saved wake_1713690000.wav
```

## 可选参数
```powershell
python wake_demo.py --wakewords hey_jarvis alexa --base-thresh 0.5 --device 1
```
- `--device`：输入设备索引，`python wake_demo.py --list-devices` 查看
- `--base-thresh`：KWS 基础阈值（默认 0.5）
- `--no-vad-gate`：关闭 VAD 门控（调试用）

## 与实现方案的对应
| 方案章节 | 代码位置 |
|---|---|
| §3.2 环形缓冲 2 s | `RingBuffer` |
| §3 VAD 门控 | `webrtcvad.Vad` 调用 |
| §4 KWS 两级（本 demo 只接入预训练单级） | `oww.predict` |
| §4.3 SNR 自适应动态阈值 + 冷却 | `NoiseTracker` + `WakeArbiter.dynamic_threshold` |
| §5.3 TTS 播报期抑制 | `WakeArbiter.tts_playing` 标志 |
| §6 回灌 | 触发时 `ring.read_last(1.5)` |
| §7 反馈 | 控制台打印 + wav 保存（占位） |

## Demo 的简化
- 仅单 mic，无 AEC/BF/DOA——桌面强噪声下可能误触发，真实产品需集成 WebRTC APM 或阵列算法
- 无多模态路径（路径 B），因为 demo 不带摄像头注意力模块
- TTS 播报期标志为静态假设（真实系统应由播放线程回调置位）
- KWS 使用 openWakeWord 的开箱即用英文唤醒词；中文"Loona"唤醒词需自行训练

这些是**方案落地时需要补齐的工程项**，demo 本身已覆盖"采集→仲裁→承接"的完整链路骨架。
