"""
auto_tune.py - 基于采集的正/负样本自动更新阈值与门权重
========================================================

新版采集只落一份 summary.json（聚合好的统计），无原始音视频。
本脚本读取每次采集会话的 summary，跨会话再聚合成正/负总体画像，
用命中率差做门权重缩放，用分位数交叠判定做阈值。

最终产物：demo/config/thresholds.json，包含
  - 标量阈值 (min_snr_db / near_field_rms / face_area_min / lip_motion_thresh)
  - 分档 profile 的 weights 覆盖

用法：
    python auto_tune.py
    python auto_tune.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob
from typing import Iterable

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(HERE, "dataset")
CONFIG_DIR = os.path.join(HERE, "config")
CONFIG_PATH = os.path.join(CONFIG_DIR, "thresholds.json")


def _iter_summary_files(label: str) -> Iterable[str]:
    pattern = os.path.join(DATASET_DIR, label, "*", "summary.json")
    yield from sorted(glob(pattern))


def _load_summaries(label: str) -> list[dict]:
    items: list[dict] = []
    for path in _iter_summary_files(label):
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict) and not obj.get("empty"):
                items.append(obj)
        except (OSError, json.JSONDecodeError):
            continue
    return items


# ---------- 跨会话聚合 ---------- #
def _mean_rate(items: list[dict], key: str) -> float:
    if not items:
        return 0.0
    return float(np.mean([float(it.get(key, 0.0)) for it in items]))


def _pool_stat(items: list[dict], key: str, sub: str) -> float:
    """从多个 summary 的 stats dict 取某分位数，按帧数加权平均。"""
    values: list[float] = []
    weights: list[float] = []
    for it in items:
        stat = it.get(key)
        if not isinstance(stat, dict):
            continue
        v = stat.get(sub)
        if v is None:
            continue
        values.append(float(v))
        weights.append(max(1.0, float(it.get("active_frames") or it.get("frames") or 1)))
    if not values:
        return 0.0
    return float(np.average(values, weights=weights))


# ---------- 阈值建议 ---------- #
def _suggest(pos_low: float, neg_high: float,
              lo: float, hi: float,
              fallback: float | None = None) -> float | None:
    """给定正样本 p15 与负样本 p90，取能尽量分离二者的阈值。
    样本都是 0 时返回 fallback。结果裁剪到 [lo, hi]。
    """
    if pos_low <= 0 and neg_high <= 0:
        return fallback
    if pos_low > 0 and neg_high > 0:
        th = (pos_low + neg_high) / 2.0
    elif pos_low > 0:
        th = pos_low
    else:
        th = neg_high
    return float(round(max(lo, min(hi, th)), 3))


def _fmt(x: float) -> str:
    return f"{x:7.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="仅打印建议，不写入 config/thresholds.json")
    args = ap.parse_args()

    pos = _load_summaries("positive")
    neg = _load_summaries("negative")

    if not pos and not neg:
        print("[auto_tune] 未找到任何 summary.json，请先在 GUI 中采集正/负样本。")
        print(f"  期望目录: {DATASET_DIR}/positive/*  与  .../negative/*")
        return 1

    print(f"[auto_tune] 样本会话数  positive={len(pos)}  negative={len(neg)}")

    # ==== 1) 门命中率（直接由 summary 里的 *_rate 跨会话平均） ==== #
    gates = ("voiced", "is_speech", "face", "gaze", "lip",
              "near_field_audio", "near_field_visual", "av_sync")
    rate_keys = {g: f"{g}_rate" for g in gates}
    pos_rates = {g: _mean_rate(pos, rate_keys[g]) for g in gates}
    neg_rates = {g: _mean_rate(neg, rate_keys[g]) for g in gates}

    print("[auto_tune] 门命中率（pos vs neg）：")
    for g in gates:
        J = pos_rates[g] - neg_rates[g]
        print(f"    {g:18s} pos={pos_rates[g]*100:5.1f}%   neg={neg_rates[g]*100:5.1f}%"
              f"   J={J*100:+5.1f}%")

    # ==== 2) 标量阈值建议（用 summary 已聚合好的分位数） ==== #
    print("[auto_tune] 特征分位（仅活跃帧）：")
    fields = [("rms", 150.0, 3000.0, "near_field_rms"),
               ("snr", 1.0,   20.0,   "min_snr_db"),
               ("face_area", 0.005, 0.20, "face_area_min"),
               ("lip_std",   0.5,   10.0, "lip_motion_thresh")]
    suggestions: dict[str, float] = {}
    for key, lo, hi, cfg_name in fields:
        p_low = _pool_stat(pos, key, "p15")
        n_high = _pool_stat(neg, key, "p90")
        p_med = _pool_stat(pos, key, "p50")
        n_med = _pool_stat(neg, key, "p50")
        print(f"    {key:10s} pos.p15={_fmt(p_low)}  pos.p50={_fmt(p_med)}  "
              f"neg.p50={_fmt(n_med)}  neg.p90={_fmt(n_high)}")
        th = _suggest(p_low, n_high, lo, hi, fallback=None)
        if th is not None:
            suggestions[cfg_name] = th

    if not suggestions:
        print("[auto_tune] 样本不足以产生阈值建议。")
        return 2

    print("[auto_tune] 建议阈值：")
    for k, v in suggestions.items():
        print(f"    {k:18s} = {v:.3f}")

    # ==== 3) 门权重：基于判别力 J 缩放默认权重 ==== #
    DEFAULT_PROFILES = {
        "relaxed": {"weights": {"gaze": 0.9, "lip": 0.8, "near_field": 0.7,
                                  "av_sync": 0.6, "snr": 0.5, "doa": 0.3}},
        "strict":  {"weights": {"gaze": 1.8, "doa": 1.2, "av_sync": 1.2}},
    }

    def _jvalue(gate: str) -> float:
        if gate == "near_field":
            ja = pos_rates["near_field_audio"] - neg_rates["near_field_audio"]
            jv = pos_rates["near_field_visual"] - neg_rates["near_field_visual"]
            return max(ja, jv)
        if gate == "snr":
            return pos_rates["voiced"] - neg_rates["voiced"]
        if gate == "doa":
            return 0.0
        return pos_rates.get(gate, 0.0) - neg_rates.get(gate, 0.0)

    profile_updates: dict[str, dict] = {}
    for prof, pdef in DEFAULT_PROFILES.items():
        new_weights: dict[str, float] = {}
        for gate, base_w in pdef["weights"].items():
            J = _jvalue(gate)
            scale = max(0.4, min(1.8, 1.0 + 1.0 * J))
            w = round(float(base_w) * scale, 3)
            new_weights[gate] = max(0.1, min(3.0, w))
        profile_updates[prof] = {"weights": new_weights}

    print("[auto_tune] 门权重更新（基于命中率差 J）：")
    for prof, cfg in profile_updates.items():
        diffs = []
        for k, v in cfg["weights"].items():
            base = DEFAULT_PROFILES[prof]["weights"][k]
            diffs.append(f"{k}:{base:.2f}→{v:.2f}")
        print(f"    [{prof}] " + ", ".join(diffs))

    if args.dry_run:
        print("[auto_tune] --dry-run: 不写文件。")
        return 0

    # ==== 4) 写入/合并到 config/thresholds.json ==== #
    os.makedirs(CONFIG_DIR, exist_ok=True)
    existing: dict = {}
    if os.path.isfile(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f) or {}
        except Exception:
            existing = {}
    existing.update(suggestions)
    prof_root = existing.get("profiles") or {}
    for prof, cfg in profile_updates.items():
        merged = prof_root.get(prof) or {}
        cur_weights = dict((merged.get("weights") or {}))
        cur_weights.update(cfg["weights"])
        merged["weights"] = cur_weights
        prof_root[prof] = merged
    existing["profiles"] = prof_root

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f"[auto_tune] 已写入 {CONFIG_PATH}")
    print("[auto_tune] 下次点击 GUI ▶ 开始 即会使用这些阈值与权重。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
