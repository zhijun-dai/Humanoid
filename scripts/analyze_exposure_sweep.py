#!/usr/bin/env python3
"""对比不同曝光档位录制的视频，给出巡线质量对照表并推荐档位。

配套 scripts/record_exposure_sweep.py 使用。

核心指标是「有中线% / conf 中位」——这是巡线的实际表现。
辅助指标解释原因：清晰度（模糊多少）、线谷底灰阶（线被洗多灰）、
平坦区噪声（缩短曝光的代价）。

用法：
    python scripts/analyze_exposure_sweep.py --dir 曝光测试
    python scripts/analyze_exposure_sweep.py --glob "曝光测试/*.mp4"
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "jetson"))

from line_detector_v1_warp import LineDetector        # noqa: E402
from camera_config import load as _load_camera        # noqa: E402

HAS_LINE_CONF = 0.15      # conf 高于此值算「有中线」


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="曝光档位对比分析")
    p.add_argument("--dir", default="曝光测试", help="录制输出目录")
    p.add_argument("--glob", default="", help="直接用 glob 指定文件（覆盖 --dir）")
    p.add_argument("--step", type=int, default=1, help="每 N 帧取一帧，加速")
    p.add_argument("--out", default="", help="结果写到此 txt（默认只打印）")
    return p.parse_args()


def exposure_tag(path: str) -> str:
    """从文件名里取曝光标记，如 expm5.mp4 -> -5，expauto.mp4 -> auto"""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"exp([a-z0-9pm]+)", stem, re.I)
    if not m:
        return stem
    t = m.group(1)
    if t.lower() == "auto":
        return "auto"
    t = t.replace("m", "-").replace("p", ".")
    return t


HALF_WIN = 80        # 线宽统计的搜索半窗（px），窗口内取局部对比度


def line_profile_stats(gray: np.ndarray):
    """下半部找赛道线，返回 (线像宽中位, 谷底灰阶中位, 局部对比度中位)

    线宽按「峰-谷半高」量，搜索限制在 ±HALF_WIN 内，避免走到别的暗物体上。
    """
    H, W = gray.shape
    widths, troughs = [], []
    k = np.ones(3, np.float32) / 3.0
    for y in range(int(H * 0.55), int(H * 0.95), 25):
        sm = np.convolve(gray[y].astype(np.float32), k, mode="same")
        for x in range(200, max(201, W - 600), 60):
            seg = sm[x:x + 400]
            if seg.size < 100:
                continue
            mn = float(seg.min())
            if mn > 140:
                continue
            xi = int(np.argmin(seg)) + x
            lo_i, hi_i = max(0, xi - HALF_WIN), min(W, xi + HALF_WIN)
            hi = float(sm[lo_i:hi_i].max())
            if hi - mn < 20:               # 局部对比度不足，不是可靠的线
                continue
            half = mn + 0.5 * (hi - mn)
            left, right = xi, xi
            while left > lo_i and sm[left] < half:
                left -= 1
            while right < hi_i - 1 and sm[right] < half:
                right += 1
            w = right - left
            if 3 <= w <= 2 * HALF_WIN:
                widths.append(w)
                troughs.append(mn)
            break
    if not widths:
        return float("nan"), float("nan")
    return float(np.median(widths)), float(np.median(troughs))


def flat_noise(gray: np.ndarray) -> float:
    """平坦区噪声：亮区内 (原图-中值滤波) 的标准差，反映增益噪声"""
    med = cv2.medianBlur(gray, 5)
    diff = np.abs(gray.astype(np.float32) - med.astype(np.float32))
    mask = med > 150
    if mask.sum() < 1000:
        mask = med > np.percentile(med, 70)
    return float(diff[mask].std()) if mask.sum() > 0 else float("nan")


def analyze(path: str, step: int, cam: dict):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ld = LineDetector(W, H, cam_height_cm=cam["mount_height_cm"],
                      cam_pitch_deg=cam["pitch_deg"], cam_vfov_deg=cam["vfov_deg"])
    lvs, confs, widths, troughs, noises, means = [], [], [], [], [], []
    i = 0
    n_frame = 0
    while True:
        ok, f = cap.read()
        if not ok or f is None:
            break
        i += 1
        if i % step:
            continue
        n_frame += 1
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        gs = cv2.resize(g, (960, int(g.shape[0] * 960 / g.shape[1])))
        lvs.append(float(cv2.Laplacian(gs, cv2.CV_64F).var()))
        means.append(float(gs.mean()))
        _, _, conf, _, _ = ld.process(f)
        confs.append(conf)
        if n_frame % 6 == 0:
            w, t = line_profile_stats(g)
            if not np.isnan(w):
                widths.append(w)
                troughs.append(t)
            noises.append(flat_noise(g))
    cap.release()
    if not lvs:
        return None
    lvs = np.array(lvs)
    confs = np.array(confs)
    return {
        "file": os.path.basename(path),
        "frames": len(lvs),
        "size": "%dx%d" % (W, H),
        "lv_med": float(np.median(lvs)),
        "lv_p25": float(np.percentile(lvs, 25)),
        "has_line": 100.0 * float(np.sum(confs > HAS_LINE_CONF)) / len(confs),
        "conf_med": float(np.median(confs)),
        "conf_p75": float(np.percentile(confs, 75)),
        "line_w": float(np.median(widths)) if widths else float("nan"),
        "trough": float(np.median(troughs)) if troughs else float("nan"),
        "noise": float(np.median(noises)) if noises else float("nan"),
        "mean": float(np.median(means)),
    }


def main() -> int:
    args = parse_args()
    if args.glob:
        files = sorted(glob.glob(args.glob))
    else:
        files = sorted(glob.glob(os.path.join(args.dir, "*.mp4")) +
                       glob.glob(os.path.join(args.dir, "*.avi")))
    if not files:
        print("没找到视频。先跑 scripts/record_exposure_sweep.py")
        return 2

    cam = _load_camera()
    out = []
    out.append("相机: vfov=%.3f 高度=%.1f 俯角=%.1f"
               % (cam["vfov_deg"], cam["mount_height_cm"], cam["pitch_deg"]))
    out.append("")
    out.append("曝光    帧数  分辨率  清晰度lv  画面亮度  线谷底灰阶  线像宽  有中线%   conf中位   conf_p75  平坦区噪声")
    out.append("-" * 118)

    rows = []
    for f in files:
        r = analyze(f, args.step, cam)
        if r is None:
            out.append("%-7s 打不开或没有帧" % exposure_tag(f))
            continue
        rows.append((exposure_tag(f), r))
        out.append("%-7s %5d  %-7s %7.1f  %7.1f   %8.1f   %6.0f  %6.0f%%    %.3f     %.3f     %.2f"
                   % (exposure_tag(f), r["frames"], r["size"], r["lv_med"], r["mean"],
                      r["trough"], r["line_w"], r["has_line"], r["conf_med"],
                      r["conf_p75"], r["noise"]))

    out.append("")
    if rows:
        # 推荐：有中线% 与 conf 中位综合最好
        def score(item):
            r = item[1]
            return r["has_line"] / 100.0 * 0.5 + r["conf_med"] * 0.5

        best = max(rows, key=score)
        out.append("推荐档位: %s  （有中线 %.0f%%，conf 中位 %.3f，清晰度 %.1f，线谷底 %.1f，噪声 %.2f）"
                   % (best[0], best[1]["has_line"], best[1]["conf_med"],
                      best[1]["lv_med"], best[1]["trough"], best[1]["noise"]))
        out.append("")
        out.append("判据：优先「有中线% + conf」；相邻档位接近时取曝光更长的那档（信噪比更好）。")
        out.append("若某档 有中线% 明显更高，说明缩短曝光确实把赛道线救回来了。")
        out.append("")

        # 换场地提醒：曝光管「钉住模糊长度」（与光线无关），亮度应交给增益。
        # 若本场地的画面亮度已经偏低，换到更暗的场地会撑不住。
        dark = [(t, r["mean"]) for t, r in rows if r["mean"] < 70]
        if dark:
            out.append("【换场地提醒】以下档位画面偏暗（亮度<70），换到更暗的比赛场地可能撑不住，")
            out.append("              因为亮度该由增益补、不该靠拖长曝光：")
            for t, m in dark:
                out.append("                曝光 %-6s 画面亮度 %.1f" % (t, m))
        else:
            out.append("【换场地提醒】各档画面亮度都正常（>=70），曝光值可跨场地沿用；")
            out.append("              比赛场地若更暗，让增益自动补，不要回退成更长曝光。")

    text = "\n".join(out)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print("\n已写入", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
