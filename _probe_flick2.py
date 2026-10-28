"""挑 solidity 最低和最高的几帧，把 warp 和预处理后的图并排存出来。"""
import glob
import os
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\用户\Lenovo\桌面\Github Repository\Jetson_orin_nano_code\new_vision\jetson")

from shape_detector import ShapeDetector

D = os.path.join(os.environ["TEMP"], "rb", "flick")
OUT = os.path.join(os.environ["TEMP"], "rb", "flick_warp")
os.makedirs(OUT, exist_ok=True)


def _cy(p):
    t = os.path.basename(p).split("_cy")[1].split("_")[0]
    return float(t) if t != "None" else -1.0


files = sorted((f for f in glob.glob(os.path.join(D, "*.jpg"))
                if 0.875 < _cy(f) < 0.879), key=_cy)

sd = ShapeDetector(stable_frames=1, cooldown_ms=0, debug=False, roi_ratio=1.0)
rows = []
for p in files:
    img = cv2.imread(p)
    _a, dbg = sd.update(img)
    w = dbg.get("warp")
    if w is None:
        continue
    pre = cv2.morphologyEx(w, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    pre = cv2.dilate(pre, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    cs, _ = cv2.findContours(pre, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    s = 0.0
    if cs:
        blob = max(cs, key=cv2.contourArea)
        s = cv2.contourArea(blob) / max(cv2.contourArea(cv2.convexHull(blob)), 1e-6)
    rows.append((s, os.path.basename(p), w, pre))

rows.sort(key=lambda r: r[0])
picks = rows[:3] + rows[-3:]

panels = []
for s, n, w, pre in picks:
    a = cv2.cvtColor(w, cv2.COLOR_GRAY2BGR)
    b = cv2.cvtColor(pre, cv2.COLOR_GRAY2BGR)
    for img, tag in ((a, "warp"), (b, "pre")):
        cv2.putText(img, "%s s=%.2f" % (tag, s), (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(img, n.split("_")[1][:6], (4, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
        panels.append(img)
        panels.append(np.full((w.shape[0], 6, 3), 90, np.uint8))
    panels.append(np.full((w.shape[0], 18, 3), 40, np.uint8))

cv2.imwrite(os.path.join(OUT, "flick_warps.png"), np.hstack(panels[:-1]))
print("最低 solidity 3 帧 + 最高 3 帧（每帧 left=warp, right=close+dilate 之后）")
print("存到", OUT)
for s, n, _w, _p in picks:
    print("  %.3f  %s" % (s, n))
