"""同一张卡 17 帧，solidity 到底飘在哪。分类器的十字/五角星分界是 0.36。"""
import glob
import os
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\用户\Lenovo\桌面\Github Repository\Jetson_orin_nano_code\new_vision\jetson")

from shape_detector import ShapeDetector

D = os.path.join(os.environ["TEMP"], "rb", "flick")
# 最后一遍跑的卡帧：cy 在 0.876~0.878
def _cy(p):
    tail = os.path.basename(p).split("_cy")[1].split("_")[0]
    return float(tail) if tail != "None" else -1.0


files = sorted((f for f in glob.glob(os.path.join(D, "*.jpg"))
                if 0.875 < _cy(f) < 0.879), key=_cy)

print("最后一遍的卡帧 %d 张\n" % len(files))
print("%-8s %-9s %-9s %-9s %-9s %s" % ("帧", "机器人当时", "本地重跑", "solidity", "fill_rect", "顶点"))
print("-" * 78)

sd = ShapeDetector(stable_frames=1, cooldown_ms=0, debug=False, roi_ratio=1.0)
sol, fil = [], []
for p in files:
    n = os.path.basename(p)
    robot_said = n.split("_")[1]
    img = cv2.imread(p)
    _a, dbg = sd.update(img)
    w = dbg.get("warp")
    s = f = v = None
    if w is not None:
        c = cv2.morphologyEx(w, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        c = cv2.dilate(c, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        cs, _ = cv2.findContours(c, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if cs:
            blob = max(cs, key=cv2.contourArea)
            a = cv2.contourArea(blob)
            hull = cv2.contourArea(cv2.convexHull(blob))
            s = a / max(hull, 1e-6)
            rect = cv2.minAreaRect(blob)
            f = a / max(rect[1][0] * rect[1][1], 1e-6)
            v = len(cv2.approxPolyDP(blob, 0.035 * cv2.arcLength(blob, True), True))
            sol.append(s); fil.append(f)
    print("%-8s %-9s %-9s %-9s %-9s %s" % (
        n[:8], robot_said, dbg.get("shape_rules") or "-",
        "%.3f" % s if s else "-", "%.3f" % f if f else "-", v or "-"))

if sol:
    print()
    print("solidity: 最小 %.3f  最大 %.3f  跨度 %.3f   （分界 0.36，正方形档 0.75）"
          % (min(sol), max(sol), max(sol) - min(sol)))
    print("落在各档的帧数:  cross(<0.36) %d   pentagon(0.36~0.75) %d   square(>=0.75) %d"
          % (sum(1 for x in sol if x < 0.36),
             sum(1 for x in sol if 0.36 <= x < 0.75),
             sum(1 for x in sol if x >= 0.75)))
