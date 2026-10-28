"""实车帧 vs 本地摄像头帧：清晰度、卡的大小、地面闸通过率。"""
import glob
import os
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\用户\Lenovo\桌面\Github Repository\Jetson_orin_nano_code\new_vision\jetson")

from shape_detector import ShapeDetector


def sharp(bgr):
    """Laplacian 方差，全图。越小越糊。"""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def report(tag, paths):
    sd = ShapeDetector(stable_frames=1, cooldown_ms=0, debug=False, roi_ratio=1.0)
    sh, found, cls = [], 0, 0
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        sh.append(sharp(img))
        _a, dbg = sd.update(img)
        found += bool(dbg.get("card_found"))
        cls += dbg.get("shape") is not None
    sh = np.array(sh)
    print("%-16s n=%-4d 锐度 中位 %6.1f  最小 %6.1f  最大 %7.1f  |  找框 %d  分类 %d"
          % (tag, len(sh), np.median(sh), sh.min(), sh.max(), found, cls))
    return sh


robot = sorted(glob.glob(os.path.join(os.environ["TEMP"], "rb", "shape", "*.jpg")))
cam = sorted(glob.glob(os.path.join(os.environ["TEMP"], "rb", "cam", "*.jpg")))

sr = report("实车 shape_dump", robot)
sc = report("本地 USB cam", cam)

print()
if len(sr) and len(sc):
    lo, hi = float(np.median(sr)), float(np.median(sc))
    print("本地帧中位锐度是实车的 %.1f 倍（%.1f vs %.1f）" % (hi / max(lo, 1e-9), hi, lo))

# 实车帧里最清的那几张，是不是就容易找框？
sd = ShapeDetector(stable_frames=1, cooldown_ms=0, debug=False, roi_ratio=1.0)
rows = []
for p in robot:
    img = cv2.imread(p)
    if img is None:
        continue
    _a, dbg = sd.update(img)
    rows.append((sharp(img), dbg.get("quad_geom", 0), os.path.basename(p)))
rows.sort(reverse=True)
print()
print("实车最清的 6 帧：")
for s, g, n in rows[:6]:
    print("  锐度 %6.1f  过了几何闸的候选 %d  %s" % (s, g, n[:42]))
print("实车最糊的 6 帧：")
for s, g, n in rows[-6:]:
    print("  锐度 %6.1f  过了几何闸的候选 %d  %s" % (s, g, n[:42]))
nz = [r for r in rows if r[0] > float(np.median(sr))]
print()
print("锐度高于中位的 %d 帧：过闸候选合计 %d，其中 >0 的 %d 帧"
      % (len(nz), sum(r[1] for r in nz), sum(1 for r in nz if r[1] > 0)))
