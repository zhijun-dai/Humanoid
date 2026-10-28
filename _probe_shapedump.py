"""把机器人上 shape_dump 的帧在本地重跑，和本地摄像头那次对照。"""
import glob
import os
import sys

import cv2

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\用户\Lenovo\桌面\Github Repository\Jetson_orin_nano_code\new_vision\jetson")

from shape_detector import ShapeDetector

DIR = os.path.join(os.environ["TEMP"], "rb", "shape")
files = sorted(glob.glob(os.path.join(DIR, "*.jpg")))

sd = ShapeDetector(stable_frames=1, cooldown_ms=0, debug=False, roi_ratio=1.0)

print("%-46s %-8s %-4s %-9s %s" % ("帧", "规则", "g/v", "框宽", "主要拒绝"))
print("-" * 104)
n_found = n_class = 0
for p in files:
    img = cv2.imread(p)
    if img is None:
        continue
    act, dbg = sd.update(img)
    qw = dbg.get("quad_work")
    w = "-"
    if qw is not None:
        import numpy as np
        a = np.asarray(qw, np.float32)
        w = "%.0f" % (a[:, 0].max() - a[:, 0].min())
    rej = sorted((dbg.get("geom_rejects") or {}).items(), key=lambda kv: -kv[1])[:3]
    if dbg.get("card_found"):
        n_found += 1
    if dbg.get("shape") is not None:
        n_class += 1
    print("%-46s %-8s %-4s %-9s %s" % (
        os.path.basename(p)[:46], dbg.get("shape_rules") or "-",
        "%s/%s" % (dbg.get("quad_geom"), len(dbg.get("scores") or [])), w,
        " ".join("%s:%d" % kv for kv in rej) or "-"))

print()
print("总 %d 帧：找到框 %d，分类出形状 %d" % (len(files), n_found, n_class))
