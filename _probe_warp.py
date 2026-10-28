"""同一张卡、相邻两帧，为什么分类翻。把 warp 和四边形都存出来。"""
import os
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\用户\Lenovo\桌面\Github Repository\Jetson_orin_nano_code\new_vision\jetson")

from shape_detector import ShapeDetector

D = os.path.join(os.environ["TEMP"], "rb", "cross")
OUT = os.path.join(os.environ["TEMP"], "rb", "warp")
os.makedirs(OUT, exist_ok=True)

FILES = [
    "0021_pentagon_cy0.8775439227068865_g1.jpg",
    "0022_cross_cy0.877385570384838_g1.jpg",
]

sd = ShapeDetector(stable_frames=1, cooldown_ms=0, debug=False, roi_ratio=1.0)
warps = []
for f in FILES:
    img = cv2.imread(os.path.join(D, f))
    act, dbg = sd.update(img)
    w = dbg.get("warp")
    q = np.asarray(dbg.get("quad_work"), np.float32) if dbg.get("quad_work") is not None else None
    print("%-46s rules=%-9s hu=%-7s dist=%6.2f" % (
        f[:46], dbg.get("shape_rules"), dbg.get("hu_best"),
        float(dbg.get("hu_dist") or 0)))
    if q is not None:
        print("   quad_work: " + " ".join("(%.1f,%.1f)" % (p[0], p[1]) for p in q))
        print("   框宽 %.1f  宽高比 %.2f" % (
            q[:, 0].max() - q[:, 0].min(),
            (q[:, 0].max() - q[:, 0].min()) / max(q[:, 1].max() - q[:, 1].min(), 1e-6)))
    if w is not None:
        warps.append(w)
        cv2.imwrite(os.path.join(OUT, f.split("_")[1] + "_warp.png"), w)

if len(warps) == 2:
    a = cv2.cvtColor(warps[0], cv2.COLOR_GRAY2BGR)
    b = cv2.cvtColor(warps[1], cv2.COLOR_GRAY2BGR)
    cv2.putText(a, FILES[0].split("_")[1], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(b, FILES[1].split("_")[1], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    both = np.hstack([a, np.full((a.shape[0], 12, 3), 128, np.uint8), b])
    cv2.imwrite(os.path.join(OUT, "warp_pair.png"), both)
    diff = cv2.absdiff(warps[0], warps[1])
    print("\n两张 warp 的逐像素差: 中位 %d  均值 %.1f  最大 %d  非零占比 %.1f%%" % (
        int(np.median(diff)), float(diff.mean()), int(diff.max()),
        100.0 * np.count_nonzero(diff) / diff.size))
print("\n图存到", OUT)
