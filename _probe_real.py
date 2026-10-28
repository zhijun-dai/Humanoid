"""实车帧上：闸开到最大，卡形状的框到底出不出得来。"""
import sys, os, glob
import numpy as np, cv2

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\用户\Lenovo\桌面\Github Repository\Jetson_orin_nano_code\new_vision\jetson")
from shape_detector import ShapeDetector, _card_ink

files = sorted(glob.glob(os.path.join(os.environ["TEMP"], "rb", "s2", "*.jpg")))
det = ShapeDetector()
c = det.cfg

WIDE = dict(area_min=200, area_max=4000000, min_w=8, min_h=6,
            aspect_min=1.0, aspect_max=8.0, ang_min=5, ang_max=175)


def run(img, wide):
    saved = {k: c[k] for k in WIDE}
    if wide:
        c.update(WIDE)
    _a, dbg = det.update(img)
    for k, v in saved.items():
        c[k] = v
    return dbg


print("%-6s %-8s | %-28s | %s" % ("帧", "cy", "原始闸 quad/g", "全开到最大后"))
for p in files:
    name = os.path.basename(p).split("_")[0]
    cy = os.path.basename(p).split("cy")[1][:4]
    img = cv2.imread(p)
    if img is None:
        continue
    d1 = run(img, False)
    d2 = run(img, True)

    def desc(d):
        rej = sorted((d.get("geom_rejects") or {}).items(), key=lambda kv: -kv[1])[:2]
        return "quad=%sg%s %s" % (d.get("quad_total"), d.get("quad_geom"),
                                  " ".join("%s:%d" % kv for kv in rej))

    flag = ""
    if d2.get("quad_geom", 0) > d1.get("quad_geom", 0):
        flag = "  <<< 放宽后多过了 %d 个" % (d2["quad_geom"] - d1["quad_geom"])
    print("%-6s %-8s | %-28s | %s%s" % (name, cy, desc(d1), desc(d2), flag))
