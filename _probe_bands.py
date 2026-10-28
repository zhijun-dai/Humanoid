"""泄 loss_dump 的帧：每个带到底返回没有、conf/pair 多少、heading 怎么炸的。"""
import sys, os, glob
import numpy as np, cv2

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\用户\Lenovo\桌面\Github Repository\Jetson_orin_nano_code\new_vision\jetson")
from line_detector_v1_warp import LineDetector

LOSS = os.path.join(os.environ["TEMP"], "rb", "loss")
FILES = sorted(glob.glob(os.path.join(LOSS, "*_frame.jpg")))

det = LineDetector()
print("conf_min=%s min_weight=%s two_band=%s simple=%s" % (
    det.conf_min, det.min_weight, det.two_band_mode, det.simple_bottom_mode))
print("low y=%s-%s  mid y=%s-%s" % (
    det.band_low_y0, det.band_low_y1, det.band_mid_y0, det.band_mid_y1))
print()
print("%-34s %-30s %-7s %-8s %-8s %-8s %-8s %-8s" % (
    "帧", "带(name:conf:pair)", "botPair", "near_px", "far_px", "curve_px", "ang", "conf"))
print("-" * 116)

group = None
for p in FILES:
    name = os.path.basename(p)
    g = name.split("_")[0]
    if g != group:
        det = LineDetector()
        group = g

    seen = []
    _orig = det._detect_two_band_lanes

    def spy(*a, **k):
        out = _orig(*a, **k)
        for r in (out or []):
            seen.append("%s:%.2f:%.2f" % (r.get("band_name"), r.get("conf", -1),
                                          r.get("pair_ratio", -1)))
        return out

    det._detect_two_band_lanes = spy

    img = cv2.imread(p)
    if img is None:
        continue
    dev, hdg, conf, vis, dbg = det.process(img)

    print("%-34s %-30s %-7.2f %-8.1f %-8.1f %-8.1f %-8.1f %-8.2f" % (
        name.replace("_frame.jpg", ""), " ".join(seen) or "(none)",
        dbg.get("bottom_pair_ratio", 0.0), dbg.get("near_err_px", 0.0),
        dbg.get("far_err_px", 0.0), dbg.get("curve_px", 0.0),
        dbg.get("angle_err_deg", 0.0), conf))
