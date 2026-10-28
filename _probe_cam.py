"""本地摄像头跑 demo 那套配置，把每帧的分类结果和触发闸门名打出来。"""
import os
import sys
import time

import cv2

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\用户\Lenovo\桌面\Github Repository\Jetson_orin_nano_code\new_vision\jetson")

from shape_detector import ShapeDetector
from utils import open_camera

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
OUT = os.path.join(os.environ["TEMP"], "rb", "cam")

sd = ShapeDetector(stable_frames=1, cooldown_ms=0, debug=False, roi_ratio=1.0)
cap = open_camera(0, 1280, 720)
if not cap.isOpened():
    print("打不开摄像头 0")
    sys.exit(1)
print("相机 %dx%d，抓 %d 帧" % (
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), N))
print()
print("%-5s %-10s %-10s %-6s %-14s %-6s %-10s %s" % (
    "帧", "规则", "shape", "action", "闸门", "框", "quad g/v", "前两个拒绝"))
print("-" * 96)

os.makedirs(OUT, exist_ok=True)
seen = {}
for i in range(N):
    ok, frame = cap.read()
    if not ok:
        time.sleep(0.02)
        continue
    act, dbg = sd.update(frame)
    rej = sorted((dbg.get("geom_rejects") or {}).items(), key=lambda kv: -kv[1])[:2]
    line = "%-5d %-10s %-10s %-6s %-14s %-6s %-10s %s" % (
        i, dbg.get("shape_rules") or "-", dbg.get("shape") or "-",
        act if act is not None else "-", dbg.get("confirm_reject") or "-",
        "有" if dbg.get("card_found") else "无",
        "%s/%s" % (dbg.get("quad_total"), len(dbg.get("scores") or [])),
        " ".join("%s:%d" % kv for kv in rej) or "-")
    print(line)

    key = dbg.get("confirm_reject")
    if key not in seen and dbg.get("card_found"):
        seen[key] = i
        p = os.path.join(OUT, "rej_%s_%03d.jpg" % (key, i))
        cv2.imwrite(p, frame)
    if act is not None:
        cv2.imwrite(os.path.join(OUT, "sent_%03d_action%d.jpg" % (i, act)), frame)

cap.release()
print()
print("有框且各闸门首次出现的帧 →", OUT)
for k, v in seen.items():
    print("  %-12s 帧 %d" % (k, v))
