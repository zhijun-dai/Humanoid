#!/usr/bin/env python3
"""按曝光档位批量录制实车视频，供 analyze_exposure_sweep.py 分析。

为什么要有这个脚本：机器人在跑的时候，摄像头的曝光档位决定了运动模糊的长度，
而模糊长度又决定赛道线会不会被"洗灰"。现场时间有限，这个脚本把
「设分辨率 → 设格式 → 关自动曝光 → 设曝光档 → 录制」一次做完，避免手忙脚乱。

用法（相机插在笔记本上，机器人正常跑）：

    python scripts/record_exposure_sweep.py --out 曝光测试 --seconds 10

默认依次录 auto / -5 / -6 / -7 四档，每档录一段。录完直接跑：

    python scripts/analyze_exposure_sweep.py --dir 曝光测试

注意：DSHOW 下必须先设分辨率再设 FOURCC，反了会把 MJPG 踢回 YUY2；
也不要设 CAP_PROP_FPS，那会触发驱动重新协商、同样会踢掉 MJPG。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import cv2

# DSHOW 的 CAP_PROP_AUTO_EXPOSURE：0.25 = 手动，0.75 = 自动
DSHOW_AUTO_EXPOSURE_MANUAL = 0.25
DSHOW_AUTO_EXPOSURE_AUTO = 0.75


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="按曝光档位批量录制")
    p.add_argument("--out", default="曝光测试", help="输出目录")
    p.add_argument("--seconds", type=float, default=10.0, help="每档录制秒数")
    p.add_argument("--width", type=int, default=1280, help="采集宽度（须与机器人实际运行一致）")
    p.add_argument("--height", type=int, default=720, help="采集高度")
    p.add_argument("--index", type=int, default=0, help="相机索引")
    p.add_argument("--exposures", default="auto,-5,-6,-7",
                   help="曝光档位，逗号分隔；auto 表示自动曝光。DSHOW 下 -5=31ms -6=16ms -7=8ms")
    p.add_argument("--repeat", type=int, default=1, help="每档重复录几段（用于重复性）")
    return p.parse_args()


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("打不开相机 index=%d（被占用？没插？）" % index)
    # 顺序不能反：先分辨率，后 FOURCC
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    for _ in range(10):
        cap.read()
    return cap


def apply_exposure(cap: cv2.VideoCapture, mode: str) -> None:
    if mode == "auto":
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, DSHOW_AUTO_EXPOSURE_AUTO)
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, DSHOW_AUTO_EXPOSURE_MANUAL)
        cap.set(cv2.CAP_PROP_EXPOSURE, float(mode))
    for _ in range(15):        # 让新设置生效并稳定
        cap.read()


def main() -> int:
    args = parse_args()
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = os.path.join(os.environ.get("TEMP", "/tmp"), "exposure_sweep_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    modes = [m.strip() for m in args.exposures.split(",") if m.strip()]
    print("输出目录 : %s" % out_dir)
    print("档位     : %s" % ", ".join(modes))
    print("每档时长 : %.0f 秒   分辨率 %dx%d" % (args.seconds, args.width, args.height))
    print("")
    print(">>> 现在把相机装到机器人上、摆成平时跑的姿态。")
    print(">>> 机器人开始跑的时候，按回车开始录第一档。")
    input()

    cap = open_camera(args.index, args.width, args.height)
    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((fc >> (8 * i)) & 0xFF) for i in range(4))
    print("实际采集 : %dx%d  FOURCC=%s" % (real_w, real_h, fourcc))
    if fourcc != "MJPG":
        print("[警告] 不是 MJPG，帧率会掉一半以上（YUY2 在 720p 只有 10fps）")
    print("")

    fps = 30.0
    made = []
    for mode in modes:
        for rep in range(args.repeat):
            tag = "exp%s" % mode.replace(".", "p").replace("-", "m")
            if args.repeat > 1:
                tag += "_r%d" % (rep + 1)
            dst = os.path.join(out_dir, tag + ".mp4")
            tmp = os.path.join(tmp_dir, tag + ".mp4")

            apply_exposure(cap, mode)
            print("--- 档位 %-5s : 录 %.0f 秒 -> %s" % (mode, args.seconds, os.path.basename(dst)))
            wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (real_w, real_h))
            n = 0
            t0 = time.time()
            while time.time() - t0 < args.seconds:
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                wr.write(frame)
                n += 1
                if n % 30 == 0:
                    sys.stdout.write("\r    已录 %d 帧 (%.1fs)" % (n, time.time() - t0))
                    sys.stdout.flush()
            wr.release()
            sys.stdout.write("\r    完成 %d 帧            \n" % n)
            if os.path.exists(dst):
                os.remove(dst)
            shutil.move(tmp, dst)
            made.append(dst)

    cap.release()
    print("")
    print("录完 %d 段：" % len(made))
    for m in made:
        print("   ", m)
    print("")
    print("下一步：")
    print("    python scripts/analyze_exposure_sweep.py --dir \"%s\"" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
