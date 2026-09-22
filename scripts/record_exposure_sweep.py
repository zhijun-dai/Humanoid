#!/usr/bin/env python3
"""按曝光档位批量录制视频，供 analyze_exposure_sweep.py 分析。

为什么要有这个脚本：机器人在跑的时候，曝光时间决定了运动模糊的长度，
而模糊长度又决定赛道线会不会被"洗灰"。现场时间有限，这个脚本把
「设分辨率 → 设格式 → 关自动曝光 → 设曝光档 → 录制」一次做完。

跨平台：Windows 用 DSHOW，Linux/Jetson 用 V4L2。两边的曝光控制语义完全不同——
  DSHOW: CAP_PROP_AUTO_EXPOSURE 0.25=手动 / 0.75=自动；曝光值单位是 log2(秒)
  V4L2 : CAP_PROP_AUTO_EXPOSURE 1=手动 / 3=自动；曝光值单位是 100µs
所以命令行统一用**毫秒**（跨平台有意义的物理量），脚本内部各转各的。

用法：
    # Jetson
    python3 scripts/record_exposure_sweep.py --out 曝光测试 --seconds 10
    # Windows 笔记本
    python scripts/record_exposure_sweep.py --out 曝光测试 --seconds 10 --index 1

默认依次录 auto / 31ms / 16ms / 8ms 四档。录完直接跑：
    python3 scripts/analyze_exposure_sweep.py --dir 曝光测试
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import time

import cv2

IS_LINUX = sys.platform.startswith("linux")

# DSHOW 的 CAP_PROP_AUTO_EXPOSURE
DSHOW_MANUAL, DSHOW_AUTO = 0.25, 0.75
# V4L2 的 CAP_PROP_AUTO_EXPOSURE（对应 V4L2_CID_EXPOSURE_AUTO）
V4L2_MANUAL, V4L2_AUTO = 1.0, 3.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="按曝光档位批量录制")
    p.add_argument("--out", default="曝光测试", help="输出目录")
    p.add_argument("--seconds", type=float, default=10.0, help="每档录制秒数")
    p.add_argument("--width", type=int, default=1280, help="采集宽度（须与机器人实际运行一致）")
    p.add_argument("--height", type=int, default=720, help="采集高度")
    p.add_argument("--index", type=int, default=0, help="相机索引（板子上只有一颗，就是 0）")
    p.add_argument("--exposures", default="auto,31,16,8",
                   help="曝光档位（毫秒），逗号分隔；auto=自动曝光。"
                        "对应 DSHOW 的 -5/-6/-7 档")
    p.add_argument("--repeat", type=int, default=1, help="每档重复录几段")
    return p.parse_args()


def backend() -> int:
    return cv2.CAP_V4L2 if IS_LINUX else cv2.CAP_DSHOW


def fourcc_of(cap) -> str:
    v = int(cap.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))


def frame_mean(cap) -> float:
    ok, f = cap.read()
    if not ok or f is None:
        return float("nan")
    return float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean())


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, backend())
    if not cap.isOpened():
        raise SystemExit("打不开相机 index=%d（被占用？没插？）" % index)
    # 顺序不能反：先分辨率，后 FOURCC（DSHOW 下反了会把 MJPG 踢回 YUY2）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    for _ in range(15):
        cap.read()
    return cap


def apply_exposure(cap: cv2.VideoCapture, mode: str) -> str:
    """mode 为 'auto' 或毫秒字符串。返回实际设置的说明。"""
    if mode == "auto":
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, V4L2_AUTO if IS_LINUX else DSHOW_AUTO)
        note = "自动曝光"
    else:
        ms = float(mode)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, V4L2_MANUAL if IS_LINUX else DSHOW_MANUAL)
        if IS_LINUX:
            val = ms * 10.0            # UVC 的 exposure_absolute 单位是 100µs
        else:
            val = math.log2(max(ms, 0.05) / 1000.0)
        cap.set(cv2.CAP_PROP_EXPOSURE, val)
        note = "手动 %.1fms（写入值 %.1f）" % (ms, val)
    for _ in range(15):
        cap.read()
    got = cap.get(cv2.CAP_PROP_EXPOSURE)
    return "%s  读回=%.2f" % (note, got)


def main() -> int:
    args = parse_args()
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = os.path.join(os.environ.get("TEMP", "/tmp"), "exposure_sweep_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    modes = [m.strip() for m in args.exposures.split(",") if m.strip()]
    print("平台     : %s（后端 %s）" % (sys.platform, "V4L2" if IS_LINUX else "DSHOW"))
    print("输出目录 : %s" % out_dir)
    print("档位     : %s" % ", ".join(modes))
    print("每档时长 : %.0f 秒   分辨率 %dx%d   index=%d"
          % (args.seconds, args.width, args.height, args.index))
    print("")
    print(">>> 相机装到机器人上、摆成平时跑的姿态。")
    print(">>> 机器人开始跑的时候按回车。")
    input()

    cap = open_camera(args.index, args.width, args.height)
    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fcc = fourcc_of(cap)
    print("实际采集 : %dx%d  FOURCC=%s" % (real_w, real_h, fcc))
    if real_w != args.width or real_h != args.height:
        print("[警告] 分辨率没设成请求值，检查相机支持的档位")
    if not IS_LINUX and fcc != "MJPG":
        print("[警告] 没协商到 MJPG。Windows 上 index 可能选错了"
              "（笔记本内置相机常吃不到 MJPG，用 --index 换一个）")
    mean = frame_mean(cap)
    print("画面亮度 : %.1f" % mean)
    if mean < 25:
        print("[警告] 画面几乎全黑！相机可能选错了（比如选中了被遮住的笔记本内置相机），"
              "或者镜头被挡住。先解决再录。")
    print("")

    fps = 30.0
    made = []
    for mode in modes:
        for rep in range(args.repeat):
            tag = "exp%s" % mode.replace(".", "p")
            if args.repeat > 1:
                tag += "_r%d" % (rep + 1)
            dst = os.path.join(out_dir, tag + ".mp4")
            tmp = os.path.join(tmp_dir, tag + ".mp4")

            note = apply_exposure(cap, mode)
            print("--- 档位 %-5s : %s" % (mode, note))
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
    print("    python3 scripts/analyze_exposure_sweep.py --dir \"%s\"" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
