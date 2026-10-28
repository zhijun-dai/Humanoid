"""看一次跑的电机日志：走了多久、在哪里停过、停了多久、IMU 偏航变化。

偏航用 imu_gyro_z_rad_s 积分，**不用 imu_yaw_rad** —— 后者是四元数解出来的
Euler 角，实测自己漂 8.5°/s（同一趟里它读 -87°，陀螺仪积分读 -14°），拿它
看"转了多少度"全是漂移。老日志没有陀螺仪列，那几份只能报"读不了"。
"""
import csv
import math
import sys

sys.stdout.reconfigure(encoding="utf-8")
path = sys.argv[1]
rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
print("行数 %d" % len(rows))
t0 = float(rows[0]["elapsed_s"])
t1 = float(rows[-1]["elapsed_s"])
print("时长 %.1f s  (%.1f Hz)" % (t1 - t0, len(rows) / max(t1 - t0, 1e-6)))

has_gyro = "imu_gyro_z_rad_s" in rows[0]

J = ["r_leg_pitch", "r_leg_roll", "r_leg_yaw", "r_knee_pitch", "r_ankle_pitch",
     "r_ankle_roll", "l_leg_pitch", "l_leg_roll", "l_leg_yaw", "l_knee_pitch",
     "l_ankle_pitch", "l_ankle_roll"]
cols = [f"actual_{j}_joint_rad" for j in J]

motion = []
for i in range(1, len(rows)):
    d = max(abs(float(rows[i][c]) - float(rows[i - 1][c])) for c in cols)
    motion.append(d)

thr = 0.002
STILL = 25          # 连续多少拍算一次停（50Hz -> 0.5s）
runs = []
i = 0
while i < len(motion):
    if motion[i] < thr:
        j = i
        while j < len(motion) and motion[j] < thr:
            j += 1
        if j - i >= STILL:
            runs.append((i + 1, j))
        i = j
    else:
        i += 1

print("\n静止段（>=0.5s）:")
if not runs:
    print("  没有 —— 全程都在动")
for a, b in runs:
    print("  %6.2fs ~ %6.2fs   停 %.2fs"
          % (float(rows[a]["elapsed_s"]), float(rows[b]["elapsed_s"]),
             float(rows[b]["elapsed_s"]) - float(rows[a]["elapsed_s"])))

print()
if not has_gyro:
    print("这份日志没有 imu_gyro_z_rad_s 列（2026-10-01 之前跑的），偏航读不了。")
    print("imu_yaw_rad 那条不能用 —— 它自己漂 8.5°/s。重跑一次就有了。")
else:
    gyro = [float(r["imu_gyro_z_rad_s"]) for r in rows]
    t = [float(r["elapsed_s"]) for r in rows]
    yaw = [0.0]
    for k in range(1, len(rows)):
        yaw.append(yaw[-1] + gyro[k] * (t[k] - t[k - 1]))
    print("全程偏航（陀螺仪积分）: 净转 %+.1f°" % math.degrees(yaw[-1]))
    print("单趟内摆动 %.1f°   噪声底约 ±2°（步态摆动是周期信号，会抵消）"
          % math.degrees(max(yaw) - min(yaw)))
    for a, b in runs:
        print("  静止段 %6.2fs ~ %6.2fs  这段漂了 %+.1f°"
              % (t[a], t[b], math.degrees(yaw[b] - yaw[a])))
seq = sorted(int(r["state_sequence"]) for r in rows)
gaps = [b - a for a, b in zip(seq, seq[1:]) if b - a > 1]
print("state_sequence: %d ~ %d，断口 %d 处%s" % (
    seq[0], seq[-1], len(gaps),
    "（最大跳 %d）" % max(gaps) if gaps else ""))
