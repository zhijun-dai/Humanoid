"""逐帧拆 fused_err 的四项，看 runaway 时是哪一项在顶。"""
import sys, os, glob, json

sys.stdout.reconfigure(encoding="utf-8")

LOSS = os.path.join(os.environ["TEMP"], "rb", "loss")
W = 160.0          # img_w/2
LOOK = 0.25
CURVE = 0.18
ANGLE = 0.06835720391816882
LCURVE = 0.35
LCURVE_PX = 6.0
TURN_PX = 18.0


def terms(j):
    n = j["near_err_px"] / W
    f = j["far_err_px"] / W
    c = j["curve_px"] / W
    gate = min(max(abs(j["curve_px"]) / TURN_PX, 0.0), 1.0)
    t_near = -n
    t_look = LOOK * (0.70 + 0.90 * gate) * (-f)
    t_curve = CURVE * (-c)
    t_ang = ANGLE * (j["angle_err_deg"] / 45.0)
    t_lo = LCURVE * c if j["curve_px"] < -LCURVE_PX else 0.0
    return t_near, t_look, t_curve, t_ang, t_lo


rows = []
for p in sorted(glob.glob(os.path.join(LOSS, "*.json"))):
    j = json.load(open(p, encoding="utf-8"))
    ts = terms(j)
    rows.append((os.path.basename(p).replace(".json", ""), j, ts, sum(ts)))

rows.sort(key=lambda r: -abs(r[3]))

print("%-32s %-7s %-7s %-7s %-7s %-7s %-7s %-7s" % (
    "帧", "total", "near", "look", "curve", "ang", "lcurve", "|ang|%"))
print("-" * 92)
for name, j, ts, tot in rows[:22]:
    share = abs(ts[3]) / max(sum(abs(t) for t in ts), 1e-9) * 100
    print("%-32s %+7.3f %+7.3f %+7.3f %+7.3f %+7.3f %+7.3f %6.0f%%" % (
        name, tot, ts[0], ts[1], ts[2], ts[3], ts[4], share))
print()
print("按 |angle 项| 单独排：")
rows.sort(key=lambda r: -abs(r[2][3]))
for name, j, ts, tot in rows[:12]:
    share = abs(ts[3]) / max(sum(abs(t) for t in ts), 1e-9) * 100
    print("%-32s %+7.3f %+7.3f %+7.3f %+7.3f %+7.3f %+7.3f %6.0f%%  ang=%+.1f n_roi=%d" % (
        name, tot, ts[0], ts[1], ts[2], ts[3], ts[4], share,
        j["angle_err_deg"], j["n_roi_results"]))
