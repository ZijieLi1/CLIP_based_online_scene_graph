"""Per-topic message counts every 10 s. Run the SAME script on the robot and on the workstation at once:
a gap on the robot = the publisher stalls; a gap only on the workstation = the network."""
import sys, time, rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
tag, dur = sys.argv[1], int(sys.argv[2])
rel = (sys.argv[3] if len(sys.argv) > 3 else "be") == "rel"
H = "/head_rgbd_sensor/"
T = {"depth":      H + "depth_registered/image_rect_raw/compressedDepth",
     "rgb":        H + "rgb/image_rect_color/compressed"     }
rclpy.init(); n = rclpy.create_node("e2e_" + tag)
cnt = {k: 0 for k in T}; kb = {k: 0 for k in T}
q = QoSProfile(reliability=ReliabilityPolicy.RELIABLE if rel else ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)
def mk(k):
    def cb(m): cnt[k] += 1; kb[k] += len(m.data)
    return cb
for k, t in T.items(): n.create_subscription(CompressedImage, t, mk(k), q)
t0 = time.time(); nxt = 10; prev = dict(cnt)
while time.time() - t0 < dur:
    rclpy.spin_once(n, timeout_sec=0.1)
    if time.time() - t0 >= nxt:
        nxt += 10
        print(tag, f"{int(time.time()-t0):4d}s", "  ".join(f"{k}={cnt[k]-prev[k]:3d}" for k in T), flush=True); prev = dict(cnt)
