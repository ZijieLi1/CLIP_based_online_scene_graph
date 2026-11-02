"""One reader per HSR image topic; prints rate and MB/s every 30 s. Run it ALONE to test link capacity."""
import rclpy, sys, time
from rclpy.qos import qos_profile_sensor_data as q
from sensor_msgs.msg import CompressedImage
dur = int(sys.argv[1]) if len(sys.argv) > 1 else 180
rclpy.init(); n = rclpy.create_node("bwmon")
T = {"depth": "/head_rgbd_sensor/depth_registered/image_rect_raw/compressedDepth",
     "rgb": "/head_rgbd_sensor/rgb/image_rect_color/compressed"}
cnt = {k: 0 for k in T}; byt = {k: 0 for k in T}
def mk(k):
    def cb(m): cnt[k] += 1; byt[k] += len(m.data)
    return cb
for k, t in T.items(): n.create_subscription(CompressedImage, t, mk(k), q)
t0 = time.time(); tp = t0; pc = dict(cnt); pb = dict(byt)
while time.time() - t0 < dur:
    rclpy.spin_once(n, timeout_sec=0.1)
    if time.time() - tp >= 30:
        dt = time.time() - tp
        print(f"{int(time.time()-t0):4d}s " + "  ".join(f"{k}: {(cnt[k]-pc[k])/dt:5.1f} Hz {(byt[k]-pb[k])/dt/1e6:5.2f} MB/s" for k in T), flush=True)
        tp = time.time(); pc = dict(cnt); pb = dict(byt)
