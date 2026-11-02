"""Raw TCP/UDP link test (no ROS). Run `sink` on one machine, `send` on the other.

  sink:  python3 nettest.py sink PORT
  send:  python3 nettest.py send HOST PORT tcp SECONDS
         python3 nettest.py send HOST PORT udp SECONDS DGRAM_BYTES MB_PER_S
UDP sizes above the MTU (~1472) are IP-fragmented like large DDS samples.
"""
import socket, struct, sys, time, threading

def sink(port):
    t = socket.socket(); t.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    t.bind(("0.0.0.0", port)); t.listen(1)
    u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); u.bind(("0.0.0.0", port))
    u.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 << 20)
    def tcp():
        while True:
            c, _ = t.accept(); n = 0; t0 = time.time()
            while True:
                b = c.recv(1 << 20)
                if not b: break
                n += len(b)
            dt = time.time() - t0; print(f"TCP recv {n/dt/1e6:.1f} MB/s ({n*8/dt/1e6:.0f} Mbit/s) over {dt:.1f}s", flush=True)
    threading.Thread(target=tcp, daemon=True).start()
    seen = set(); first = None; last_t = 0; nb = 0; maxseq = -1; t0 = None; wt = time.time(); wb = 0
    u.settimeout(1.0)
    while True:
        try:
            b, _ = u.recvfrom(70000)
            now = time.time()
            if t0 is None: t0 = now
            seq = struct.unpack("!I", b[:4])[0]; seen.add(seq); maxseq = max(maxseq, seq); nb += len(b); last_t = now
            wb += len(b)
            if now - wt >= 5:
                print(f"  [{time.strftime('%H:%M:%S')}] {wb/(now-wt)/1e6:5.2f} MB/s in last {now-wt:.0f}s", flush=True); wt = now; wb = 0
        except socket.timeout:
            if t0 is not None and time.time() - last_t > 2:
                exp = maxseq + 1; dt = last_t - t0
                print(f"UDP recv {len(seen)}/{exp} datagrams, loss {100*(1-len(seen)/exp):.1f}%  {nb/dt/1e6:.2f} MB/s received", flush=True)
                seen = set(); nb = 0; maxseq = -1; t0 = None

def send(host, port, mode, secs, size=0, rate=0.0):
    if mode == "tcp":
        s = socket.create_connection((host, port)); buf = b"x" * (1 << 20); t0 = time.time(); n = 0
        while time.time() - t0 < secs: s.sendall(buf); n += len(buf)
        s.close(); dt = time.time() - t0; print(f"TCP sent {n/dt/1e6:.1f} MB/s over {dt:.1f}s", flush=True)
    else:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); pay = b"x" * (size - 4)
        interval = size / (rate * 1e6); t0 = time.time(); seq = 0; nxt = t0
        while time.time() - t0 < secs:
            s.sendto(struct.pack("!I", seq) + pay, (host, port)); seq += 1; nxt += interval
            d = nxt - time.time()
            if d > 0: time.sleep(d)
        print(f"UDP sent {seq} datagrams of {size} B at ~{rate} MB/s", flush=True)

if __name__ == "__main__":
    a = sys.argv
    if a[1] == "sink":
        sink(int(a[2]))
    else:
        host, port, mode, secs = a[2], int(a[3]), a[4], float(a[5])
        if mode == "tcp":
            send(host, port, "tcp", secs)
        else:
            send(host, port, "udp", secs, int(a[6]), float(a[7]))
