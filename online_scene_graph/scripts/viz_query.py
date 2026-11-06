"""Open3D viewer that highlights objects by CLIP similarity to a text query.

    # live: talk to a running mapper (find + snapshot services)
    python3 online_scene_graph/scripts/viz_query.py

    # offline: load a saved map (needs save_pcd=True when mapping)
    python3 online_scene_graph/scripts/viz_query.py --pcd outputs/online/exps/online/pcd_online.pkl.gz

Type a query in the terminal and press Enter (empty line clears). Window keys:
    R  RGB     I  instance colors     L  class-name labels     C  clear query
Hover over an object to see its point xyz, class name and (with a query) cosine similarity.
The top-left panel always shows the current color mode and query.
"""

from __future__ import annotations

import argparse
import gzip
import io
import os
import pickle
import queue
import sys
import threading
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Scene:
    ids: list
    class_names: list
    n_obs: np.ndarray
    counts: np.ndarray  # points per object
    points: np.ndarray  # (N,3) float32
    colors: np.ndarray  # (N,3) float32 in [0,1]
    clip_fts: np.ndarray | None = None  # (M,D) normalized; offline mode only
    generation: int = -1
    offsets: np.ndarray = field(init=False)

    def __post_init__(self):
        self.offsets = np.concatenate([[0], np.cumsum(self.counts)]).astype(np.int64)


def scene_from_npz(data: bytes, generation: int) -> Scene:
    z = np.load(io.BytesIO(data), allow_pickle=False)
    return Scene(
        ids=[str(x) for x in z["ids"]],
        class_names=[str(x) for x in z["class_names"]],
        n_obs=z["n_obs"],
        counts=z["counts"],
        points=z["points"].astype(np.float32),
        colors=z["colors"].astype(np.float32) / 255.0,
        generation=generation,
    )


def scene_from_pkl(path: str) -> Scene:
    with gzip.open(path, "rb") as f:
        results = pickle.load(f)
    objs = results["objects"] if isinstance(results, dict) else None
    if not objs:
        raise SystemExit(f"{path}: no 'objects' in file")
    ids, names, n_obs, counts, pts, cols, fts = [], [], [], [], [], [], []
    for o in objs:
        p = np.asarray(o["pcd_np"], dtype=np.float32)
        if len(p) == 0 or o.get("clip_ft") is None:
            continue
        c = np.asarray(o["pcd_color_np"], dtype=np.float32)
        ft = o["clip_ft"]
        ft = ft.detach().cpu().numpy() if hasattr(ft, "detach") else np.asarray(ft)
        ft = ft.astype(np.float32).reshape(-1)
        ids.append(str(o.get("id", len(ids))))
        names.append(str(o.get("class_name", "")))
        n_obs.append(int(o.get("num_detections", 0)))
        counts.append(len(p))
        pts.append(p)
        cols.append(c)
        fts.append(ft / max(float(np.linalg.norm(ft)), 1e-8))
    return Scene(
        ids=ids, class_names=names, n_obs=np.array(n_obs), counts=np.array(counts),
        points=np.concatenate(pts), colors=np.concatenate(cols), clip_fts=np.stack(fts),
    )


class LocalScorer:
    """Offline: loads the CLIP text tower on first query."""

    def __init__(self, scene: Scene, device: str):
        self.scene, self.device = scene, device
        self.model = self.tok = None

    def score(self, text: str) -> np.ndarray:
        import open_clip
        import torch
        import torch.nn.functional as F

        if self.model is None:
            print("loading CLIP ViT-H-14 text tower ...")
            self.model = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")[0]
            self.model = self.model.eval().to(self.device)
            self.tok = open_clip.get_tokenizer("ViT-H-14")
        with torch.no_grad():
            q = self.model.encode_text(self.tok([text]).to(self.device))
            q = F.normalize(q, dim=-1).squeeze(0).cpu().numpy()
        return self.scene.clip_fts @ q


class LiveClient:
    """All rclpy calls happen on the worker thread that owns this object."""

    def __init__(self, find_service, snapshot_service, max_points):
        import rclpy
        from online_scene_graph_interfaces.srv import Find, Snapshot

        self.rclpy, self.Find, self.Snapshot = rclpy, Find, Snapshot
        rclpy.init()
        self.node = rclpy.create_node("osg_viz_query")
        self.find_cli = self.node.create_client(Find, find_service)
        self.snap_cli = self.node.create_client(Snapshot, snapshot_service)
        self.max_points = max_points
        for cli, name in ((self.find_cli, find_service), (self.snap_cli, snapshot_service)):
            if not cli.wait_for_service(timeout_sec=5.0):
                raise SystemExit(f"service {name} not available (is the mapper running / interfaces rebuilt?)")

    def _call(self, cli, req, timeout=30.0):
        fut = cli.call_async(req)
        self.rclpy.spin_until_future_complete(self.node, fut, timeout_sec=timeout)
        return fut.result()

    def snapshot(self, since: int):
        req = self.Snapshot.Request()
        req.since_generation = int(since)
        req.max_points_per_object = int(self.max_points)
        res = self._call(self.snap_cli, req)
        if res is None or not res.ok:
            print("snapshot failed:", getattr(res, "message", "timeout"), file=sys.stderr)
            return None
        if res.unchanged:
            return None
        return scene_from_npz(bytes(res.data), int(res.generation))

    def score(self, text: str, scene: Scene) -> np.ndarray:
        import json

        req = self.Find.Request()
        req.text, req.k, req.min_sim, req.min_obs = text, 100000, -1.0, 1
        res = self._call(self.find_cli, req)
        if res is None or not res.ok:
            raise RuntimeError(getattr(res, "message", "") or "find() timed out")
        by_id = {h["id"]: h["sim"] for h in json.loads(res.json or "[]")}
        return np.array([by_id.get(i, np.nan) for i in scene.ids], dtype=np.float32)

    def close(self):
        self.node.destroy_node()
        self.rclpy.shutdown()


def sim_colors(sims: np.ndarray, top_k: int, cmap) -> np.ndarray:
    """ConceptGraphs-style min-max + turbo; everything outside top-k is faded toward gray."""
    s = np.nan_to_num(sims, nan=np.nanmin(sims) if np.isfinite(sims).any() else 0.0)
    span = float(s.max() - s.min())
    norm = (s - s.min()) / span if span > 1e-8 else np.zeros_like(s)
    rgb = cmap(norm)[:, :3].astype(np.float32)
    keep = np.zeros(len(s), bool)
    keep[np.argsort(-s)[:top_k]] = True
    rgb[~keep] = 0.25 * rgb[~keep] + 0.75 * 0.35
    return rgb


MODE_TEXT = {
    "rgb": "RGB (original colors)",
    "instance": "instance (turbo, one color per object)",
    "sim": "CLIP similarity (turbo, min-max over objects)",
}


class Viewer:
    """Open3D GUI window. Everything here runs on the GUI thread (use post() from others)."""

    def __init__(self, args, state, lock, cmap):
        import open3d as o3d
        from open3d.visualization import gui, rendering

        self.o3d, self.gui, self.rendering = o3d, gui, rendering
        self.args, self.state, self.lock, self.cmap = args, state, lock, cmap
        self.app = gui.Application.instance
        self.app.initialize()
        self.win = self.app.create_window("online_scene_graph query", 1280, 800)
        em = self.win.theme.font_size
        self.em = em

        self.widget = gui.SceneWidget()
        self.widget.scene = rendering.Open3DScene(self.win.renderer)
        self.widget.scene.set_background([0.08, 0.08, 0.1, 1.0])
        self.win.add_child(self.widget)

        def panel(n_lines, bg):
            labels = [gui.Label("") for _ in range(n_lines)]
            col = gui.Vert(0.25 * em, gui.Margins(0.6 * em, 0.4 * em, 0.6 * em, 0.4 * em))
            for lb in labels:
                lb.text_color = gui.Color(1, 1, 1)
                col.add_child(lb)
            col.background_color = gui.Color(*bg)
            self.win.add_child(col)
            return col, labels

        self.hud, self.hud_lines = panel(4, (0.0, 0.0, 0.0, 0.6))
        self.tip, self.tip_lines = panel(3, (0.1, 0.1, 0.1, 0.9))
        self.tip.visible = False
        self.tip_xy = (0, 0)

        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit"
        self.mat.point_size = float(args.point_size) * self.win.scaling
        self.line_mat = rendering.MaterialRecord()
        self.line_mat.shader = "unlitLine"
        self.line_mat.line_width = 2.0 * self.win.scaling

        self.pcd = o3d.geometry.PointCloud()
        self.has_cloud = False
        self.box_names: list[str] = []
        self.label3d: list = []
        self.show_labels = False
        self.scene: Scene | None = None
        self.pts_h = None
        self.obj_of = None
        self.centers = None
        self.camera_set = False
        self.last_pick = 0.0
        self.mouse_xy = None

        self.win.set_on_layout(self.on_layout)
        self.win.set_on_key(self.on_key)
        self.widget.set_on_mouse(self.on_mouse)
        self.win.set_on_close(self.on_close)

    # ---- plumbing -------------------------------------------------------------------
    def post(self):
        """Thread-safe: ask the GUI thread to re-read state and redraw."""
        self.app.post_to_main_thread(self.win, self.refresh)

    def on_close(self):
        # Open3D/Filament segfaults while tearing the GL context down in the container
        # (leaving a multi-GB core file), so leave without running that teardown.
        self.state["stop"] = True
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    def on_layout(self, ctx):
        r = self.win.content_rect
        em = self.em
        self.widget.frame = r
        self.hud.frame = self.gui.Rect(r.x + 0.8 * em, r.y + 0.8 * em, 44 * em, 6.2 * em)
        tx = min(max(self.tip_xy[0], r.x), r.get_right() - 26 * em)
        ty = min(max(self.tip_xy[1], r.y), r.get_bottom() - 5 * em)
        self.tip.frame = self.gui.Rect(tx, ty, 26 * em, 5 * em)

    # ---- state -> screen ------------------------------------------------------------
    def refresh(self):
        with self.lock:
            scene, sims, query = self.state["scene"], self.state["sims"], self.state["query"]
            mode = self.state["mode"]
        if scene is not self.scene:
            self.scene = scene
            self._load_scene(scene)
        n_obj = 0 if scene is None else len(scene.ids)
        if sims is not None and (scene is None or len(sims) != n_obj):
            sims = None  # stale scores for a different map; worker re-scores shortly
        if mode == "sim" and sims is None:
            mode = "rgb"
        self.sims = sims
        per_obj = None
        if scene is not None:
            if mode == "sim":
                per_obj = sim_colors(sims, self.args.top_k, self.cmap)
            elif mode == "instance":
                per_obj = self.cmap(np.linspace(0, 1, max(n_obj, 1)))[:n_obj, :3].astype(np.float32)
            col = scene.colors if per_obj is None else np.repeat(per_obj, scene.counts, axis=0)
            self.pcd.colors = self.o3d.utility.Vector3dVector(col.astype(np.float64))
            if self.has_cloud:
                self.widget.scene.remove_geometry("map")
            self.widget.scene.add_geometry("map", self.pcd, self.mat)
            self.has_cloud = True
            self._set_boxes(per_obj if mode == "sim" else None, sims)
        self._set_labels(sims)
        self._set_hud(scene, mode, query)
        self.tip.visible = False
        self.win.post_redraw()

    def _load_scene(self, scene):
        if scene is None:
            return
        self.pcd.points = self.o3d.utility.Vector3dVector(scene.points.astype(np.float64))
        self.pts_h = np.concatenate([scene.points, np.ones((len(scene.points), 1), np.float32)], axis=1)
        self.obj_of = np.repeat(np.arange(len(scene.ids)), scene.counts)
        seg_lo = np.array([scene.points[a:b].min(0) for a, b in zip(scene.offsets[:-1], scene.offsets[1:])])
        seg_hi = np.array([scene.points[a:b].max(0) for a, b in zip(scene.offsets[:-1], scene.offsets[1:])])
        self.seg_lo, self.seg_hi = seg_lo, seg_hi
        self.centers = (seg_lo + seg_hi) / 2.0
        if not self.camera_set and len(scene.points):
            bb = self.pcd.get_axis_aligned_bounding_box()
            self.widget.setup_camera(60.0, bb, bb.get_center())
            self.camera_set = True

    def _set_boxes(self, per_obj, sims):
        for n in self.box_names:
            self.widget.scene.remove_geometry(n)
        self.box_names = []
        if per_obj is None:
            return
        s = np.nan_to_num(sims, nan=-9)
        for i in np.argsort(-s)[: self.args.top_k]:
            box = self.o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
                self.o3d.geometry.AxisAlignedBoundingBox(self.seg_lo[i], self.seg_hi[i]))
            box.paint_uniform_color(per_obj[i].tolist())
            name = f"box{i}"
            self.widget.scene.add_geometry(name, box, self.line_mat)
            self.box_names.append(name)

    def _set_labels(self, sims):
        for lb in self.label3d:
            self.widget.remove_3d_label(lb)
        self.label3d = []
        if not self.show_labels or self.scene is None:
            return
        for i, name in enumerate(self.scene.class_names):
            text = name or "?"
            if sims is not None and np.isfinite(sims[i]):
                text += f" {sims[i]:.2f}"
            lb = self.widget.add_3d_label(self.centers[i].astype(np.float64), text)
            lb.color = self.gui.Color(1, 1, 1)
            self.label3d.append(lb)

    def _set_hud(self, scene, mode, query):
        n = 0 if scene is None else len(scene.ids)
        gen = "" if scene is None or scene.generation < 0 else f"   map generation {scene.generation}"
        lines = [
            f"Color: {MODE_TEXT[mode]}",
            f"Query: {query!r}" if query else "Query: (none)",
            f"Objects: {n}{gen}   labels: {'on' if self.show_labels else 'off'}",
            "Keys: R rgb | I instance | L labels | C clear query",
        ]
        for lb, t in zip(self.hud_lines, lines):
            lb.text = t
        self.win.set_needs_layout()

    # ---- input ---------------------------------------------------------------------
    def on_key(self, event):
        if event.type != self.gui.KeyEvent.DOWN:
            return False
        K = self.gui.KeyName
        if event.key == K.R:
            self._set_mode("rgb")
        elif event.key == K.I:
            self._set_mode("instance")
        elif event.key == K.L:
            self.show_labels = not self.show_labels
            self.refresh()
        elif event.key == K.C:
            with self.lock:
                self.state.update(query=None, sims=None, mode="rgb")
            self.refresh()
        else:
            return False
        return True

    def _set_mode(self, m):
        with self.lock:
            self.state["mode"] = m
        self.refresh()

    def on_mouse(self, event):
        gui = self.gui
        E = gui.MouseEvent
        if event.type in (E.BUTTON_DOWN, E.WHEEL, E.DRAG):
            if self.tip.visible:
                self.tip.visible = False
                self.win.post_redraw()
        elif event.type == E.MOVE:
            f = self.widget.frame
            self.hover(event.x - f.x, event.y - f.y, event.x, event.y)
        return gui.Widget.EventCallbackResult.IGNORED  # keep camera controls working

    # ---- picking -------------------------------------------------------------------
    def project(self, pts_h):
        """World -> (pixel x, pixel y, ndc depth, valid), in the scene widget's frame."""
        cam = self.widget.scene.camera
        M = np.asarray(cam.get_projection_matrix()) @ np.asarray(cam.get_view_matrix())
        clip = pts_h @ M.T.astype(np.float32)
        w = clip[:, 3]
        ok = w > 1e-6
        w = np.where(ok, w, 1.0)
        ndc = clip[:, :3] / w[:, None]
        f = self.widget.frame
        sx = (ndc[:, 0] * 0.5 + 0.5) * f.width
        sy = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * f.height
        return sx, sy, ndc[:, 2], ok

    def pick(self, px, py):
        if self.pts_h is None or len(self.pts_h) == 0:
            return None
        sx, sy, depth, ok = self.project(self.pts_h)
        r = 7.0 * self.win.scaling
        near = ok & ((sx - px) ** 2 + (sy - py) ** 2 < r * r)
        cand = np.flatnonzero(near)
        if len(cand) == 0:
            return None
        j = cand[np.argmin(depth[cand])]
        return int(self.obj_of[j]), j

    def hover(self, px, py, win_x, win_y):
        now = time.time()
        if now - self.last_pick < 0.04:  # ~25 Hz max
            return
        self.last_pick = now
        hit = self.pick(px, py)
        if hit is None:
            if self.tip.visible:
                self.tip.visible = False
                self.win.post_redraw()
            return
        i, j = hit
        x, y, z = self.scene.points[j]
        self.tip_lines[0].text = f"xyz: ({x:.2f}, {y:.2f}, {z:.2f})"
        self.tip_lines[1].text = f"class: {self.scene.class_names[i] or '?'}"
        sims = getattr(self, "sims", None)
        if sims is not None and np.isfinite(sims[i]):
            self.tip_lines[2].text = f"cos sim: {sims[i]:.3f}"
        else:
            self.tip_lines[2].text = ""
        self.tip_xy = (win_x + 16, win_y + 16)
        self.tip.visible = True
        self.win.set_needs_layout()
        self.win.post_redraw()


def main():
    ap = argparse.ArgumentParser(description="Highlight objects by CLIP similarity in an Open3D window.")
    ap.add_argument("--pcd", help="saved pcd_*.pkl.gz; omit for live mode (default)")
    ap.add_argument("--find-service", default="/online_scene_graph/find")
    ap.add_argument("--snapshot-service", default="/online_scene_graph/snapshot")
    ap.add_argument("--refresh", type=float, default=2.0, help="live: seconds between snapshot polls")
    ap.add_argument("--max-points", type=int, default=2000, help="live: points per object in a snapshot")
    ap.add_argument("--top-k", type=int, default=5, help="objects kept bright; the rest fade")
    ap.add_argument("--device", default="cuda", help="offline: device for the CLIP text tower")
    ap.add_argument("--point-size", type=float, default=3.0)
    args = ap.parse_args()

    import matplotlib

    cmap = matplotlib.colormaps["turbo"]
    live = args.pcd is None
    lock = threading.Lock()
    state = {"scene": None, "sims": None, "query": None, "mode": "rgb", "stop": False}
    queries: queue.Queue = queue.Queue()

    scene0 = None
    if not live:
        scene0 = scene_from_pkl(args.pcd)
        state["scene"] = scene0
        print(f"loaded {len(scene0.ids)} objects from {args.pcd}")

    viewer = Viewer(args, state, lock, cmap)

    def report(scene, sims, text):
        order = [i for i in np.argsort(-np.nan_to_num(sims, nan=-9)) if np.isfinite(sims[i])][: args.top_k]
        print(f"query {text!r}:")
        for r, i in enumerate(order, 1):
            c = scene.points[scene.offsets[i]:scene.offsets[i + 1]].mean(0)
            print(f"  {r}. sim={sims[i]:.3f} n={int(scene.n_obs[i]):3d} {scene.class_names[i]:20s} "
                  f"xyz=({c[0]:6.2f},{c[1]:6.2f},{c[2]:6.2f}) {scene.ids[i][:8]}")

    def worker():
        try:
            run_worker()
        except BaseException as exc:  # SystemExit in a thread is otherwise silent
            print(f"worker stopped: {exc}", file=sys.stderr)
            state["stop"] = True

    def run_worker():
        client = LiveClient(args.find_service, args.snapshot_service, args.max_points) if live else None
        local = None if live else LocalScorer(scene0, args.device)
        last_poll = 0.0
        try:
            while not state["stop"]:
                try:
                    text = queries.get(timeout=0.2)
                except queue.Empty:
                    text = None
                if live and (time.time() - last_poll > args.refresh or state["scene"] is None):
                    last_poll = time.time()
                    gen = state["scene"].generation if state["scene"] is not None else -1
                    new = client.snapshot(gen)
                    if new is not None:
                        with lock:
                            state["scene"] = new
                        viewer.post()
                        if state["query"] and text is None:
                            text = state["query"]  # re-score against the grown map
                if text is None:
                    continue
                with lock:
                    scene = state["scene"]
                if scene is None:
                    print("no map yet")
                    continue
                try:
                    sims = client.score(text, scene) if live else local.score(text)
                except Exception as exc:
                    print("query failed:", exc, file=sys.stderr)
                    continue
                with lock:
                    first = state["query"] != text
                    state.update(sims=sims, query=text, mode="sim")
                viewer.post()
                if first:
                    report(scene, sims, text)
        finally:
            if client:
                client.close()

    def stdin_loop():
        print("Type a query and press Enter (empty line = clear).")
        for line in sys.stdin:
            line = line.strip()
            if line:
                queries.put(line)
            else:
                with lock:
                    state.update(query=None, sims=None, mode="rgb")
                viewer.post()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    threading.Thread(target=stdin_loop, daemon=True).start()
    viewer.refresh()
    try:
        viewer.app.run()
    finally:
        state["stop"] = True
        t.join(timeout=3.0)


if __name__ == "__main__":
    main()
