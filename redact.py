#!/usr/bin/env python3
"""screencast-redact: find a piece of text (an email address, a hostname, a name) everywhere it appears
in a long screen recording and cover it with black boxes, touching nothing else.

Pipeline
  frames  grab still frames at given times so you can crop templates of the rendered text
  scan    template-match one or more templates across a low-resolution proxy, in parallel
  render  burn timed black boxes into the full-resolution original, in parallel, join, remux

Requires: ffmpeg/ffprobe on PATH, Python 3.9+, opencv-python, numpy.
"""
import argparse, glob, json, os, re, subprocess, sys, tempfile, time
from concurrent.futures import ProcessPoolExecutor

def run(cmd, **kw):
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw)

def probe(path):
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-show_entries", "format=duration", "-of", "json", path]).stdout
    j = json.loads(out); return int(j["streams"][0]["width"]), int(j["streams"][0]["height"]), float(j["format"]["duration"])

# ---------- frames ----------
def cmd_frames(a):
    os.makedirs(a.out, exist_ok=True)
    for t in a.times:
        dst = os.path.join(a.out, f"frame_{t}.png")
        run(["ffmpeg", "-v", "error", "-y", "-ss", str(t), "-i", a.video, "-frames:v", "1", dst]); print(dst)

# ---------- proxy ----------
def cmd_proxy(a):
    run(["ffmpeg", "-v", "error", "-y", "-i", a.video, "-vf", f"fps={a.fps},scale={a.width}:-2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p", "-an", a.out]); print(a.out)

# ---------- scan ----------
def _scan_range(args):
    import cv2
    proxy, templates, scale, start, end, step, thr, band, partial = args
    tpls = {}
    for path in templates:
        img = cv2.imread(path); g = cv2.cvtColor(cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        name = os.path.splitext(os.path.basename(path))[0]; tpls[name] = g
        if partial:  # left and right halves catch text cut off by a window edge
            w = g.shape[1]; tpls[name + "~L"] = g[:, : int(w * 0.45)]; tpls[name + "~R"] = g[:, int(w * 0.55):]
    cap = cv2.VideoCapture(proxy); fps = cap.get(cv2.CAP_PROP_FPS); cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    hits = []; i = start
    while i < end and cap.grab():
        if (i - start) % step == 0:
            ok, frame = cap.retrieve()
            if not ok: break
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY); y0 = 0
            if band: y0, y1 = band; g = g[y0:y1]
            for k, t in tpls.items():
                if t.shape[0] > g.shape[0] or t.shape[1] > g.shape[1]: continue
                r = cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED); th, tw = t.shape
                for _ in range(16):
                    _, mx, _, loc = cv2.minMaxLoc(r)
                    if mx < thr: break
                    x, y = loc
                    hits.append({"t": round(i / fps, 3), "k": k, "x": x / scale, "y": (y + y0) / scale, "w": tw / scale, "h": th / scale, "s": round(float(mx), 3)})
                    r[max(0, y - th): y + th, max(0, x - tw): x + tw] = 0
        i += 1
    return hits

def cmd_scan(a):
    import cv2
    W, H, _ = probe(a.video); pw, ph, _ = probe(a.proxy); scale = pw / W
    cap = cv2.VideoCapture(a.proxy); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    band = None
    if a.band:
        y0, y1 = (int(v) for v in a.band.split(":")); band = (int(y0 * scale), int(y1 * scale))
    jobs = [(a.proxy, a.templates, scale, w * n // a.jobs, (w + 1) * n // a.jobs, a.step, a.threshold, band, a.partial) for w in range(a.jobs)]
    t0 = time.time(); hits = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for part in ex.map(_scan_range, jobs): hits += part
    hits.sort(key=lambda h: h["t"]); json.dump(hits, open(a.out, "w"))
    kinds = {}
    for h in hits: kinds[h["k"]] = kinds.get(h["k"], 0) + 1
    print(f"{len(hits)} hits in {time.time()-t0:.0f}s -> {a.out}; by template: {kinds}")
    if a.calibrate:
        print("calibration: lower --threshold if real text is missed, raise it if boxes appear on unrelated text; scores of hits:",
              sorted(round(h["s"], 2) for h in hits)[:5], "...", sorted(round(h["s"], 2) for h in hits)[-5:])

# ---------- boxes ----------
def build_boxes(hits, pad_t, pad_px, gap, q=8):
    groups = {}
    for h in hits:
        key = (h["k"].split("~")[0], round(h["x"] / q) * q, round(h["y"] / q) * q); groups.setdefault(key, []).append(h)
    boxes = []
    for hs in groups.values():
        hs.sort(key=lambda h: h["t"])
        x = min(h["x"] for h in hs) - pad_px; y = min(h["y"] for h in hs) - pad_px
        w = max(h["x"] + h["w"] for h in hs) - x + pad_px; hh = max(h["y"] + h["h"] for h in hs) - y + pad_px
        start = end = None
        for h in hs:
            s, e = h["t"] - pad_t, h["t"] + pad_t
            if start is None: start, end = s, e
            elif s <= end + gap: end = max(end, e)
            else: boxes.append((start, end, x, y, w, hh)); start, end = s, e
        boxes.append((start, end, x, y, w, hh))
    boxes.sort(); return boxes

def filter_script(boxes, color):
    lines = [f"drawbox=x={max(0,int(x))}:y={max(0,int(y))}:w={int(w)}:h={int(h)}:color={color}:t=fill:enable='between(t,{max(0,a):.3f},{b:.3f})'" for a, b, x, y, w, h in boxes]
    return ",\n".join(lines)

# ---------- render ----------
def _encode_segment(args):
    video, script, start, end, out, crf, threads = args
    # -copyts keeps the original timestamps so the enable=between(t,...) windows stay absolute; with -copyts the end must be given as -to (absolute), not -t
    run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", video, "-to", f"{end:.3f}", "-copyts", "-filter_script:v", script,
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", str(crf), "-threads", str(threads), "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", "-an", out])
    return out

def cmd_render(a):
    hits = json.load(open(a.detections))
    boxes = build_boxes(hits, a.pad_time, a.pad_px, a.gap)
    if not boxes: sys.exit("no detections, nothing to render")
    W, H, D = probe(a.video)
    with tempfile.TemporaryDirectory() as d:
        script = os.path.join(d, "boxes.txt"); open(script, "w").write(filter_script(boxes, a.color))
        print(f"{len(hits)} hits -> {len(boxes)} boxes covering {sum(b-a_ for a_,b,*_ in boxes):.0f} s in total; rendering {a.jobs} segments")
        jobs = [(a.video, script, i * D / a.jobs, min(D, (i + 1) * D / a.jobs + 0.5), os.path.join(d, f"seg_{i:02d}.mov"), a.crf, a.threads) for i in range(a.jobs)]
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.jobs) as ex: segs = list(ex.map(_encode_segment, jobs))
        lst = os.path.join(d, "list.txt"); open(lst, "w").write("".join(f"file '{s}'\n" for s in segs))
        # join without re-encoding; -video_track_timescale keeps the mov duration field inside 32 bits so QuickTime plays the whole file
        run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", "-video_track_timescale", "15360", "-movflags", "+faststart", a.out])
    w, h, dur = probe(a.out); print(f"wrote {a.out}: {w}x{h}, {dur:.1f} s, {os.path.getsize(a.out)/1e6:.0f} MB in {time.time()-t0:.0f}s")

# ---------- verify ----------
def cmd_verify(a):
    hits = json.load(open(a.detections)); boxes = build_boxes(hits, a.pad_time, a.pad_px, a.gap)
    os.makedirs(a.out, exist_ok=True); picks = [boxes[i] for i in range(0, len(boxes), max(1, len(boxes) // a.count))][: a.count]
    for s, e, x, y, w, h in picks:
        t = (s + e) / 2; dst = os.path.join(a.out, f"check_{t:.1f}.png")
        cw, ch = int(w * 4), int(h * 6); cx, cy = max(0, int(x - w * 1.5)), max(0, int(y - h * 2.5))
        run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", a.video, "-frames:v", "1", "-vf", f"crop={cw}:{ch}:{cx}:{cy}", dst]); print(dst)
    print("open these crops: each should show a solid box with the surrounding text intact")

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sp = p.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("frames", help="grab still frames at given times (seconds) to crop templates from"); f.add_argument("video"); f.add_argument("times", nargs="+", type=float); f.add_argument("--out", default="frames"); f.set_defaults(fn=cmd_frames)
    x = sp.add_parser("proxy", help="make a small, low-fps proxy to scan quickly"); x.add_argument("video"); x.add_argument("out"); x.add_argument("--width", type=int, default=2048); x.add_argument("--fps", type=int, default=10); x.set_defaults(fn=cmd_proxy)
    s = sp.add_parser("scan", help="find every occurrence of the templates in the proxy"); s.add_argument("video", help="the full-resolution original (for the scale)"); s.add_argument("proxy"); s.add_argument("templates", nargs="+", help="PNG crops of the rendered text, cut from full-resolution frames")
    s.add_argument("--out", default="detections.json"); s.add_argument("--threshold", type=float, default=0.70); s.add_argument("--step", type=int, default=5, help="scan every Nth proxy frame"); s.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    s.add_argument("--band", help="restrict the search to a horizontal band y0:y1 in full-resolution pixels (fast; for text that sits in a fixed bar)"); s.add_argument("--partial", action="store_true", help="also match the left and right halves of each template, for text cut off by a window edge"); s.add_argument("--calibrate", action="store_true"); s.set_defaults(fn=cmd_scan)
    for name, fn, hlp in [("render", cmd_render, "burn the boxes into the original"), ("verify", cmd_verify, "crop frames from inside the boxed spans to check")]:
        r = sp.add_parser(name, help=hlp); r.add_argument("video"); r.add_argument("detections"); r.add_argument("--pad-time", type=float, default=0.7); r.add_argument("--pad-px", type=int, default=6); r.add_argument("--gap", type=float, default=1.5, help="merge box intervals closer than this (seconds)")
        if name == "render": r.add_argument("--out", default="redacted.mov"); r.add_argument("--color", default="black"); r.add_argument("--crf", type=int, default=22); r.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2)); r.add_argument("--threads", type=int, default=2)
        else: r.add_argument("--out", default="verify"); r.add_argument("--count", type=int, default=8)
        r.set_defaults(fn=fn)
    a = p.parse_args(); a.fn(a)

if __name__ == "__main__": main()
