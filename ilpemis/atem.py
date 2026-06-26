r"""ATEM montage: anchor to the task timeline, de-hum the audio, cut per-condition clips.

The ATEM video is not on the Delsys hardware sync and shares no audio with OT/Delsys,
so it is anchored by **motion-template matching**: the four task bouts have a
distinctive duration/gap signature in ``trial_times`` which is slid over the video's
motion-energy envelope to recover the ATEM<->task offset. Clips are then cut from the
montage with the de-hummed CAM-4 audio.
"""
from __future__ import annotations

import os
import subprocess
import numpy as np

from . import preprocess as pp

ATEM_DIR = os.path.join(pp.ROOT, "atem")
MONTAGE = os.path.join(ATEM_DIR, "table_wiping_atem.mp4")
CAM4_WAV = os.path.join(ATEM_DIR, "seg1", "Immersion Lab CAM 4 01.wav")
OUT_DIR = os.path.join(pp.WORKDIR, "atem")
PROXY = os.path.join(OUT_DIR, "atem_proxy.mp4")
DEHUMMED = os.path.join(OUT_DIR, "cam4_dehummed.wav")
PROXY_FPS = 10.0
HUM_HARMONICS = (60, 120, 180, 240, 300, 360)     # 60 Hz line hum + harmonics
CLIP_CONDITIONS = ("Normal", "Tensed", "Fast")    # Slow dropped


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd[:8])}...\n{r.stderr[-600:]}")


def dehum(src=CAM4_WAV, out=DEHUMMED, harmonics=HUM_HARMONICS):
    """Notch 60 Hz + harmonics out of the CAM-4 audio (verified -20..-41 dB)."""
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if not os.path.exists(out):
        notch = ",".join(f"bandreject=f={f}:t=q:w=12" for f in harmonics)
        _run(["ffmpeg", "-y", "-i", src, "-af", notch, "-c:a", "pcm_s24le", out])
    return out


def _build_proxy(montage=MONTAGE, proxy=PROXY, fps=PROXY_FPS):
    os.makedirs(os.path.dirname(proxy), exist_ok=True)
    if not os.path.exists(proxy):
        _run(["ffmpeg", "-y", "-v", "error", "-i", montage,
              "-vf", f"fps={fps:g},scale=320:-2,format=gray", "-an",
              "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", proxy])
    return proxy


def motion_envelope(proxy=None, fps=PROXY_FPS):
    """Mean abs frame-diff motion energy at ``fps`` over a low-res proxy of the montage."""
    import cv2
    proxy = proxy or _build_proxy()
    cap = cv2.VideoCapture(proxy)
    prev, mot = None, []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float64) if fr.ndim == 3 else fr.astype(np.float64)
        if prev is not None:
            mot.append(float(np.mean(np.abs(g - prev))))
        prev = g
    cap.release()
    return np.asarray(mot)


def anchor(trials_delsys, env, fps=PROXY_FPS):
    """Slide the bout template over the motion envelope; return (atem_of_delsys, windows)."""
    t0 = min(a for a, _ in trials_delsys.values())
    span = max(b for _, b in trials_delsys.values()) - t0
    templ = np.zeros(int(span * fps) + 1)
    for a, b in trials_delsys.values():
        templ[int((a - t0) * fps):int((b - t0) * fps)] = 1.0
    e = (env - env.mean()) / (env.std() + 1e-9)
    L = int(np.argmax(np.correlate(e, templ - templ.mean(), mode="valid")))
    f = lambda td: (np.asarray(td, float) - t0) + L / fps
    windows = {k: (float(f(a)), float(f(b))) for k, (a, b) in trials_delsys.items()}
    return f, windows


def cut_clips(windows, montage=MONTAGE, audio=None, out_dir=OUT_DIR, conditions=CLIP_CONDITIONS):
    """Cut a clip per condition (montage video + de-hummed audio), scaled to 1080p."""
    audio = audio or dehum()
    os.makedirs(out_dir, exist_ok=True)
    out = {}
    for k in conditions:
        a, b = windows[k]
        op = os.path.join(out_dir, f"table_wiping_{k.lower()}.mp4")
        _run(["ffmpeg", "-y", "-v", "error", "-ss", f"{a:.3f}", "-i", montage,
              "-ss", f"{a:.3f}", "-i", audio, "-t", f"{b - a:.3f}",
              "-map", "0:v:0", "-map", "1:a:0", "-vf", "scale=1920:-2",
              "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-c:a", "aac", "-b:a", "192k", op])
        out[k] = op
    return out


def run():
    trials = pp.read_trial_times()
    _, windows = anchor(trials, motion_envelope())
    clips = cut_clips(windows)
    for k, (a, b) in windows.items():
        print(f"{k:7s} ATEM {a:7.1f}-{b:7.1f}s" + (f"  -> {clips[k]}" if k in clips else ""))
    return windows, clips


if __name__ == "__main__":
    run()
