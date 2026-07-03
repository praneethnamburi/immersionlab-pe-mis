"""Markerless (MediaPipe, single phone) vs. marker-based (OptiTrack) wrist tracking,
on the Day-1 table-wiping capture — the "is markerless good enough?" comparison for the
MIT-PE Markerless Motion Capture module (Day 2 S1).

Run in an env with mediapipe + opencv + h5py + matplotlib (e.g. a `pip install
mediapipe opencv-python h5py matplotlib` env). Produces figures/markerless_vs_rig.png.

Pipeline: crop the camera-view quadrant of the ATEM montage -> MediaPipe PoseLandmarker
-> right-wrist pixel trajectory; compare to the OptiTrack right-wrist marker over the
same (Normal) segment. Both recover the same wiping oscillation; markerless is noisier
in the non-rhythmic start.
"""
import os, tempfile, urllib.request
import numpy as np, cv2, h5py, mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.abspath(__file__))
VID  = os.path.join(REPO, "_out", "atem", "table_wiping_normal.mp4")
BUND = os.path.join(REPO, "_out", "bundle", "table_wiping.h5")
OUT  = os.path.join(REPO, "figures", "markerless_vs_rig.png")
CROP = (540, 1080, 960, 1920)   # bottom-right quadrant = camera view of the subject
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/latest/pose_landmarker_full.task")

def get_model():
    p = os.path.join(tempfile.gettempdir(), "pose_landmarker_full.task")
    if not os.path.exists(p):
        print("downloading pose model..."); urllib.request.urlretrieve(MODEL_URL, p)
    return p

def mediapipe_wrist(video, model):
    y0, y1, x0, x1 = CROP; W, H = x1 - x0, y1 - y0
    cap = cv2.VideoCapture(video); fps = cap.get(cv2.CAP_PROP_FPS)
    opts = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=model),
            running_mode=vision.RunningMode.VIDEO, num_poses=1)
    out, i = [], 0
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        while True:
            ok, fr = cap.read()
            if not ok: break
            crop = fr[y0:y1, x0:x1]
            res = lm.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
                int(i / fps * 1000))
            if res.pose_landmarks:
                rw = res.pose_landmarks[0][16]           # right wrist
                out.append((i / fps, rw.x * W, rw.y * H, rw.visibility))
            else:
                out.append((i / fps, np.nan, np.nan, 0.0))
            i += 1
    cap.release()
    return np.array(out)

def optitrack_wrist(bundle):
    with h5py.File(bundle, "r") as f:
        tot = f["mocap/t_ot"][:]
        wrist = 0.5 * (f["mocap/markers/ref_02_RWristThumb"][:] + f["mocap/markers/ref_03_RWristPinky"][:])
        n0, n1 = f["segments"].attrs["Normal"]
    m = (tot >= n0) & (tot <= n1)
    t = tot[m] - tot[m][0]
    w = wrist[m]
    return t, w[:, int(np.nanargmax(w.var(0)))]           # time-from-0, dominant (wiping) axis

def z(a): a = a - np.nanmean(a); return a / np.nanstd(a)
def strokes(s): s = s - s.mean(); return int(((s[:-1] < 0) & (s[1:] >= 0)).sum())

def main():
    mpw = mediapipe_wrist(VID, get_model())
    tv, mpx, vis = mpw[:, 0], mpw[:, 1], mpw[:, 3]
    tS, ot_sig = optitrack_wrist(BUND)
    zmp = z(mpx); zot = z(np.interp(tv, tS, ot_sig))
    if np.corrcoef(zmp, zot)[0, 1] < 0: zot = -zot         # axis-sign convention
    # correct the unknown sub-second video<->OT clock offset (max-correlation lag, within +/-0.5s)
    def corr_at(L):
        a, b = (zmp[L:], zot[:len(zot) - L]) if L >= 0 else (zmp[:len(zmp) + L], zot[-L:])
        return np.corrcoef(a, b)[0, 1]
    lag = max(range(-15, 16), key=corr_at)
    zot = np.roll(zot, lag)                                # small shift on a 954-sample signal
    r_all = np.corrcoef(zmp, zot)[0, 1]
    ss = tv > 3.0                                          # steady-state = rhythmic wiping
    r_ss = np.corrcoef(zmp[ss], zot[ss])[0, 1]
    print(f"lag={lag} frames ({lag/30*1000:.0f} ms)  r(all)={r_all:.3f}  "
          f"r(rhythmic, t>3s)={r_ss:.3f}  strokes MediaPipe={strokes(zmp)} OptiTrack={strokes(zot)}")

    fig, ax = plt.subplots(2, 1, figsize=(9, 5.2), height_ratios=[3, 1])
    ax[0].axvspan(0, 3, color="#f0d9b5", alpha=.35, lw=0)
    ax[0].plot(tv, zot, color="#0b7a52", lw=2, label="OptiTrack wrist (marker rig, 200 Hz)")
    ax[0].plot(tv, zmp, color="#c026d3", lw=1.4, alpha=.85, label="MediaPipe wrist (single phone, markerless)")
    ax[0].set_ylabel("wrist position\n(wiping axis, normalized)")
    ax[0].legend(loc="upper right", fontsize=9)
    ax[0].set_title(f"Markerless vs. the rig, same motion  ·  {strokes(zmp)} strokes each  ·  "
                    f"r = {r_ss:.2f} once rhythmic (shaded = noisy start)", fontsize=11)
    ax[0].grid(alpha=.2)
    ax[1].plot(tv, vis, color="#788a82"); ax[1].set_ylabel("MediaPipe\nvisibility")
    ax[1].set_xlabel("time (s)"); ax[1].set_ylim(0, 1); ax[1].grid(alpha=.2)
    plt.tight_layout(); os.makedirs(os.path.dirname(OUT), exist_ok=True)
    plt.savefig(OUT, dpi=130); print("saved", OUT)

if __name__ == "__main__":
    main()
