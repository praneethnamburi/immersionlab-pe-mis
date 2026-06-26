r"""Cross-modal analysis + teaching figures for the table-wiping dataset.

Builds on :mod:`ilpemis.preprocess` (load + Delsys->OT sync + US c-spike timing).
The headline product is the **EMG -> ultrasound -> motion** reveal: the chain that
is invisible in any single stream and only appears once everything is on the
OptiTrack clock.

US tissue motion (proxy): per-frame mean-abs frame-difference in a forearm depth
ROI of the COM-free native-grid mp4 (no DLC needed to *see* the chain; DUSTrack
tracking upgrades it later). The frame-diff series (n_frames-1) is placed on the
OT clock at the FrameOutput c-spike times.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import preprocess as pp

US_ROI_DEPTH = (100, 800)        # native depth rows (active=884); skip near-probe + far field


def us_motion_energy(mp4=None, roi_depth=US_ROI_DEPTH):
    """Per-frame tissue-motion proxy: mean |frame_i - frame_{i-1}| over a depth ROI.

    Returns an array of length (n_frames - 1) -- one value per inter-frame interval,
    aligning naturally with the FrameOutput c-spikes (also one per frame)."""
    import cv2
    mp4 = mp4 or pp.comfree_mp4()
    cap = cv2.VideoCapture(mp4)
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open {mp4}")
    prev, mot = None, []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = fr[:, :, 0] if fr.ndim == 3 else fr
        g = g[roi_depth[0]:roi_depth[1], :].astype(np.float32)
        if prev is not None:
            mot.append(float(np.mean(np.abs(g - prev))))
        prev = g
    cap.release()
    return np.asarray(mot)


def _signals(out_dir=None):
    """Assemble the three OT-clock signals for the reveal: flexor EMG, US motion, hand speed."""
    r = pp.run(out_dir or pp.WORKDIR)
    d2ot = r["d2ot"]
    flex = r["emg_envelopes"]["RForearmFlexors"]
    ext = r["emg_envelopes"]["RForearmExtensors"]
    fe_t, fe_v = np.asarray(d2ot(flex.t)), np.asarray(flex()).ravel()
    ee_t, ee_v = np.asarray(d2ot(ext.t)), np.asarray(ext()).ravel()
    ht, hv = r["hand"]
    mot = us_motion_energy()
    fts = np.asarray(r["us_frame_times_delsys"])
    m = min(len(mot), len(fts))
    us_t, us_v = np.asarray(d2ot(fts[:m])), mot[:m]
    return dict(r=r, flex=(fe_t, fe_v), ext=(ee_t, ee_v), us=(us_t, us_v),
                hand=(ht, hv), trials_ot=r["trials_ot"])


def reveal(out_dir=None, zoom=(14.0, 19.0)):
    """The Day-1 payoff: flexor EMG / US tissue motion / hand speed on the OT clock.
    Writes a full-session overview + a zoomed cascade to ``WORKDIR``/figures."""
    s = _signals(out_dir)
    figdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
    os.makedirs(figdir, exist_ok=True)
    colors = {"Normal": "#3a7d44", "Tensed": "#b03a3a", "Slow": "#999", "Fast": "#2f6f9f"}
    rows = [("flexor EMG\nRMS (mV)", s["flex"], "#b03a3a"),
            ("US tissue motion\n(ROI frame-diff)", s["us"], "#7a3fa0"),
            ("hand speed\nRIndex (mm/s)", s["hand"], "#333333")]

    def _draw(xlim, fname, title, shade):
        fig, ax = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
        for a, (lbl, (t, v), c) in zip(ax, rows):
            a.plot(t, v, color=c, lw=0.7)
            a.set_ylabel(lbl)
            a.grid(alpha=0.2)
            a.set_xlim(xlim)
            if shade:
                for k, (x0, x1) in s["trials_ot"].items():
                    a.axvspan(x0, x1, color=colors.get(k, "#ccc"), alpha=0.10)
        ax[0].set_title(title)
        ax[-1].set_xlabel("OptiTrack time (s from recording start)")
        fig.tight_layout(); fig.savefig(os.path.join(figdir, fname), dpi=130); plt.close(fig)
        return os.path.join(figdir, fname)

    full = s["us"][0]
    p1 = _draw((0, float(np.nanmax(full)) if len(full) else 145),
               "reveal_overview.png",
               "table_wiping: EMG -> ultrasound -> motion, aligned on the OptiTrack clock", True)
    p2 = _draw(zoom, "reveal_zoom.png",
               f"the cascade, zoomed {zoom[0]:.0f}-{zoom[1]:.0f}s (Normal): EMG -> tissue -> hand", False)
    print(f"saved {p1}\nsaved {p2}")
    return p1, p2


if __name__ == "__main__":
    reveal()
