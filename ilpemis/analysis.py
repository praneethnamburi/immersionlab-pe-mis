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


def us_motion_energy(mp4=None, roi_depth=US_ROI_DEPTH, cache=True):
    """Per-frame tissue-motion proxy: mean |frame_i - frame_{i-1}| over a depth ROI.

    Returns an array of length (n_frames - 1) -- one value per inter-frame interval,
    aligning naturally with the FrameOutput c-spikes (also one per frame). Cached to
    ``WORKDIR/us_motion.npy`` (decoding the 1.5 GB native mp4 is the slow step)."""
    cache_path = os.path.join(pp.WORKDIR, "us_motion.npy")
    if cache and os.path.exists(cache_path):
        return np.load(cache_path)
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
    mot = np.asarray(mot)
    if cache:
        os.makedirs(pp.WORKDIR, exist_ok=True)
        np.save(cache_path, mot)
    return mot


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


def _mask(t, win):
    t = np.asarray(t)
    return (t >= win[0]) & (t <= win[1])


def _cadence(t, v):
    """Wiping cadence as hand-speed peaks per second (robust on short windows; each
    half-stroke gives one speed peak, so this scales with stroke rate)."""
    from scipy.signal import find_peaks
    t = np.asarray(t, float); v = np.asarray(v, float)
    if len(v) < 8:
        return float("nan")
    sr = 1.0 / float(np.median(np.diff(t)))
    pk, _ = find_peaks(v, distance=int(sr * 0.15), prominence=0.3 * np.std(v))
    return len(pk) / (t[-1] - t[0])


def condition_metrics(out_dir=None):
    """Per-condition means: flexor/extensor EMG, co-contraction index, US motion,
    hand speed, and stroke cadence. Returns ``{condition: {...}}``."""
    s = _signals(out_dir)
    conds = [c for c in ("Normal", "Tensed", "Fast") if c in s["trials_ot"]]
    out = {}
    for k in conds:
        w = s["trials_ot"][k]
        fe = float(np.nanmean(s["flex"][1][_mask(s["flex"][0], w)]))
        ee = float(np.nanmean(s["ext"][1][_mask(s["ext"][0], w)]))
        hm = _mask(s["hand"][0], w)
        hv = float(np.nanmean(s["hand"][1][hm]))
        us = float(np.nanmean(s["us"][1][_mask(s["us"][0], w)]))
        sf = _cadence(s["hand"][0][hm], s["hand"][1][hm])
        out[k] = dict(flex=fe, ext=ee, cci=min(fe, ee) / max(fe, ee), hand=hv, us=us, stroke=sf)
    return out, s


def contrasts(out_dir=None):
    """Teaching figure for the two contrasts + a printed stats table.
    Normal-vs-Tensed: EMG amplitude + co-contraction. Normal-vs-Fast: speed + cadence."""
    m, _ = condition_metrics(out_dir)
    conds = list(m)
    print(f"{'cond':8s} {'flexEMG':>9} {'extEMG':>9} {'CCI':>6} {'handSpd':>9} {'USmot':>7} {'spd pk/s':>10}")
    for k in conds:
        d = m[k]
        print(f"{k:8s} {d['flex']:9.4f} {d['ext']:9.4f} {d['cci']:6.2f} {d['hand']:9.1f} {d['us']:7.2f} {d['stroke']:10.2f}")
    if "Normal" in m and "Tensed" in m:
        print(f"\nNormal->Tensed: flexor EMG x{m['Tensed']['flex']/m['Normal']['flex']:.2f}, "
              f"extensor x{m['Tensed']['ext']/m['Normal']['ext']:.2f} (co-contraction; CCI "
              f"{m['Normal']['cci']:.2f}->{m['Tensed']['cci']:.2f})")
    if "Normal" in m and "Fast" in m:
        print(f"Normal->Fast: hand speed x{m['Fast']['hand']/m['Normal']['hand']:.2f}, "
              f"speed-peak rate {m['Normal']['stroke']:.2f}->{m['Fast']['stroke']:.2f} /s")

    figdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
    os.makedirs(figdir, exist_ok=True)
    cc = {"Normal": "#3a7d44", "Tensed": "#b03a3a", "Fast": "#2f6f9f"}
    x = np.arange(len(conds))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    # A: EMG flexor/extensor grouped -> co-contraction in Tensed
    ax[0].bar(x - 0.18, [m[k]["flex"] for k in conds], 0.36, label="flexor", color="#b03a3a")
    ax[0].bar(x + 0.18, [m[k]["ext"] for k in conds], 0.36, label="extensor", color="#2f6f9f")
    ax[0].set_xticks(x); ax[0].set_xticklabels(conds); ax[0].set_ylabel("EMG RMS (mV)")
    ax[0].set_title("EMG amplitude (Normal vs Tensed = co-contraction)"); ax[0].legend()
    # B: hand speed
    ax[1].bar(x, [m[k]["hand"] for k in conds], 0.6, color=[cc[k] for k in conds])
    ax[1].set_xticks(x); ax[1].set_xticklabels(conds); ax[1].set_ylabel("mean hand speed (mm/s)")
    ax[1].set_title("hand speed (Normal vs Fast)")
    # C: stroke cadence
    ax[2].bar(x, [m[k]["stroke"] for k in conds], 0.6, color=[cc[k] for k in conds])
    ax[2].set_xticks(x); ax[2].set_xticklabels(conds); ax[2].set_ylabel("hand-speed peaks / s")
    ax[2].set_title("wiping cadence (Normal vs Fast)")
    for a in ax:
        a.grid(alpha=0.2, axis="y")
    fig.suptitle("table_wiping condition contrasts", fontsize=13)
    fig.tight_layout()
    p = os.path.join(figdir, "contrasts.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"\nsaved {p}")
    return p


if __name__ == "__main__":
    reveal()
    contrasts()
