r"""Preprocessing for the MIT-PE *table-wiping* multimodal teaching dataset.

A distilled, replicable record of the steps taken to load, synchronize, segment,
and extract the capture at::

    S:\_Events\20260706 - PE course MIS\table_wiping

Run in the **b4** conda env (has pysampled, immersionlab, delsys, telemed, mithic,
scipy, cv2). This module deliberately **reuses validated lab tooling** rather than
reimplementing anything:

  * ``delsys.process`` / ``delsys.Log``                       -- Delsys CSV -> h5 + load
  * ``immersionlab.ot.Log``                                   -- OptiTrack csv load
  * ``mithic.synchronization.delsys_gates.detect_ttl_gates``  -- recording-gate (TTL) detection
  * ``immersionlab.syncdelsystelemed.detect_frame_pulses``    -- Telemed FrameOutput c-spikes
  * ``projects.wobble._core_algorithms.get_emg_amplitude_rms``-- EMG filter + RMS envelope
  * ``telemed.extract_comfree``                               -- COM-free .tvd -> native mp4 + timing h5

Time model (see the pia02 / mithic timeline architecture): **OptiTrack (mocap) is
the reference clock; Delsys is the sync hub.** Two signals on the Delsys analog
sync sensor anchor everything:

  * **ch A** -- the OptiTrack recording **gate** (TTL: low->high at record start).
  * **ch B** -- the Telemed **FrameOutput** line, *inverted* (idles HIGH, drops to
    a LOW baseline while recording, with a narrow **c-spike UP per ultrasound
    frame**). The c-spikes are the device's per-frame clock, rock-steady on the
    Delsys (ISI std ~0.16 ms) -- a better per-frame timebase than the .tvd
    ``time_ms`` (which oscillates 14/16 ms). ``declared_frames - n_cspikes`` is the
    known small "ladder" (modally +1).

Everything maps to OT seconds (0 = first mocap frame) via::

    t_ot = (t_delsys - ot_gate_onset) / clock_mul_delsys_ot

with ``clock_mul_delsys_ot = 1.000015411`` (mithic ``calibration/delsys_ot``, the
pia02 fit). Channel map for this session (``delsys_channelmap.txt``):
Ch1 EMG R-forearm-flexors, Ch2 EMG R-forearm-extensors, Ch13 sync (A=OT gate,
B=Telemed FrameOutput, C/D disregard).
"""
from __future__ import annotations

import os
import numpy as np

# ----------------------------------------------------------------------------- config
ROOT = r"S:\_Events\20260706 - PE course MIS\table_wiping"
DELSYS_CSV = os.path.join(ROOT, "delsys", "Trial_1.csv")
DELSYS_H5 = os.path.join(ROOT, "delsys", "Trial_1.h5")
CHANNELMAP = os.path.join(ROOT, "delsys_channelmap.txt")
OT_CSV = os.path.join(ROOT, "ot", "table_wiping_001.csv")
TVD = os.path.join(ROOT, "telemed", "20260626 141416 PE course table wiping.tvd")
TRIAL_TIMES = os.path.join(ROOT, "trial_times.txt")

#: local working dir for regenerable artifacts (gitignored)
WORKDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_out"))
COMFREE_DIR = os.path.join(WORKDIR, "comfree")

#: Final DUSTrack tracking (DLC-corrected + LK moving-average filtered, the LK-RSTC
#: output). Two forearm-tissue points; the **inter-point distance** is the ultrasound
#: metric (tissue deformation). 19366 frames == n c-spikes (1:1). JSON shape:
#: ``{point: {frame_idx: [x, y]}}`` with points "0" and "1".
TRACK_JSON = os.path.join(
    ROOT, "telemed",
    "20260626 141416 PE course table wiping_annotations_dlccorr_lkmovavg_0.500.json")

#: analog sync sub-channel indices in delsys.Log.analog.split_to_1d()
CH_OT_GATE = 0          # A
CH_TELEMED_FRAMES = 1   # B
HAND_MARKER = "ref_00_RIndex"
#: conditions to contrast (Slow is captured but dropped per the study design)
CONTRAST = ("Normal", "Tensed", "Fast")


def clock_mul():
    """``clock_mul_delsys_ot`` from the mithic calibration (fallback to the committed value)."""
    try:
        from mithic import _config
        cm = _config.CALIB["clock_mul_delsys_ot"]
        return float(cm["value"]) if isinstance(cm, dict) else float(cm)
    except Exception:
        return 1.000015411


# ----------------------------------------------------------------------------- loaders
def ensure_delsys_h5(csv=DELSYS_CSV, channelmap=CHANNELMAP):
    """Build the native ``.h5`` checkpoint (channelmap + native rates embedded) if missing.

    Mirrors ``delsys.process`` on the folder; loading the ``.h5`` is preferred over
    re-parsing the CSV (the analog sync channel stays at its native 2222 Hz)."""
    h5 = os.path.splitext(csv)[0] + ".h5"
    if not os.path.exists(h5):
        import delsys
        delsys.process(os.path.dirname(csv))    # resolves the sibling channelmap, writes <stem>.h5
    return h5


def load_delsys(h5=DELSYS_H5):
    import delsys
    return delsys.Log(h5)


def load_ot(csv=OT_CSV):
    from immersionlab import ot
    return ot.Log(csv)


def read_trial_times(path=TRIAL_TIMES):
    """Parse ``<Label> - <a> s to <b> s`` lines -> ``{label: (a, b)}`` in **Delsys** seconds."""
    import re
    out = {}
    with open(path) as f:
        for line in f:
            m = re.match(r"\s*(\w+)\s*-\s*([\d.]+)\s*s?\s*to\s*([\d.]+)", line)
            if m:
                out[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return out


# ----------------------------------------------------------------------------- sync
def ot_gate_onset(delsys_log):
    """(onset, offset) of the OptiTrack recording gate on analog ch A, in Delsys seconds."""
    from mithic.synchronization import delsys_gates as dg
    chA = delsys_log.analog.split_to_1d()[CH_OT_GATE]
    on, off = dg.detect_ttl_gates(chA)
    return dg.select_recording(on, off)


def make_d2ot(onset_delsys, cm=None):
    """A function mapping Delsys seconds -> OT seconds-from-recording-start."""
    cm = clock_mul() if cm is None else cm
    return lambda t: (np.asarray(t, float) - onset_delsys) / cm


# ----------------------------------------------------------------------------- modalities
def emg_rms_envelopes(delsys_log):
    """``{location: pysampled.Data}`` RMS envelopes via the wobble EMG pipeline
    (20-500 Hz band, 60 Hz notch, 50 ms RMS @ 240 Hz). Time base = Delsys seconds."""
    from projects.wobble import _core_algorithms as wob
    out = {}
    for s in delsys_log.find(modality="EMG", as_="sensor"):
        loc = getattr(s, "location", None) or getattr(s, "name", None) or f"sensor{getattr(s,'number','?')}"
        out[loc] = wob.get_emg_amplitude_rms(s.emg)
    return out


def hand_speed(ot_log, marker=HAND_MARKER, lowpass_hz=15.0):
    """(t_ot_relative, speed_mm_s) for a marker -- magnitude of the gradient of position."""
    import pysampled
    mk = ot_log.pos[marker]
    pos_mm = np.asarray(mk.co, float) * 100.0      # decimetres -> mm
    vel = np.gradient(pos_mm, 1.0 / ot_log.sr, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    speed = np.asarray(pysampled.Data(speed, sr=ot_log.sr).lowpass(lowpass_hz)()).ravel()
    return np.asarray(ot_log.t) - float(ot_log.t[0]), speed


def us_frame_times(delsys_log):
    """Telemed per-frame **c-spike** times (Delsys seconds) via the validated
    ``detect_frame_pulses`` (centroid; handles sub-sample amplitude beating).
    ``distance=10`` (< one ~133 fps frame interval at 2222 Hz)."""
    from immersionlab.syncdelsystelemed import detect_frame_pulses
    chB = delsys_log.analog.split_to_1d()[CH_TELEMED_FRAMES]
    x = np.asarray(chB()).ravel()
    try:
        t = np.asarray(chB.t).ravel()
    except Exception:
        t = np.arange(len(x)) / float(chB.sr)
    return detect_frame_pulses(x, t, distance=10)


# ----------------------------------------------------------------------------- US video (COM-free)
def extract_us_comfree(tvd=TVD, out_dir=None):
    """COM-free ``.tvd`` -> native-grid mp4 (DLC-tracking-ready) + timing/metadata h5."""
    import telemed
    out_dir = out_dir or COMFREE_DIR
    return telemed.extract_comfree(tvd, out_dir, progress=True)


def comfree_mp4(comfree_dir=COMFREE_DIR):
    """Path to the native-grid US mp4 produced by :func:`extract_us_comfree`."""
    import glob
    hits = sorted(glob.glob(os.path.join(comfree_dir, "*.mp4")))
    if not hits:
        raise FileNotFoundError(f"no comfree mp4 in {comfree_dir}; run extract_us_comfree()")
    return hits[0]


def tracked_points(track_json=TRACK_JSON):
    """Final tracked tissue points -> ``{name: (n_frames, 2)}`` xy arrays, frame-ordered.

    Source = the DUSTrack ``..._dlccorr_lkmovavg_*.json`` (DLC correction + LK-RSTC filter).
    JSON shape ``{point: {frame_idx: [x, y]}}``; any missing frame is NaN."""
    import json
    with open(track_json) as f:
        d = json.load(f)
    out = {}
    for name, frames in d.items():
        a = np.full((len(frames), 2), np.nan)
        for k, xy in frames.items():
            a[int(k)] = xy
        out[name] = a
    return out


# ----------------------------------------------------------------------------- orchestration
def run(out_dir):
    """Load -> sync -> segment -> US c-spike timing; return a dict of aligned products."""
    os.makedirs(out_dir, exist_ok=True)
    ensure_delsys_h5()
    d = load_delsys()
    o = load_ot()
    onset, offset = ot_gate_onset(d)
    d2ot = make_d2ot(onset)

    env = emg_rms_envelopes(d)
    ht, hv = hand_speed(o)
    frame_t = us_frame_times(d)
    trials = read_trial_times()
    seg_ot = {k: (float(d2ot(a)), float(d2ot(b))) for k, (a, b) in trials.items()}

    return dict(
        ot=o, delsys=d, ot_gate=(onset, offset), clock_mul=clock_mul(), d2ot=d2ot,
        emg_envelopes=env, hand=(ht, hv),
        us_frame_times_delsys=frame_t, us_frame_times_ot=d2ot(frame_t),
        us_fps=float(1.0 / np.median(np.diff(frame_t))) if len(frame_t) > 2 else None,
        trials_delsys=trials, trials_ot=seg_ot,
    )


if __name__ == "__main__":
    r = run(os.path.join(os.path.dirname(__file__), "..", "_out"))
    g = r["ot_gate"]
    print(f"OT gate (delsys): {g[0]:.4f}-{g[1]:.4f}  clock_mul={r['clock_mul']:.9f}")
    print(f"EMG channels: {list(r['emg_envelopes'])}")
    print(f"US c-spikes: n={len(r['us_frame_times_delsys'])}  fps={r['us_fps']:.2f}")
    for k, (a, b) in r["trials_ot"].items():
        print(f"  {k:7s} OT {a:7.2f}-{b:7.2f} s")
