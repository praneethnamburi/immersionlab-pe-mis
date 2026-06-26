r"""Portable bundle exporter -- the single, Colab-friendly artifact the teaching
notebooks consume.

Everything is pre-aligned onto the **OptiTrack clock** (0 = first mocap frame) and
written to one HDF5 (`table_wiping.h5`) with a JSON manifest. The native-grid US
mp4 lives alongside (referenced by the manifest; not duplicated into the h5).

HDF5 layout::

    /                      attrs: clock_mul_delsys_ot, ot_gate_onset/offset_delsys, description
    /emg/<location>        t_ot, rms_mV               (flexor / extensor RMS envelopes)
    /mocap                 t_ot, hand_speed_mm_s ; attrs: sr
    /mocap/markers/<name>  (n,3) mm positions on the mocap t_ot grid
    /us                    frame_times_ot, motion_proxy, motion_t_ot ; attrs: fps, mp4, n_frames
    /segments              attrs: <Condition> = [start_ot, end_ot]
"""
from __future__ import annotations

import json
import os
import numpy as np

from . import preprocess as pp


def build_bundle(out_dir=None):
    """Write ``table_wiping.h5`` + ``manifest.json`` to ``out_dir`` (default WORKDIR/bundle)."""
    import h5py
    from . import analysis

    out_dir = out_dir or os.path.join(pp.WORKDIR, "bundle")
    os.makedirs(out_dir, exist_ok=True)
    r = pp.run(pp.WORKDIR)
    d2ot = r["d2ot"]
    h5path = os.path.join(out_dir, "table_wiping.h5")

    with h5py.File(h5path, "w") as h:
        h.attrs["description"] = "table_wiping multimodal teaching dataset (MIT-PE); all on the OptiTrack clock"
        h.attrs["clock_mul_delsys_ot"] = float(r["clock_mul"])
        h.attrs["ot_gate_onset_delsys"] = float(r["ot_gate"][0])
        h.attrs["ot_gate_offset_delsys"] = float(r["ot_gate"][1])

        emg = h.create_group("emg")
        for loc, env in r["emg_envelopes"].items():
            sub = emg.create_group(loc)
            sub.create_dataset("t_ot", data=np.asarray(d2ot(env.t), float))
            sub.create_dataset("rms_mV", data=np.asarray(env(), float).ravel())

        o = r["ot"]
        mg = h.create_group("mocap")
        mg.attrs["sr"] = float(o.sr)
        mg.create_dataset("t_ot", data=np.asarray(o.t, float) - float(o.t[0]))
        mg.create_dataset("hand_speed_mm_s", data=np.asarray(r["hand"][1], float))
        mk = mg.create_group("markers")
        for name in o.marker_names:
            mk.create_dataset(name, data=np.asarray(o.pos[name].co, float) * 100.0)  # mm

        mot = analysis.us_motion_energy()
        fts = np.asarray(r["us_frame_times_delsys"], float)
        m = min(len(mot), len(fts))
        ug = h.create_group("us")
        ug.attrs["fps"] = float(r["us_fps"])
        ug.attrs["mp4"] = os.path.basename(pp.comfree_mp4())
        ug.attrs["n_frames"] = int(len(fts))
        ug.create_dataset("frame_times_ot", data=np.asarray(d2ot(fts), float))
        ug.create_dataset("motion_t_ot", data=np.asarray(d2ot(fts[:m]), float))
        ug.create_dataset("motion_proxy", data=mot[:m])

        sg = h.create_group("segments")
        for k, (a, b) in r["trials_ot"].items():
            sg.attrs[k] = np.array([a, b], float)

    manifest = dict(
        h5="table_wiping.h5",
        us_mp4=pp.comfree_mp4(),
        conditions=list(r["trials_ot"]),
        clock_mul_delsys_ot=float(r["clock_mul"]),
        us_fps=float(r["us_fps"]),
        emg_channels=list(r["emg_envelopes"]),
        markers=list(o.marker_names),
        note="All series on the OptiTrack clock (0 = first mocap frame). US mp4 is the "
             "native-grid comfree video; frame i <-> frame_times_ot[i] (c-spike).",
    )
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return h5path


def load_bundle(h5path):
    """Minimal reader -> nested dict of numpy arrays + attrs (for the notebooks / quick checks)."""
    import h5py

    def _grp(g):
        d = {k: (np.asarray(v) if hasattr(v, "shape") else _grp(v)) for k, v in g.items()}
        d.update({f"@{k}": v for k, v in g.attrs.items()})
        return d

    with h5py.File(h5path, "r") as h:
        return _grp(h)


if __name__ == "__main__":
    p = build_bundle()
    print("wrote", p)
    b = load_bundle(p)
    print("groups:", [k for k in b if not k.startswith("@")])
    print("emg:", list(b["emg"]))
    print("us fps:", b["us"]["@fps"], " n_frames:", b["us"]["@n_frames"])
    print("segments:", {k[1:]: v for k, v in b["segments"].items() if k.startswith("@")})
