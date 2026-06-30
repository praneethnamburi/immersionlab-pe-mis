# immersionlab-pe-mis

Preprocessing + teaching materials for the MIT Professional Education course
**Mastering Integrated Systems: Imaging, Machine Learning & Physical AI**
(July 6–9, 2026, MIT.nano Immersion Lab).

This repo will host the public, `pip install`-able helper the course's Colab
notebooks consume. First module: the **preprocessing** of the *table-wiping*
teaching dataset — a replicable record of the load → synchronize → segment →
extract steps.

## Environment

Run in the lab **`b4`** conda env (has `pysampled`, `immersionlab`, `delsys`,
`telemed`, `mithic`, `scipy`, `cv2`, `ffmpeg`). The preprocessing **reuses
validated lab tooling** rather than reimplementing it:

| Step | Tool |
|---|---|
| Delsys CSV → native `.h5`, load | `delsys.process` / `delsys.Log` |
| OptiTrack csv load | `immersionlab.ot.Log` |
| OT recording-gate (TTL) detection | `mithic.synchronization.delsys_gates.detect_ttl_gates` |
| Telemed FrameOutput **c-spikes** | `immersionlab.syncdelsystelemed.detect_frame_pulses` |
| EMG filter + RMS envelope | `projects.wobble._core_algorithms.get_emg_amplitude_rms` |
| COM-free `.tvd` → native mp4 + timing h5 | `telemed.extract_comfree` |

## The dataset (table-wiping)

One continuous trial, one right-handed subject; four task segments —
**Normal, Tensed, Slow (dropped), Fast** — for the contrasts *Normal vs Tensed*
(EMG / co-contraction) and *Normal vs Fast* (kinematics). Modalities:

- **OptiTrack** mocap, 200 Hz — right-arm markers (`ref_00_RIndex`…`ref_07_RShoulder`).
- **Delsys** — EMG R-forearm flexors (Ch1) + extensors (Ch2), and the analog sync
  sensor (Ch13): **A = OT recording gate**, **B = Telemed FrameOutput** (inverted:
  idles high, drops low while recording, one c-spike up per US frame), C/D unused.
- **Telemed** ultrasound — transverse forearm, LF9-5N60 probe, 9 MHz, 50 mm depth,
  native 128 × 884, ~133 fps, 19,367 frames. Tissue is tracked with DUSTrack (two
  points; DLC + LK-RSTC); the **inter-point distance** is the US deformation metric.
- **ATEM** multicam montage (4K) + CAM 4 audio.

## Time model

OptiTrack (mocap) is the **reference clock**; Delsys is the **sync hub**. Map any
Delsys time to OT seconds (0 = first mocap frame):

    t_ot = (t_delsys − ot_gate_onset) / clock_mul_delsys_ot

`clock_mul_delsys_ot = 1.000015411` (mithic `calibration/delsys_ot`, pia02 fit).
The Telemed c-spikes (Delsys-clock, ISI std ~0.16 ms) are the per-frame US
timebase — better than the device `time_ms` (which oscillates 14/16 ms);
`declared − cspikes = 1` here (the modal +1 ladder). COM is **not** needed for
timing or alignment — only for the optional production display pixels.

## Usage

```python
from ilpemis import preprocess as pp
r = pp.run(out_dir="_out")          # load → sync → segment → US c-spike timing
pp.extract_us_comfree()             # COM-free .tvd → native-grid mp4 + timing h5
```

## Status / next

- [x] Load + Delsys→OT sync (gate onset + clock_mul), verified on EMG + mocap.
- [x] EMG flexor/extensor RMS envelopes (wobble); contrasts confirmed
      (Tensed/Normal flexor EMG ≈ 2.7×, Fast/Normal hand speed ≈ 1.4×).
- [x] US per-frame c-spike timing (`detect_frame_pulses`) → OT clock.
- [x] COM-free `.tvd` extraction (native mp4 + timing h5).
- [x] ATEM CAM-4 audio de-hum (60 Hz + harmonics, −20…−41 dB).
- [x] ATEM task-condition snippets (motion-template anchored) + de-hummed audio.
- [x] EMG→US→motion reveal (proxy) + condition-contrast figures (`figures/`).
- [x] Portable bundle exporter + Day-1 Colab teaching notebook (`notebooks/`).
- [x] Final DUSTrack tracking (DLC-corrected + LK-RSTC, `..._dlccorr_lkmovavg_0.500.json`)
      consumed; the **inter-point distance** (point0↔point1) is the ultrasound metric
      (tissue deformation). Reveal + contrasts + bundle use it (frame-diff proxy = fallback).
- [x] Day-2 notebook (tracked points → tissue metric → design-your-own) + take-home
      hosted on Hugging Face (`praneethnamburimit/immersionlab-pe-mis-table-wiping`);
      the notebooks auto-download it in Colab.

## Modules

- `ilpemis/preprocess.py` — load + Delsys→OT sync + EMG envelopes + US c-spike timing + comfree extract.
- `ilpemis/analysis.py` — US tissue-motion proxy, the EMG→US→motion reveal, condition contrasts.
- `ilpemis/atem.py` — motion-template anchor, de-hum, per-condition clips.
- `ilpemis/bundle.py` — `build_bundle` (b4) / `load_bundle` (Colab-light).
- `notebooks/table_wiping_day1.ipynb` — Day-1 teaching notebook (capture → sync → reveal; h5py only).
- `notebooks/table_wiping_day2.ipynb` — Day-2 notebook (tracked points → tissue metric → design-your-own).
