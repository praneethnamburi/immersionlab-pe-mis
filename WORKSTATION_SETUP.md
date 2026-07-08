# Workstation setup plan — for Claude to execute

**Read this whole file, then execute the steps in order.** You are running on one of the
lab workstations for the MIT-PE course *Mastering Integrated Systems* (July 6–9, 2026).
Praneeth has signed in with his Claude credentials and will sign out when you're done.

## Goal
Make **this workstation** ready for student use during the course:
1. The **three course Colab notebooks** open in the browser (the primary path when wifi is good).
2. **DUSTrack** installed and **GPU-verified** (for the Day-2 ultrasound demo/hands-on).
3. A **local copy of the dataset** so the notebooks also run **offline** — the wifi fallback if the
   room network struggles with ~9 people at once.

Do all three. Prefer idempotent commands (safe to re-run). Print a clear PASS/FAIL for each check.

## Guardrails
- **Do not store, print, or log any credentials.** Do not `git push` or modify any remote.
- Work under a single course directory: **`C:\pe-course\`** (create it). Don't touch other user data.
- If a step needs a decision you can't safely make, **stop and leave a note** in
  `C:\pe-course\SETUP-STATUS.txt` describing what's blocked, then continue with the other steps.

## Base facts you'll need
- Course repo: `https://github.com/praneethnamburi/immersionlab-pe-mis` (public).
- Dataset: Hugging Face `praneethnamburimit/immersionlab-pe-mis-table-wiping`, file `table_wiping.h5`
  (public, no token). The notebooks look for a local bundle at `<repo>/_out/bundle/table_wiping.h5`
  (or `../_out/bundle` when run from `notebooks/`, or `$ILPEMIS_BUNDLE`) before falling back to HF.
- The three notebooks (in `<repo>/notebooks/`), and their Colab URLs:
  - `Day1_S3_multimodal_reveal.ipynb` → https://colab.research.google.com/github/praneethnamburi/immersionlab-pe-mis/blob/main/notebooks/Day1_S3_multimodal_reveal.ipynb
  - `Day2_S1_markerless_mocap.ipynb` → https://colab.research.google.com/github/praneethnamburi/immersionlab-pe-mis/blob/main/notebooks/Day2_S1_markerless_mocap.ipynb
  - `Day2_S2_ultrasound_tissue_tracking.ipynb` → https://colab.research.google.com/github/praneethnamburi/immersionlab-pe-mis/blob/main/notebooks/Day2_S2_ultrasound_tissue_tracking.ipynb

---

## Step 1 — Check prerequisites
Run and record results:
- `python --version` (need 3.10–3.11) and whether **conda/miniconda** is present.
- `git --version`.
- `nvidia-smi` — confirm an NVIDIA GPU + driver (DUSTrack needs CUDA).
- Confirm a browser exists (Chrome/Edge).

If Python or git is missing, install miniconda + git first (or note the blocker and skip the local-env steps — the Colab tabs in Step 6 still work as long as there's internet).

## Step 2 — Get the course repo
```
mkdir C:\pe-course  2>NUL
git clone https://github.com/praneethnamburi/immersionlab-pe-mis.git C:\pe-course\immersionlab-pe-mis
```
(If it already exists, `git -C C:\pe-course\immersionlab-pe-mis pull`.)

## Step 3 — Local env for the notebooks (offline fallback)
Create an env with exactly what the Colab notebooks install, so they can run locally with no internet:
```
conda create -y -n pe-mis python=3.11
conda activate pe-mis
pip install numpy h5py matplotlib opencv-python-headless mediapipe huggingface_hub jupyterlab ipython
```

## Step 4 — Download the dataset locally
From the repo root, put `table_wiping.h5` where the notebooks look for it:
```
cd C:\pe-course\immersionlab-pe-mis
python -c "import os,shutil;from huggingface_hub import hf_hub_download;os.makedirs('_out/bundle',exist_ok=True);p=hf_hub_download('praneethnamburimit/immersionlab-pe-mis-table-wiping','table_wiping.h5',repo_type='dataset');shutil.copy(p,'_out/bundle/table_wiping.h5');print('bundle at',os.path.abspath('_out/bundle/table_wiping.h5'))"
```
**Check:** `_out/bundle/table_wiping.h5` exists and is > 100 MB.

## Step 5 — DUSTrack (install if missing) + GPU check
DUSTrack may already be installed on this station. Verify first; install only if missing.
```
python -c "import torch;print('CUDA:',torch.cuda.is_available())"
python -c "import dustrack;print('dustrack OK')"
```
- If `dustrack` imports and CUDA is True → done.
- If `dustrack` is missing: install it in its own env (it pulls DeepLabCut + a CUDA-enabled PyTorch —
  this is the heavy one). Use a dedicated env, e.g. `dlc-dustrack`, and install a CUDA build of torch
  matching this machine's driver, then `pip install dustrack`. If the CUDA/torch/DLC combo won't
  resolve cleanly, **stop and note it** in `SETUP-STATUS.txt` (this is the one step most likely to need
  a human) — the notebooks and Colab tabs don't depend on it; only the Day-2 S2 tracking demo does.
**Check:** `dustrack OK` prints and `CUDA: True`.

## Step 6 — Open the three Colab notebooks in the browser
Pre-load them so an instructor/student just has to run them:
```
start "" "https://colab.research.google.com/github/praneethnamburi/immersionlab-pe-mis/blob/main/notebooks/Day1_S3_multimodal_reveal.ipynb"
start "" "https://colab.research.google.com/github/praneethnamburi/immersionlab-pe-mis/blob/main/notebooks/Day2_S1_markerless_mocap.ipynb"
start "" "https://colab.research.google.com/github/praneethnamburi/immersionlab-pe-mis/blob/main/notebooks/Day2_S2_ultrasound_tissue_tracking.ipynb"
```

## Step 7 — Verify the notebooks run OFFLINE (with the local bundle) + warm caches
From `<repo>/notebooks` (so `../_out/bundle` is found), execute each notebook once. This both verifies
them and caches the small extra downloads (the MediaPipe model, the sample clip) locally:
```
cd C:\pe-course\immersionlab-pe-mis\notebooks
set MPLBACKEND=Agg
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 Day1_S3_multimodal_reveal.ipynb --output _check_day1.ipynb
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 Day2_S1_markerless_mocap.ipynb --output _check_day2s1.ipynb
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 Day2_S2_ultrasound_tissue_tracking.ipynb --output _check_day2s2.ipynb
```
Each should complete without error. Delete the `_check_*.ipynb` outputs after. If a notebook errors,
note which cell in `SETUP-STATUS.txt`.
> The offline fallback for students is: open these notebooks in **local JupyterLab** (`jupyter lab`
> from `<repo>/notebooks`) — they auto-use `../_out/bundle/table_wiping.h5`, so no internet is needed.

## Step 8 — Write the status report
Write `C:\pe-course\SETUP-STATUS.txt` with a PASS/FAIL line for each check below, plus the machine name
and the local bundle path. Leave the three Colab tabs open.

## Final checklist (all should PASS)
- [ ] `nvidia-smi` shows an NVIDIA GPU.
- [ ] repo cloned at `C:\pe-course\immersionlab-pe-mis`.
- [ ] `pe-mis` conda env created with the notebook deps.
- [ ] `_out/bundle/table_wiping.h5` present (> 100 MB).
- [ ] `torch.cuda.is_available()` is True **and** `import dustrack` works (or a clear note if not).
- [ ] all three notebooks executed end-to-end offline with the local bundle.
- [ ] the three Colab notebooks are open in the browser.

When done, tell Praneeth it's ready (or what's blocked), and he'll sign out.
