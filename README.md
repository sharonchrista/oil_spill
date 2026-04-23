# Persian Gulf Oil Spill — SAR Spread Estimation

Reproducible Python + Google Earth Engine pipeline for detecting, quantifying, and forward-projecting conflict-induced oil spills using Sentinel-1 C-band SAR time-series analysis.

---

## Key Findings

| Site | Peak Area | Mean Area | Drift Speed | Bearing |
|------|-----------|-----------|-------------|---------|
| Lavan Island | 1.83 km² | 1.03 ± 0.51 km² | 1.27 km/day | 226° (S) |
| Qeshm Island | 1.59 km² | 1.20 ± 0.34 km² | — | — |
| Shidvar Island (MPA) | 1.87 km² | 1.36 ± 0.35 km² | 1.79 km/day | 130° (SE) |

**Shidvar Island Marine Protected Area has been breached.** Lagrangian modeling confirms oil plume transit of the MPA with a 3.2-day occupancy window and eastward plume extension of 22.6 km (14-day) to 47.4 km (28-day).

---

## Project Structure

```
oil_spill/
├── src/
│   └── gee_helpers.py          # GEE init, loaders, processors (shared helpers)
├── notebooks/
│   ├── 01_data_acquisition.py  # Scene inventory & backscatter stats
│   ├── 02_segmentation.py      # Adaptive threshold, area time series
│   ├── 03_drift_estimation.py  # Centroid tracking, drift vectors
│   ├── 04_risk_model.py        # Lagrangian particle ensemble
│   ├── 05_validation.py        # SAR vs Sentinel-2 optical IoU
│   └── 06_paper_figures_and_results.py  # Publication figures + results text
├── data/
│   ├── raw/
│   │   └── scene_inventory.csv       # All 132 S1 scenes (dates, orbits)
│   └── processed/
│       ├── area_timeseries.csv       # Area (km²) per site per date
│       ├── centroids.csv             # Slick centroid lat/lon per scene
│       └── drift_vectors.csv         # Speed, bearing, distance per step
├── outputs/
│   ├── figures/                      # All publication figures (300 dpi PNG)
│   ├── tables/
│   │   ├── summary_stats.csv         # Peak/mean area per site
│   │   ├── iou_scores.csv            # SAR vs optical validation scores
│   │   └── shidvar_risk.csv          # Lagrangian risk metrics
│   ├── results_section.txt           # Auto-filled Results + Discussion text
│   └── persian_gulf_oil_spill_paper.docx  # Complete draft paper
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone and create environment

```bash
git clone https://github.com/YOUR_USERNAME/oil_spill.git
cd oil_spill
conda create -n uhi python=3.11
conda activate uhi
pip install -r requirements.txt
```

### 2. Authenticate Google Earth Engine

```bash
earthengine authenticate
```

This opens a browser window. Sign in with the Google account linked to your GEE project (`black-octagon-291810`). The token is saved locally and does not need to be repeated.

### 3. Run the pipeline in order

```bash
cd notebooks
python 01_data_acquisition.py   # ~2 min  — scene inventory
python 02_segmentation.py       # ~5 min  — area time series
python 03_drift_estimation.py   # ~5 min  — centroid drift
python 04_risk_model.py         # ~3 min  — Lagrangian risk
python 05_validation.py         # ~5 min  — IoU validation
python 06_paper_figures_and_results.py  # ~3 min — paper figures
```

Each script saves its outputs to `data/processed/` or `outputs/` before the next script reads them.

---

## Data Sources

| Dataset | Provider | Access | Description |
|---------|----------|--------|-------------|
| Sentinel-1 GRD | ESA / GEE | Free | C-band SAR, IW mode, VV pol, 10 m |
| Sentinel-2 SR | ESA / GEE | Free | Optical validation, 10 m |
| HYCOM currents | NOAA | Free | 1/12° surface u/v (optional, for drift validation) |

All data accessed via [Google Earth Engine](https://code.earthengine.google.com) — no manual download required.

---

## Methodology Summary

```
Sentinel-1 GRD (132 scenes)
        │
        ▼
  Speckle filter (3×3 boxcar focal mean)
        │
        ▼
  Per-scene adaptive threshold  (T = μ - 1.5σ)
        │
        ▼
  Binary slick mask  ──► Area (km²) per scene
        │
        ▼
  Centroid (lat/lon) per scene  ──► Drift vectors (speed, bearing)
        │
        ▼
  Lagrangian ensemble (N=1000, σ=1.2 km/day, Δt=6h)
        │
        ▼
  Protected zone occupancy, plume extent, persistence
```

**Key methodological decisions:**
- **Single orbit per site** — dominant relative orbit number selected per site to eliminate look-angle artifacts from mixed ascending/descending passes
- **Adaptive threshold** — per-scene `mean − 1.5σ` over site buffer adapts to wind-driven sea clutter variation; avoids 10–15× false positive rate of fixed −18 dB threshold
- **3-scene rolling median** — suppresses remaining noise without over-smoothing genuine area changes

---

## Output Figures

| File | Description |
|------|-------------|
| `fig1_area_timeseries.png` | 3-panel area time series with trend lines and event markers |
| `fig2_drift_map.png` | Centroid trajectory map + speed/bearing panels |
| `fig3_zone_occupancy.png` | Shidvar MPA contamination occupancy over time |
| `lagrangian_risk.png` | Particle ensemble at 14-day and 28-day horizons |
| `iou_validation.png` | SAR vs Sentinel-2 optical IoU validation scores |

---

## Environment

Tested on:
- Windows 11 / Miniconda 3
- Python 3.11 (conda env: `uhi`)
- `earthengine-api` 0.1.390+
- GEE project: `black-octagon-291810`

---

## License

Code: MIT License
Data: derived from ESA Copernicus open data (CC BY 4.0)
Paper text: © the authors, all rights reserved
