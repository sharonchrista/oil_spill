"""
Step 2 — Slick segmentation & area time series
Fixed: descending orbit only + adaptive threshold + rolling median
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import ee
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from gee_helpers import (
    init_ee, get_site_buffers, PIXEL_AREA_KM2, SCALE_M,
    START_DATE, END_DATE, SPECKLE_RADIUS
)

init_ee()

SITE_BUFFERS = get_site_buffers(radius_m=25_000)

# ── Load DESCENDING only (more scenes over Persian Gulf, consistent geometry) ──
def load_s1_desc():
    from gee_helpers import get_roi
    roi = get_roi()
    return (
        ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterBounds(roi)
        .filterDate(START_DATE, END_DATE)
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
        .select('VV')
        .map(lambda img: img.focal_mean(SPECKLE_RADIUS, 'square', 'pixels')
                            .copyProperties(img, ['system:time_start', 'system:index']))
    )

s1 = load_s1_desc()
n_desc = s1.size().getInfo()
print(f"Descending scenes: {n_desc}")

# ── Adaptive per-scene threshold ───────────────────────────────────────────────
def get_area_adaptive(site_name, buffer_geom):
    """
    Per-scene threshold = scene_mean - 1.5 * scene_std over local water pixels.
    Eliminates bias from varying wind/incidence angle across dates.
    """
    def area_per_img(img):
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean().combine(
                ee.Reducer.stdDev(), sharedInputs=True),
            geometry=buffer_geom,
            scale=200,
            maxPixels=1e9
        )
        threshold = (ee.Number(stats.get('VV_mean'))
                       .subtract(ee.Number(stats.get('VV_stdDev')).multiply(1.5)))
        slick_px = img.lt(threshold).unmask(0).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=buffer_geom,
            scale=SCALE_M,
            maxPixels=1e10
        ).get('VV')
        return img.set('slick_px', slick_px)

    with_area = s1.filterBounds(buffer_geom).map(area_per_img)
    times  = with_area.aggregate_array('system:time_start').getInfo()
    pixels = with_area.aggregate_array('slick_px').getInfo()

    return [{
        'site':     site_name,
        'date':     datetime.fromtimestamp(t / 1000),
        'area_km2': (px or 0) * PIXEL_AREA_KM2,
    } for t, px in zip(times, pixels)]


print("Computing area time series (descending orbit, adaptive threshold)...")
all_rows = []
for site_name, buf in SITE_BUFFERS.items():
    print(f"  {site_name}...")
    all_rows.extend(get_area_adaptive(site_name, buf))

df = pd.DataFrame(all_rows).sort_values(['site','date']).reset_index(drop=True)

# ── Rolling median (3-scene) to suppress remaining noise ──────────────────────
df['area_smooth'] = (
    df.groupby('site')['area_km2']
      .transform(lambda x: x.rolling(3, min_periods=1, center=True).median())
)
df['growth_pct'] = df.groupby('site')['area_smooth'].pct_change() * 100

os.makedirs('../data/processed', exist_ok=True)
os.makedirs('../outputs/tables', exist_ok=True)
os.makedirs('../outputs/figures', exist_ok=True)
df.to_csv('../data/processed/area_timeseries.csv', index=False)
print("Saved → data/processed/area_timeseries.csv")

summary = df.groupby('site').agg(
    scenes        =('area_km2',    'count'),
    max_area_km2  =('area_km2',    'max'),
    mean_smooth   =('area_smooth', 'mean'),
    peak_growth   =('growth_pct',  'max'),
).round(3)
print("\n── Area summary ──────────────────────────────────")
print(summary.to_string())
summary.to_csv('../outputs/tables/summary_stats.csv')

# ── Plot ───────────────────────────────────────────────────────────────────────
COLORS = {'Qeshm':'#E24B4A', 'Lavan':'#378ADD', 'Shidvar':'#EF9F27'}

fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
fig.suptitle('Persian Gulf Oil Spill — Area Over Time (km²)\n[Descending orbit · Adaptive threshold · 3-scene median]',
             fontsize=13, fontweight='bold')

for ax, (site, grp) in zip(axes, df.groupby('site')):
    grp = grp.sort_values('date')
    c   = COLORS[site]

    # Raw detections as faint bars
    ax.bar(grp['date'], grp['area_km2'], width=0.8,
           color=c, alpha=0.18, label='Raw (adaptive threshold)')

    # Smoothed line — the real signal
    ax.plot(grp['date'], grp['area_smooth'], 'o-',
            color=c, linewidth=2.2, markersize=5, label='Smoothed (3-scene median)')

    # Polynomial trend on smoothed
    grp_nonzero = grp[grp['area_smooth'] > 0]
    if len(grp_nonzero) >= 4:
        x_num = mdates.date2num(grp_nonzero['date'])
        z     = np.polyfit(x_num, grp_nonzero['area_smooth'], 2)
        x_fit = np.linspace(x_num.min(), x_num.max(), 300)
        y_fit = np.poly1d(z)(x_fit)
        y_fit = np.clip(y_fit, 0, None)
        ax.plot(mdates.num2date(x_fit), y_fit,
                '--', color=c, alpha=0.65, linewidth=1.8, label='Trend (poly-2)')

    # Annotate peak
    peak_row = grp.loc[grp['area_smooth'].idxmax()]
    ax.annotate(f"Peak: {peak_row['area_smooth']:.1f} km²",
                xy=(peak_row['date'], peak_row['area_smooth']),
                xytext=(10, 6), textcoords='offset points',
                fontsize=8, color=c, fontweight='bold')

    ax.set_ylabel('Area (km²)', fontsize=10)
    ax.set_title(site, fontsize=11, fontweight='bold', color=c)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.2)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

plt.xticks(rotation=30)
plt.tight_layout()
fig.savefig('../outputs/figures/area_timeseries.png', dpi=150, bbox_inches='tight')
print("Saved → outputs/figures/area_timeseries.png")
plt.show()
print("\nStep 2 complete. Run 03_drift_estimation.py next.")