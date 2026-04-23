"""
Step 3 — Drift vector estimation (v4)
Fix: pick best orbit per site individually (not one global orbit).
Also lowers the slick_px threshold to 1 to catch smaller detections.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import ee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from math import atan2, degrees, radians, sin, cos, sqrt
from collections import Counter
from gee_helpers import (
    init_ee, get_site_buffers,
    SCALE_M, START_DATE, END_DATE, SPECKLE_RADIUS
)

init_ee()

SITE_BUFFERS = get_site_buffers(radius_m=25_000)

# ── Helper: build S1 collection for one orbit number ──────────────────────────
def s1_for_orbit(orbit_num):
    from gee_helpers import get_roi
    return (
        ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterBounds(get_roi())
        .filterDate(START_DATE, END_DATE)
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
        .filter(ee.Filter.eq('relativeOrbitNumber_start', orbit_num))
        .select('VV')
        .map(lambda img: img.focal_mean(SPECKLE_RADIUS, 'square', 'pixels')
                            .copyProperties(img, ['system:time_start', 'system:index']))
    )

# ── Step A: find best orbit per site ──────────────────────────────────────────
def best_orbit_for_site(site_name, buffer_geom):
    """Return the orbit number with most scenes covering this buffer."""
    from gee_helpers import get_roi
    col = (
        ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterBounds(buffer_geom)          # filter to THIS site buffer
        .filterDate(START_DATE, END_DATE)
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
    )
    orbits = col.aggregate_array('relativeOrbitNumber_start').getInfo()
    if not orbits:
        return None, 0
    cnt = Counter(orbits)
    best_orb, best_n = cnt.most_common(1)[0]
    return best_orb, best_n

print("Finding best orbit per site...")
site_orbits = {}
for site_name, buf in SITE_BUFFERS.items():
    orb, n = best_orbit_for_site(site_name, buf)
    site_orbits[site_name] = orb
    print(f"  {site_name}: orbit {orb} ({n} scenes)")

# ── Step B: compute centroids per site using its best orbit ───────────────────
def get_centroids_for_site(site_name, buffer_geom, orbit_num):
    if orbit_num is None:
        print(f"  {site_name}: no orbit found, skipping")
        return []

    col = s1_for_orbit(orbit_num).filterBounds(buffer_geom)
    n   = col.size().getInfo()
    print(f"  {site_name} orbit {orbit_num}: {n} scenes")
    if n == 0:
        return []

    lat_lon = ee.Image.pixelLonLat()

    def centroid_per_img(img):
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean().combine(
                ee.Reducer.stdDev(), sharedInputs=True),
            geometry=buffer_geom, scale=200, maxPixels=1e9
        )
        vv_mean = ee.Number(stats.get('VV_mean'))
        vv_std  = ee.Number(stats.get('VV_stdDev'))
        threshold = vv_mean.subtract(vv_std.multiply(1.5))

        slick_mask = img.lt(threshold)
        px_count   = slick_mask.unmask(0).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=buffer_geom, scale=SCALE_M, maxPixels=1e10
        ).get('VV')

        coords = lat_lon.updateMask(slick_mask)
        cstats = coords.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=buffer_geom, scale=SCALE_M, maxPixels=1e10
        )
        return img.set({
            'centroid_lon': cstats.get('longitude'),
            'centroid_lat': cstats.get('latitude'),
            'slick_px':     px_count,
        })

    with_c = col.map(centroid_per_img)
    times  = with_c.aggregate_array('system:time_start').getInfo()
    lons   = with_c.aggregate_array('centroid_lon').getInfo()
    lats   = with_c.aggregate_array('centroid_lat').getInfo()
    pixels = with_c.aggregate_array('slick_px').getInfo()

    rows = []
    for t, lon, lat, px in zip(times, lons, lats, pixels):
        # Keep any scene where centroid was computable (px >= 1)
        if lon is not None and lat is not None and (px or 0) >= 1:
            rows.append({
                'site':     site_name,
                'orbit':    orbit_num,
                'date':     datetime.fromtimestamp(t / 1000),
                'lon':      round(lon, 6),
                'lat':      round(lat, 6),
                'slick_px': int(px),
            })

    print(f"    → {len(rows)} scenes with detected slick")
    return rows


print("\nComputing centroids (per-site best orbit)...")
all_rows = []
for site_name, buf in SITE_BUFFERS.items():
    all_rows.extend(get_centroids_for_site(site_name, buf, site_orbits[site_name]))

if not all_rows:
    print("\nERROR: No centroids found for any site.")
    print("Try lowering the slick_px threshold or widening the buffer.")
    sys.exit(1)

df_c = pd.DataFrame(all_rows).sort_values(['site','date']).reset_index(drop=True)

# ── Step C: outlier removal (2σ on lat and lon per site) ──────────────────────
def remove_outliers(grp):
    for col in ['lat', 'lon']:
        med = grp[col].median()
        std = grp[col].std()
        if std > 0:
            grp = grp[np.abs(grp[col] - med) < 2 * std]
    return grp

df_c = df_c.groupby('site', group_keys=False).apply(remove_outliers).reset_index(drop=True)

os.makedirs('../data/processed', exist_ok=True)
df_c.to_csv('../data/processed/centroids.csv', index=False)
print(f"\nSaved {len(df_c)} centroids → data/processed/centroids.csv")

print("\nCentroid spread per site (std lat/lon — should be < 0.05°):")
print(df_c.groupby('site')[['lat','lon']].std().round(5).to_string())

print("\nScene counts per site:")
print(df_c.groupby('site')['date'].count().to_string())

# ── Step D: drift vectors ──────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    a = (sin(radians(lat2-lat1)/2)**2 +
         cos(radians(lat1)) * cos(radians(lat2)) *
         sin(radians(lon2-lon1)/2)**2)
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def bearing_deg(lat1, lon1, lat2, lon2):
    x = sin(radians(lon2-lon1)) * cos(radians(lat2))
    y = (cos(radians(lat1)) * sin(radians(lat2)) -
         sin(radians(lat1)) * cos(radians(lat2)) * cos(radians(lon2-lon1)))
    return (degrees(atan2(x, y)) + 360) % 360

drift_rows = []
for site, grp in df_c.groupby('site'):
    grp = grp.sort_values('date').reset_index(drop=True)
    for i in range(1, len(grp)):
        prev, curr = grp.iloc[i-1], grp.iloc[i]
        dt_days = (curr['date'] - prev['date']).total_seconds() / 86400
        if dt_days < 1:
            continue
        dist_km = haversine_km(prev['lat'], prev['lon'], curr['lat'], curr['lon'])
        drift_rows.append({
            'site':         site,
            'date_from':    prev['date'],
            'date_to':      curr['date'],
            'dt_days':      round(dt_days, 1),
            'dist_km':      round(dist_km, 3),
            'speed_km_day': round(dist_km / dt_days, 3),
            'bearing_deg':  round(bearing_deg(prev['lat'], prev['lon'],
                                              curr['lat'], curr['lon']), 1),
            'dlon':         round(curr['lon'] - prev['lon'], 6),
            'dlat':         round(curr['lat'] - prev['lat'], 6),
        })

df_drift = pd.DataFrame(drift_rows)
df_drift.to_csv('../data/processed/drift_vectors.csv', index=False)

print("\n── Drift summary per site ────────────────────────")
if not df_drift.empty:
    print(df_drift.groupby('site')[['speed_km_day','bearing_deg']].agg(
        ['mean','std','min','max']).round(2).to_string())
else:
    print("  No drift vectors computed (need ≥2 valid centroid scenes per site).")

# ── Step E: drift map ──────────────────────────────────────────────────────────
COLORS  = {'Qeshm':'#C0392B', 'Lavan':'#2471A3', 'Shidvar':'#D68910'}
MARKERS = {'Qeshm':'o',       'Lavan':'s',        'Shidvar':'^'}

plt.rcParams.update({'font.family':'serif','font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False})

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_facecolor('#EDF4FB')

for site, grp in df_c.groupby('site'):
    grp = grp.sort_values('date')
    c, m = COLORS[site], MARKERS[site]
    ax.plot(grp['lon'], grp['lat'], marker=m, color=c,
            linewidth=1.6, markersize=6, label=site, zorder=3, alpha=0.85)
    for idx, label in [(0, grp['date'].iloc[0].strftime('%b %d')),
                       (-1, grp['date'].iloc[-1].strftime('%b %d'))]:
        ax.annotate(label, xy=(grp['lon'].iloc[idx], grp['lat'].iloc[idx]),
                    fontsize=7.5, color=c, xytext=(5, 3), textcoords='offset points')
    ax.annotate(site, xy=(grp['lon'].mean(), grp['lat'].mean()),
                fontsize=10, fontweight='bold', color=c,
                xytext=(0, 13), textcoords='offset points', ha='center')

if not df_drift.empty:
    for site, grp in df_drift.groupby('site'):
        mdlon = grp['dlon'].mean()
        mdlat = grp['dlat'].mean()
        mspd  = grp['speed_km_day'].mean()
        mbrg  = grp['bearing_deg'].mean()
        start = df_c[df_c['site']==site].sort_values('date').iloc[0]
        ax.annotate('',
            xy=(start['lon'] + mdlon*5, start['lat'] + mdlat*5),
            xytext=(start['lon'], start['lat']),
            arrowprops=dict(arrowstyle='->', color=COLORS[site],
                            lw=2.5, mutation_scale=16), zorder=5)
        ax.text(start['lon'] + mdlon*5 + 0.02, start['lat'] + mdlat*5,
                f'{mspd:.1f} km/d\n{mbrg:.0f}°',
                fontsize=8, color=COLORS[site], va='center')

shidvar_lon, shidvar_lat = 53.90, 26.71
ax.add_patch(plt.Circle((shidvar_lon, shidvar_lat), 0.10,
             color='#D68910', fill=True, alpha=0.18, zorder=2))
ax.add_patch(plt.Circle((shidvar_lon, shidvar_lat), 0.10,
             color='#D68910', fill=False, linewidth=1.8, linestyle='--', zorder=4))
ax.annotate('Shidvar protected zone',
            xy=(shidvar_lon, shidvar_lat - 0.12),
            ha='center', fontsize=8.5, color='#7D6608', style='italic')

ax.set_xlabel('Longitude (°E)', fontsize=11)
ax.set_ylabel('Latitude (°N)', fontsize=11)
ax.set_title('Oil Spill Centroid Drift — Persian Gulf (Feb–Apr 2026)\n'
             '[Per-site dominant orbit · Outlier-filtered centroids]',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3, linewidth=0.5)
plt.tight_layout()

os.makedirs('../outputs/figures', exist_ok=True)
fig.savefig('../outputs/figures/drift_map.png', dpi=150, bbox_inches='tight')
print("\nSaved → outputs/figures/drift_map.png")
plt.show()
print("\nStep 3 complete. Run 04_risk_model.py next.")