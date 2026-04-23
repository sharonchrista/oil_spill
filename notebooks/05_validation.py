"""
Step 5 — Validation against Sentinel-2 optical imagery
Cross-validates SAR slick masks against cloud-free S2 scenes. Computes IoU.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import ee
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from gee_helpers import (
    init_ee, load_s1, load_s2, speckle_filter, detect_slick,
    get_site_buffers, SCALE_M
)

init_ee()

SITE_BUFFERS = get_site_buffers()
s1_slicks    = load_s1().map(speckle_filter).map(detect_slick)
s2_col       = load_s2()

def detect_oil_optical(img):
    ndwi  = img.normalizedDifference(['B3', 'B8'])
    water = ndwi.gt(0.1)
    dark  = img.select('B4').lt(500).And(img.select('B3').lt(600))
    return dark.And(water).selfMask().rename('oil_optical')

print("Computing IoU scores (may take 2–3 min)...")
iou_rows = []
for site, buf in SITE_BUFFERS.items():
    n_s2    = s2_col.filterBounds(buf).size().getInfo()
    s2_list = s2_col.filterBounds(buf).toList(50)
    if n_s2 == 0:
        print(f"  {site}: no cloud-free S2 scenes — skipping")
        continue
    for i in range(min(n_s2, 5)):
        s2_img  = ee.Image(s2_list.get(i))
        s2_date = datetime.fromtimestamp(s2_img.get('system:time_start').getInfo()/1000)
        w_start = (s2_date - pd.Timedelta(days=6)).strftime('%Y-%m-%d')
        w_end   = (s2_date + pd.Timedelta(days=6)).strftime('%Y-%m-%d')
        s1_near = s1_slicks.filterDate(w_start, w_end).filterBounds(buf).first()
        if s1_near.getInfo() is None:
            continue
        opt = detect_oil_optical(s2_img).unmask(0)
        sar = s1_near.unmask(0)
        i_val = sar.And(opt).reduceRegion(ee.Reducer.sum(), buf, SCALE_M, maxPixels=1e10).get('VV').getInfo() or 0
        u_val = sar.Or(opt).reduceRegion(ee.Reducer.sum(), buf, SCALE_M, maxPixels=1e10).get('VV').getInfo() or 1
        iou   = i_val / u_val
        iou_rows.append({'site':site,'s2_date':s2_date.strftime('%Y-%m-%d'),'iou':round(iou,4)})
        print(f"  {site} | {s2_date.date()} | IoU = {iou:.4f}")

os.makedirs('../outputs/tables', exist_ok=True)
if iou_rows:
    df_iou = pd.DataFrame(iou_rows)
    df_iou.to_csv('../outputs/tables/iou_scores.csv', index=False)
    print("\n── IoU summary ──────────────────────────────")
    print(df_iou.groupby('site')['iou'].describe().round(3).to_string())

    COLORS = {'Qeshm':'#E24B4A','Lavan':'#378ADD','Shidvar':'#EF9F27'}
    fig, ax = plt.subplots(figsize=(9,4))
    for site, grp in df_iou.groupby('site'):
        ax.bar(grp['s2_date'], grp['iou'], color=COLORS.get(site,'#888'), label=site, width=0.4, alpha=0.85)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8)
    ax.set_title('SAR vs Optical Validation — IoU Scores', fontsize=12)
    ax.set_ylabel('IoU')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')
    plt.xticks(rotation=30)
    plt.tight_layout()
    os.makedirs('../outputs/figures', exist_ok=True)
    fig.savefig('../outputs/figures/iou_validation.png', dpi=150, bbox_inches='tight')
    print("Saved → outputs/figures/iou_validation.png")
    plt.show()
else:
    print("No S1/S2 coincident pairs found — validation skipped.")

print("\nAll steps complete. Check outputs/ for all figures and tables.")
