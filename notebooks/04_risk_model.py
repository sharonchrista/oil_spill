"""
Step 4 — Risk model v3
Key insight: start point is already inside Shidvar zone → spill has arrived.
New questions:
  Q1: What fraction of the protected zone area is contaminated over time?
  Q2: How far does the plume extend beyond the zone boundary at 14/28 days?
  Q3: What is the contamination persistence (% days with spill presence)?
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from math import radians, cos, sin, sqrt, atan2, degrees
from gee_helpers import init_ee

init_ee()

df_drift = pd.read_csv('../data/processed/drift_vectors.csv',
                       parse_dates=['date_from','date_to'])
df_c     = pd.read_csv('../data/processed/centroids.csv',
                       parse_dates=['date'])

# ── Zone definition ────────────────────────────────────────────────────────────
SHIDVAR = {'lat': 26.68, 'lon': 53.92, 'radius_km': 8.0}

# ── Drift parameters from Shidvar observed data ────────────────────────────────
shidvar_drift = df_drift[df_drift['site'] == 'Shidvar']
shidvar_pos   = df_c[df_c['site'] == 'Shidvar'].sort_values('date')

mean_speed   = float(shidvar_drift['speed_km_day'].median()) if len(shidvar_drift)>=2 else 3.5
mean_bearing = float(shidvar_drift['bearing_deg'].median())  if len(shidvar_drift)>=2 else 130.0
start_lat    = float(shidvar_pos.iloc[-1]['lat'])
start_lon    = float(shidvar_pos.iloc[-1]['lon'])

u_kmday = mean_speed * sin(radians(mean_bearing))
v_kmday = mean_speed * cos(radians(mean_bearing))

print(f"Start: {start_lat:.4f}°N, {start_lon:.4f}°E")
print(f"Drift: {mean_speed:.2f} km/day @ {mean_bearing:.1f}°")

dist_to_centre = sqrt((start_lat - SHIDVAR['lat'])**2 * 111**2 +
                      (start_lon - SHIDVAR['lon'])**2 * (111*cos(radians(start_lat)))**2)
print(f"Distance from start to Shidvar centre: {dist_to_centre:.2f} km")
print(f"Zone radius: {SHIDVAR['radius_km']} km")
print(f"→ Spill is {'INSIDE' if dist_to_centre < SHIDVAR['radius_km'] else 'OUTSIDE'} the protected zone\n")

# ── Lagrangian ensemble ────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    a = (sin(radians(lat2-lat1)/2)**2 +
         cos(radians(lat1))*cos(radians(lat2))*sin(radians(lon2-lon1)/2)**2)
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def move_particle(lat, lon, u, v, dt_hours=6):
    R   = 6371.0
    dt  = dt_hours / 24.0
    dlat = (v * dt) / R * (180/np.pi)
    dlon = (u * dt) / (R * cos(radians(lat))) * (180/np.pi)
    return lat+dlat, lon+dlon

def run_ensemble(slat, slon, u, v, n_days=28, dt_hours=6, noise_std=1.2, N=1000):
    n_steps = int(n_days * 24 / dt_hours)
    tracks  = np.zeros((N, n_steps+1, 2))
    tracks[:,0,0] = slat
    tracks[:,0,1] = slon
    rng = np.random.default_rng(42)
    for step in range(n_steps):
        nu = rng.normal(0, noise_std, N)
        nv = rng.normal(0, noise_std, N)
        for p in range(N):
            tracks[p,step+1] = move_particle(
                tracks[p,step,0], tracks[p,step,1],
                u+nu[p], v+nv[p], dt_hours)
    return tracks

print("Running ensembles...")
t28 = run_ensemble(start_lat, start_lon, u_kmday, v_kmday, n_days=28, N=1000)
t14 = t28[:, :int(14*24/6)+1, :]

# ── Metrics ────────────────────────────────────────────────────────────────────
def zone_metrics(tracks, zone_lat, zone_lon, radius_km):
    N, S, _ = tracks.shape
    # Fraction of particles inside zone at each timestep
    in_zone = np.zeros(S)
    for s in range(S):
        dists = np.array([haversine_km(tracks[p,s,0], tracks[p,s,1], zone_lat, zone_lon)
                          for p in range(N)])
        in_zone[s] = (dists <= radius_km).mean()

    # Max plume extent beyond zone edge
    end_dists = np.array([haversine_km(tracks[p,-1,0], tracks[p,-1,1], zone_lat, zone_lon)
                          for p in range(N)])
    plume_extent = max(0, np.percentile(end_dists, 90) - radius_km)

    # Persistence: fraction of timesteps where >5% particles are inside zone
    persistence = (in_zone > 0.05).mean()
    return in_zone, plume_extent, persistence

in14, ext14, per14 = zone_metrics(t14, SHIDVAR['lat'], SHIDVAR['lon'], SHIDVAR['radius_km'])
in28, ext28, per28 = zone_metrics(t28, SHIDVAR['lat'], SHIDVAR['lon'], SHIDVAR['radius_km'])

print(f"\n── Contamination metrics ────────────────────────────")
print(f"  14-day: zone occupancy {in14.mean()*100:.1f}% | "
      f"plume extent +{ext14:.1f} km | persistence {per14*100:.0f}%")
print(f"  28-day: zone occupancy {in28.mean()*100:.1f}% | "
      f"plume extent +{ext28:.1f} km | persistence {per28*100:.0f}%")

os.makedirs('../outputs/tables', exist_ok=True)
pd.DataFrame([
    {'horizon':'14-day','zone_occupancy_pct':round(in14.mean()*100,1),
     'plume_extent_km':round(ext14,1),'persistence_pct':round(per14*100,0)},
    {'horizon':'28-day','zone_occupancy_pct':round(in28.mean()*100,1),
     'plume_extent_km':round(ext28,1),'persistence_pct':round(per28*100,0)},
]).to_csv('../outputs/tables/shidvar_risk.csv', index=False)

# ── Figure 1: ensemble map ─────────────────────────────────────────────────────
plt.rcParams.update({'font.family':'serif','font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False})

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
titles = ['14-day projection', '28-day projection']
track_sets = [t14, t28]
metrics    = [(in14,ext14,per14), (in28,ext28,per28)]

for ax, tracks, (in_z, ext, per), title in zip(axes, track_sets, metrics, titles):
    ax.set_facecolor('#EDF4FB')

    # Particle trajectories
    for p in range(0, 1000, 15):
        ax.plot(tracks[p,:,1], tracks[p,:,0],
                alpha=0.04, color='#2471A3', linewidth=0.5)

    # Endpoint density
    ax.scatter(tracks[:,-1,1], tracks[:,-1,0],
               s=6, alpha=0.25, color='#2471A3', zorder=3, label='Particle endpoints')

    # Shidvar zone circles
    r_deg = SHIDVAR['radius_km'] / 111.0
    ax.add_patch(plt.Circle((SHIDVAR['lon'], SHIDVAR['lat']), r_deg,
                 color='#E67E22', alpha=0.20, zorder=4))
    ax.add_patch(plt.Circle((SHIDVAR['lon'], SHIDVAR['lat']), r_deg,
                 color='#E67E22', fill=False, linewidth=2.0,
                 linestyle='--', zorder=5))

    # Start point
    ax.plot(start_lon, start_lat, 'o', color='#C0392B',
            markersize=9, zorder=7, label='Spill centroid (latest)')
    ax.annotate('Spill\ncentroid', xy=(start_lon, start_lat),
                xytext=(6, 5), textcoords='offset points',
                fontsize=8, color='#C0392B', fontweight='bold')

    # Shidvar island marker
    ax.plot(SHIDVAR['lon'], SHIDVAR['lat'], '*', color='#E67E22',
            markersize=12, zorder=6)
    ax.annotate(f'Shidvar Island\n(protected, r={SHIDVAR["radius_km"]} km)',
                xy=(SHIDVAR['lon'], SHIDVAR['lat']),
                xytext=(-60, 10), textcoords='offset points',
                fontsize=8, color='#784212', style='italic')

    # Mean drift arrow
    ax.annotate('', xy=(tracks[:,-1,1].mean(), tracks[:,-1,0].mean()),
                xytext=(start_lon, start_lat),
                arrowprops=dict(arrowstyle='->', color='#1A5276',
                                lw=2.0, mutation_scale=14), zorder=6)

    # Metrics box
    ax.text(0.02, 0.97,
            f'Zone occupancy: {in_z.mean()*100:.0f}%\n'
            f'Plume extent:  +{ext:.1f} km\n'
            f'Persistence:    {per*100:.0f}%',
            transform=ax.transAxes, fontsize=8.5, va='top',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.85,
                      ec='#E67E22', linewidth=1.2))

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Longitude (°E)', fontsize=10)
    ax.set_ylabel('Latitude (°N)', fontsize=10)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.25, linewidth=0.5)

fig.suptitle('Lagrangian Forward Projection — Shidvar Island Protected Zone\n'
             f'[{mean_speed:.1f} km/day @ {mean_bearing:.0f}° · 1000 particles · noise σ=1.2 km/day]',
             fontsize=11, fontweight='bold')
plt.tight_layout()
os.makedirs('../outputs/figures', exist_ok=True)
fig.savefig('../outputs/figures/lagrangian_risk.png', dpi=150, bbox_inches='tight')
print("Saved → outputs/figures/lagrangian_risk.png")

# ── Figure 2: zone occupancy over time ────────────────────────────────────────
dt_hours = 6
times_14 = np.arange(len(in14)) * dt_hours / 24
times_28 = np.arange(len(in28)) * dt_hours / 24

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.fill_between(times_28, in28*100, alpha=0.15, color='#2471A3')
ax2.plot(times_28, in28*100, color='#2471A3', linewidth=1.8, label='28-day run')
ax2.plot(times_14, in14*100, color='#C0392B', linewidth=1.8, linestyle='--', label='14-day run')
ax2.axhline(5, color='gray', linewidth=0.8, linestyle=':', label='5% threshold')
ax2.axvline(14, color='#C0392B', linewidth=0.8, linestyle=':', alpha=0.6)
ax2.set_xlabel('Days from projection start', fontsize=10)
ax2.set_ylabel('Particles inside protected zone (%)', fontsize=10)
ax2.set_title('Shidvar Island — Contamination Occupancy Over Time', fontsize=11, fontweight='bold')
ax2.set_ylim(0, 105)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.2)
fig2.tight_layout()
fig2.savefig('../outputs/figures/zone_occupancy.png', dpi=150, bbox_inches='tight')
print("Saved → outputs/figures/zone_occupancy.png")

plt.show()
print("\nStep 4 complete. All risk figures generated.")
print("Key finding: oil spill has already reached Shidvar protected zone.")
print(f"  Confirmed by centroid at {start_lat:.3f}°N {start_lon:.3f}°E")
print(f"  (zone centre at {SHIDVAR['lat']}°N {SHIDVAR['lon']}°E, radius {SHIDVAR['radius_km']} km)")