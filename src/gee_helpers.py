"""Earth Engine initialization and shared helpers."""
import ee

GEE_PROJECT = 'black-octagon-291810'

# ── Plain Python config (safe at import time — no ee calls) ───────────────────
START_DATE      = '2026-02-01'
END_DATE        = '2026-04-22'
VV_THRESHOLD_DB = -18.0
SPECKLE_RADIUS  = 3
SCALE_M         = 100
PIXEL_AREA_KM2  = (10 * 10) / 1e6   # 10 m pixel → km²

SITE_COORDS = {
    'Qeshm':   [55.96, 26.76],
    'Lavan':   [53.37, 26.80],
    'Shidvar': [53.90, 26.71],
}


def init_ee():
    """Initialize Earth Engine. Prompts auth if needed."""
    try:
        ee.Initialize(project=GEE_PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT)
    print(f"Earth Engine initialized with project: {GEE_PROJECT}")


# ── GEE geometry helpers — call AFTER init_ee() ───────────────────────────────
def get_roi():
    return ee.Geometry.BBox(52.0, 25.5, 57.0, 27.5)


def get_sites():
    return {name: ee.Geometry.Point(coords) for name, coords in SITE_COORDS.items()}


def get_site_buffers(radius_m=30_000):
    return {name: ee.Geometry.Point(coords).buffer(radius_m)
            for name, coords in SITE_COORDS.items()}


# ── Collection loaders ────────────────────────────────────────────────────────
def load_s1(roi=None, start=START_DATE, end=END_DATE):
    """Load Sentinel-1 GRD IW VV descending collection."""
    roi = roi or get_roi()
    return (
        ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterBounds(roi)
        .filterDate(start, end)
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
        .select('VV')
    )


def load_s2(roi=None, start=START_DATE, end=END_DATE, cloud_pct=20):
    """Load cloud-filtered Sentinel-2 SR collection."""
    roi = roi or get_roi()
    return (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(roi)
        .filterDate(start, end)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_pct))
        .select(['B4', 'B3', 'B2', 'B8'])
    )


# ── Image processors ──────────────────────────────────────────────────────────
def speckle_filter(img):
    """Boxcar mean speckle filter, preserves metadata."""
    return (
        img.focal_mean(SPECKLE_RADIUS, 'square', 'pixels')
           .copyProperties(img, ['system:time_start', 'system:index'])
    )


def detect_slick(img):
    """Threshold VV → binary oil-slick mask."""
    return (
        img.lt(VV_THRESHOLD_DB)
           .selfMask()
           .copyProperties(img, ['system:time_start', 'system:index'])
    )
