#!/usr/bin/env python3
"""
================================================================================
IISER MOHALI CAMPUS CAR TRACKING SIMULATION  (v2 — high-contrast edition)
Kalman Filter / Extended Kalman Filter / Particle Filter comparison
================================================================================
"""

import numpy as np
import matplotlib
import os

# ---------------------------------------------------------------------
# BACKEND FIX: the old code forced matplotlib.use("Agg") unconditionally.
# Agg is a headless, *non-interactive* backend -- calling plt.show() on
# it is a silent no-op (no window, no error, no file). That's the #1
# reason this script looked "broken": running it with no arguments
# produced nothing at all, with no error message to explain why.
#
# Fix: only force Agg when there's genuinely no display to draw to
# (servers, containers, CI, this sandbox). On a normal desktop with a
# display, let matplotlib pick an interactive backend so plt.show()
# actually opens a window.
# ---------------------------------------------------------------------
_HAS_DISPLAY = bool(os.environ.get("DISPLAY")) or os.name == "nt" or os.uname().sysname == "Darwin"
if not _HAS_DISPLAY:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers the 3d projection)
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# 0. SHARED VISUAL PALETTE  (single source of truth for every plot)
# =====================================================================

PALETTE = {
    "bg":          "#0a0e27",
    "panel_edge":  "#333333",
    "grid":        "#ffffff",
    "road":        "#5b6478",   # neutral grey -- no longer clashes with EKF
    "measured":    "#8892b0",   # dim slate -- raw noisy GPS fix
    "true":        "#ffffff",   # true car position marker
    "true_edge":   "#ffcc00",

    "kf":          "#00e676",   # green   -- solid,  thick
    "ekf":         "#ffb300",   # amber   -- dashed, medium
    "pf":          "#ff3d81",   # magenta -- dotted, thin

    "title":       "#00ff88",
    "subtitle":    "#ee82ee",
    "text":        "#00ff88",
}

LINESTYLES = {"kf": "-", "ekf": "--", "pf": ":"}
LINEWIDTHS = {"kf": 4.0, "ekf": 2.6, "pf": 2.0}
MARKER_EVERY = 12  # sparse trail markers so a still frame still reads clearly

OUTLINE = [pe.withStroke(linewidth=1.6, foreground="#000814", alpha=0.55)]


# =====================================================================
# 1. CAMPUS CONFIGURATION
# =====================================================================

class IISERMohaliConfig:
    CAMPUS_LAT_MIN = 30.6580
    CAMPUS_LAT_MAX = 30.6700
    CAMPUS_LON_MIN = 76.7200
    CAMPUS_LON_MAX = 76.7380

    M_PER_DEG_LAT = 111000.0
    M_PER_DEG_LON = 96000.0

    FIGURE_SIZE = (18, 11)
    ANIMATION_INTERVAL = 60

    KF_PROCESS_NOISE = 5e-6
    KF_MEASUREMENT_NOISE = 1e-5
    EKF_PROCESS_NOISE = 8e-6
    EKF_MEASUREMENT_NOISE = 1.2e-5
    PF_PROCESS_NOISE_M = 28.0
    PF_MEASUREMENT_NOISE_M = 15.0

    GPS_ERROR_DEG = 0.00012
    NUM_LAPS = 2

    GRID_SIGMA_MULT = 4.0
    GRID_MIN_EXTENT_M = 6.0
    GRID_MAX_EXTENT_M = 60.0
    GRID_MIN_RES = 12
    GRID_MAX_RES = 45
    GRID_COV_EPS_M2 = 1.0


class IISERMohaliCampus:
    BUILDINGS = {
        "Main Gate": {"lat": 30.6652, "lon": 76.7314, "color": "#ff6b6b"},
        "Director Office": {"lat": 30.6645, "lon": 76.7320, "color": "#ff9500"},
        "Registrar Office": {"lat": 30.6648, "lon": 76.7325, "color": "#ff9500"},
        "Lecture Hall Complex": {"lat": 30.6644, "lon": 76.7276, "color": "#4ecdc4"},
        "Academic Block A": {"lat": 30.6650, "lon": 76.7265, "color": "#45b7d1"},
        "Academic Block B": {"lat": 30.6635, "lon": 76.7260, "color": "#45b7d1"},
        "Science Building": {"lat": 30.6641, "lon": 76.7245, "color": "#45b7d1"},
        "Central Library": {"lat": 30.6636, "lon": 76.7291, "color": "#f7b731"},
        "Reading Room": {"lat": 30.6638, "lon": 76.7305, "color": "#f7b731"},
        "Hostel 1 (Boys)": {"lat": 30.6620, "lon": 76.7285, "color": "#a55eea"},
        "Hostel 2 (Girls)": {"lat": 30.6625, "lon": 76.7275, "color": "#ee5a6f"},
        "Hostel 5": {"lat": 30.6601, "lon": 76.7310, "color": "#ee5a6f"},
        "Hostel 8": {"lat": 30.6611, "lon": 76.7268, "color": "#a55eea"},
        "Hostel 10": {"lat": 30.6628, "lon": 76.7245, "color": "#a55eea"},
        "Sports Complex": {"lat": 30.6625, "lon": 76.7325, "color": "#0be881"},
        "Basketball Court": {"lat": 30.6630, "lon": 76.7330, "color": "#0be881"},
        "Football Ground": {"lat": 30.6615, "lon": 76.7340, "color": "#0be881"},
        "Dining Hall": {"lat": 30.6615, "lon": 76.7300, "color": "#ff6348"},
        "Cafeteria": {"lat": 30.6650, "lon": 76.7310, "color": "#ff6348"},
        "Medical Center": {"lat": 30.6670, "lon": 76.7290, "color": "#ff1744"},
        "Security Office": {"lat": 30.6660, "lon": 76.7320, "color": "#424242"},
        "Guest House": {"lat": 30.6680, "lon": 76.7280, "color": "#8c9eff"},
        "Maintenance": {"lat": 30.6590, "lon": 76.7250, "color": "#9e9e9e"},
        "Generator Room": {"lat": 30.6585, "lon": 76.7330, "color": "#9e9e9e"},
    }

    MAIN_LOOP = [
        (30.6652, 76.7314), (30.6645, 76.7330), (30.6625, 76.7345),
        (30.6610, 76.7335), (30.6600, 76.7310), (30.6595, 76.7270),
        (30.6610, 76.7240), (30.6650, 76.7245), (30.6652, 76.7314),
    ]

    ROAD_NETWORK = {
        "Main Loop": MAIN_LOOP,
        "Academic Road": [
            (30.6644, 76.7276), (30.6650, 76.7265), (30.6635, 76.7260), (30.6641, 76.7245),
        ],
        "Hostel Road": [
            (30.6620, 76.7285), (30.6625, 76.7275), (30.6611, 76.7268), (30.6601, 76.7310),
        ],
    }


# =====================================================================
# 2. KALMAN / EKF / PARTICLE FILTERS
# =====================================================================

class StandardKalmanFilter:
    def __init__(self, dt=1.0, process_noise=5e-6, measurement_noise=1e-5):
        self.dt = dt
        self.X = np.zeros((4, 1))
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        self.Q = np.eye(4) * process_noise
        self.Q[2:, 2:] *= 10
        self.R = np.eye(2) * measurement_noise
        self.P = np.eye(4) * 1e-4
        self.P[2:, 2:] *= 1e-6

    def predict(self):
        self.X = self.F @ self.X
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.X[:2].flatten()

    def update(self, measurement):
        Z = np.array(measurement).reshape(2, 1)
        y = Z - self.H @ self.X
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.X = self.X + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.X[:2].flatten()


class ExtendedKalmanFilter:
    def __init__(self, dt=1.0, process_noise=8e-6, measurement_noise=1.2e-5):
        self.dt = dt
        self.X = np.zeros((6, 1))
        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], dtype=np.float64)
        self.Q = np.eye(6) * process_noise
        self.Q[2:4, 2:4] *= 8
        self.Q[4:6, 4:6] *= 15
        self.R = np.eye(2) * measurement_noise
        self.P = np.eye(6) * 1e-4
        self.P[2:4, 2:4] *= 1e-6
        self.velocity_damping = 0.94

    def _state_transition_jacobian(self):
        F_jac = np.eye(6, dtype=np.float64)
        F_jac[0, 2] = self.dt
        F_jac[1, 3] = self.dt
        F_jac[2, 5] = self.dt
        F_jac[3, 5] = self.dt
        return F_jac

    def _f(self, x):
        x_new = x.copy()
        x_new[0] += x[2] * self.dt
        x_new[1] += x[3] * self.dt
        vel_magnitude = np.sqrt(x[2, 0] ** 2 + x[3, 0] ** 2)
        if vel_magnitude > 1e-8:
            angle = np.arctan2(x[3, 0], x[2, 0])
            angle += x[4, 0] * self.dt
            x_new[2, 0] = vel_magnitude * np.cos(angle)
            x_new[3, 0] = vel_magnitude * np.sin(angle)
        x_new[2, 0] += x[5, 0] * self.dt
        x_new[3, 0] += x[5, 0] * self.dt
        x_new[2:4] *= self.velocity_damping
        return x_new

    def predict(self):
        self.X = self._f(self.X)
        F_jac = self._state_transition_jacobian()
        self.P = F_jac @ self.P @ F_jac.T + self.Q
        return self.X[:2].flatten()

    def update(self, measurement):
        Z = np.array(measurement).reshape(2, 1)
        y = Z - self.H @ self.X
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.X = self.X + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.X[:2].flatten()


class ParticleFilter:
    def __init__(self, num_particles=300, dt=1.0, process_noise_m=28.0,
                 measurement_noise_m=15.0,
                 m_per_deg_lat=111000.0, m_per_deg_lon=96000.0):
        self.num_particles = num_particles
        self.dt = dt
        self.process_noise_m = process_noise_m
        self.process_noise_lon_deg = process_noise_m / m_per_deg_lon
        self.process_noise_lat_deg = process_noise_m / m_per_deg_lat
        self.measurement_noise_m = measurement_noise_m
        self.m_per_deg_lat = m_per_deg_lat
        self.m_per_deg_lon = m_per_deg_lon

        self.particles = np.zeros((num_particles, 4))
        self.weights = np.ones(num_particles) / num_particles
        self.X = np.zeros((4, 1))

    def _motion_model(self, particles):
        n = self.num_particles
        particles_new = particles.copy()
        particles_new[:, 0] += particles[:, 2] * self.dt + np.random.normal(0, self.process_noise_lon_deg, n)
        particles_new[:, 1] += particles[:, 3] * self.dt + np.random.normal(0, self.process_noise_lat_deg, n)
        particles_new[:, 2] = particles[:, 2] * 0.6 + np.random.normal(0, self.process_noise_lon_deg, n)
        particles_new[:, 3] = particles[:, 3] * 0.6 + np.random.normal(0, self.process_noise_lat_deg, n)
        return particles_new

    def _measurement_model(self, particles, measurement):
        z = np.array(measurement)
        dx_m = (particles[:, 0] - z[0]) * self.m_per_deg_lon
        dy_m = (particles[:, 1] - z[1]) * self.m_per_deg_lat
        distances_m = np.sqrt(dx_m ** 2 + dy_m ** 2)
        likelihood = np.exp(-distances_m ** 2 / (2 * self.measurement_noise_m ** 2))
        total = np.sum(likelihood)
        if total < 1e-300:
            return np.ones(self.num_particles) / self.num_particles
        return likelihood / total

    def predict(self):
        self.particles = self._motion_model(self.particles)
        self.X[:2, 0] = np.average(self.particles[:, :2], axis=0, weights=self.weights)
        self.X[2:, 0] = np.average(self.particles[:, 2:], axis=0, weights=self.weights)
        return self.X[:2].flatten()

    def update(self, measurement):
        self.weights = self._measurement_model(self.particles, measurement)
        weight_sum = np.sum(self.weights)
        if weight_sum < 1e-300:
            self.weights = np.ones(self.num_particles) / self.num_particles
        else:
            self.weights /= weight_sum

        n_eff = 1.0 / np.sum(self.weights ** 2)
        if n_eff < self.num_particles * 0.5:
            self._resample()

        self.X[:2, 0] = np.average(self.particles[:, :2], axis=0, weights=self.weights)
        self.X[2:, 0] = np.average(self.particles[:, 2:], axis=0, weights=self.weights)
        return self.X[:2].flatten()

    def _resample(self):
        indices = np.argsort(self.weights)[::-1]
        cum_weights = np.cumsum(self.weights[indices])
        cum_weights[-1] = 1.0
        new_indices = []
        u = np.random.uniform(0, 1.0 / self.num_particles)
        j = 0
        for _ in range(self.num_particles):
            while u > cum_weights[j] and j < self.num_particles - 1:
                j += 1
            new_indices.append(indices[j])
            u += 1.0 / self.num_particles
        self.particles = self.particles[new_indices].copy()
        self.weights = np.ones(self.num_particles) / self.num_particles


# =====================================================================
# 3. BUILD THE CAR'S CONTINUOUS MULTI-LAP ROUTE
# =====================================================================

def build_patrol_route(num_laps=2, steps_per_segment=15):
    waypoints = np.array(IISERMohaliCampus.MAIN_LOOP)

    lats, lons = [], []
    for lap in range(num_laps):
        for i in range(len(waypoints) - 1):
            lats.extend(np.linspace(waypoints[i][0], waypoints[i + 1][0], steps_per_segment, endpoint=False))
            lons.extend(np.linspace(waypoints[i][1], waypoints[i + 1][1], steps_per_segment, endpoint=False))
    return np.array(lats), np.array(lons)


def compute_adaptive_confidence_grid(pf, config):
    mean_lon, mean_lat = pf.X[0, 0], pf.X[1, 0]

    diffs_deg = pf.particles[:, :2] - np.array([mean_lon, mean_lat])
    diffs_m = np.column_stack([
        diffs_deg[:, 0] * config.M_PER_DEG_LON,
        diffs_deg[:, 1] * config.M_PER_DEG_LAT,
    ])

    w = pf.weights
    cov = (diffs_m * w[:, None]).T @ diffs_m
    cov += np.eye(2) * config.GRID_COV_EPS_M2

    eigvals = np.linalg.eigvalsh(cov)
    sigma_max_m = np.sqrt(max(eigvals.max(), 1e-6))

    extent_m = np.clip(config.GRID_SIGMA_MULT * sigma_max_m, config.GRID_MIN_EXTENT_M, config.GRID_MAX_EXTENT_M)
    frac_tight = 1.0 - (extent_m - config.GRID_MIN_EXTENT_M) / (config.GRID_MAX_EXTENT_M - config.GRID_MIN_EXTENT_M)
    resolution = int(config.GRID_MIN_RES + frac_tight * (config.GRID_MAX_RES - config.GRID_MIN_RES))

    inv_cov = np.linalg.inv(cov)
    axis = np.linspace(-extent_m, extent_m, resolution)
    XX, YY = np.meshgrid(axis, axis)
    quad = (inv_cov[0, 0] * XX**2 + 2 * inv_cov[0, 1] * XX * YY + inv_cov[1, 1] * YY**2)
    ZZ = np.exp(-0.5 * quad)

    return XX, YY, ZZ, extent_m, resolution, diffs_m


def car_marker(heading_deg):
    return MarkerStyle('^', transform=Affine2D().rotate_deg(heading_deg - 90))


# =====================================================================
# 4. SIMULATION SETUP
# =====================================================================

def _run_filters_full(config=IISERMohaliConfig):
    true_lats, true_lons = build_patrol_route(num_laps=config.NUM_LAPS, steps_per_segment=15)
    total_frames = len(true_lats)

    np.random.seed(7)
    measured_lats = true_lats + np.random.normal(0, config.GPS_ERROR_DEG, total_frames)
    measured_lons = true_lons + np.random.normal(0, config.GPS_ERROR_DEG, total_frames)

    dlat = np.gradient(true_lats)
    dlon = np.gradient(true_lons)
    headings = np.degrees(np.arctan2(dlat * config.M_PER_DEG_LAT, dlon * config.M_PER_DEG_LON))

    kf = StandardKalmanFilter(process_noise=config.KF_PROCESS_NOISE, measurement_noise=config.KF_MEASUREMENT_NOISE)
    ekf = ExtendedKalmanFilter(process_noise=config.EKF_PROCESS_NOISE, measurement_noise=config.EKF_MEASUREMENT_NOISE)
    pf = ParticleFilter(num_particles=300, process_noise_m=config.PF_PROCESS_NOISE_M,
                         measurement_noise_m=config.PF_MEASUREMENT_NOISE_M,
                         m_per_deg_lat=config.M_PER_DEG_LAT, m_per_deg_lon=config.M_PER_DEG_LON)

    for filt in (kf, ekf):
        filt.X[0, 0] = true_lons[0]
        filt.X[1, 0] = true_lats[0]
    pf.particles[:, 0] = true_lons[0] + np.random.normal(0, 0.0002, 300)
    pf.particles[:, 1] = true_lats[0] + np.random.normal(0, 0.0002, 300)

    def to_m(dx_deg, dy_deg):
        return np.sqrt((dx_deg * config.M_PER_DEG_LON) ** 2 + (dy_deg * config.M_PER_DEG_LAT) ** 2)

    hist_kf, hist_ekf, hist_pf = [], [], []
    errors_kf, errors_ekf, errors_pf = [], [], []
    last_grid = None

    for frame in range(total_frames):
        meas_lat, meas_lon = measured_lats[frame], measured_lons[frame]
        true_lat, true_lon = true_lats[frame], true_lons[frame]

        kf.predict();  kf_pos = kf.update((meas_lon, meas_lat))
        ekf.predict(); ekf_pos = ekf.update((meas_lon, meas_lat))
        pf.predict();  pf_pos = pf.update((meas_lon, meas_lat))

        hist_kf.append(kf_pos); hist_ekf.append(ekf_pos); hist_pf.append(pf_pos)
        errors_kf.append(to_m(kf_pos[0] - true_lon, kf_pos[1] - true_lat))
        errors_ekf.append(to_m(ekf_pos[0] - true_lon, ekf_pos[1] - true_lat))
        errors_pf.append(to_m(pf_pos[0] - true_lon, pf_pos[1] - true_lat))

        if frame == total_frames - 1:
            last_grid = compute_adaptive_confidence_grid(pf, config)

    return dict(
        true_lats=true_lats, true_lons=true_lons,
        measured_lats=measured_lats, measured_lons=measured_lons,
        headings=headings, total_frames=total_frames,
        hist_kf=np.array(hist_kf), hist_ekf=np.array(hist_ekf), hist_pf=np.array(hist_pf),
        errors_kf=np.array(errors_kf), errors_ekf=np.array(errors_ekf), errors_pf=np.array(errors_pf),
        last_grid=last_grid, ekf_final=hist_ekf[-1],
    )


# =====================================================================
# 5. STATIC, PRESENTATION-READY FIGURE EXPORTER
# =====================================================================

def export_static_figures(out_prefix="/mnt/user-data/outputs/fig", config=IISERMohaliConfig, dpi=200):
    d = _run_filters_full(config)
    paths = {}

    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor(PALETTE["bg"]); ax.set_facecolor(PALETTE["bg"])
    ax.set_title("Final Campus Tracking Map — KF vs EKF vs PF", color=PALETTE["title"],
                 fontweight="bold", fontsize=15, pad=12)

    for road_coords in IISERMohaliCampus.ROAD_NETWORK.values():
        road_coords = np.array(road_coords)
        ax.plot(road_coords[:, 1], road_coords[:, 0], color=PALETTE["road"], linewidth=6, alpha=0.45, zorder=1)

    for name, info in IISERMohaliCampus.BUILDINGS.items():
        ax.plot(info['lon'], info['lat'], marker='s', color=info['color'], markersize=7,
                markeredgecolor='white', markeredgewidth=0.7, zorder=3)
        ax.text(info['lon'] + 0.0002, info['lat'] - 0.0002, name, color='#e0e0e0', fontsize=5.5, zorder=4)

    ax.scatter(d["measured_lons"], d["measured_lats"], s=4, color=PALETTE["measured"], alpha=0.35,
               zorder=2, label="Measured GPS (noisy)")
    ax.plot(d["true_lons"], d["true_lats"], color="#ffffff", linestyle="--", linewidth=1.0, alpha=0.4,
            label="True Road Route", zorder=2)

    for key, hist, label in (("kf", d["hist_kf"], "KF Track"),
                              ("ekf", d["hist_ekf"], "EKF Track"),
                              ("pf", d["hist_pf"], "PF Track")):
        ax.plot(hist[:, 0], hist[:, 1], color=PALETTE[key], linestyle=LINESTYLES[key],
                linewidth=LINEWIDTHS[key], label=label, zorder=5, alpha=0.95,
                path_effects=OUTLINE, solid_capstyle="round")
        ax.plot(hist[::MARKER_EVERY, 0], hist[::MARKER_EVERY, 1], "o", color=PALETTE[key],
                markersize=4.5, markeredgecolor="black", markeredgewidth=0.4, zorder=6)

    ax.plot(d["true_lons"][-1], d["true_lats"][-1], marker=car_marker(d["headings"][-1]),
            color=PALETTE["true"], markersize=16, markeredgecolor=PALETTE["true_edge"],
            markeredgewidth=1.6, zorder=7, label="Car (final position)")

    ax.set_xlim(config.CAMPUS_LON_MIN, config.CAMPUS_LON_MAX)
    ax.set_ylim(config.CAMPUS_LAT_MIN, config.CAMPUS_LAT_MAX)
    ax.set_xlabel("Longitude", color="#aaa"); ax.set_ylabel("Latitude", color="#aaa")
    ax.tick_params(colors="#888", labelsize=8)
    ax.legend(loc="upper left", fontsize=8.5, facecolor="#141824", edgecolor=PALETTE["panel_edge"],
                     labelcolor="#eee", framealpha=0.92)
    ax.grid(True, alpha=0.08, color="white")
    fig.tight_layout()
    p = f"{out_prefix}_map.png"; fig.savefig(p, dpi=dpi, facecolor=fig.get_facecolor()); plt.close(fig)
    paths["map"] = p

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(PALETTE["bg"]); ax.set_facecolor(PALETTE["bg"])
    ax.set_title("Tracking Error Over Time (meters)", color="#ffff88", fontweight="bold", fontsize=14, pad=10)
    frames_arr = np.arange(d["total_frames"])
    for key, err, label in (("kf", d["errors_kf"], "KF error"),
                             ("ekf", d["errors_ekf"], "EKF error"),
                             ("pf", d["errors_pf"], "PF error")):
        ax.plot(frames_arr, err, color=PALETTE[key], linestyle=LINESTYLES[key], linewidth=2.4,
                label=f"{label}  (mean {err.mean():.1f} m)", path_effects=OUTLINE)
    for lap in range(1, config.NUM_LAPS):
        ax.axvline(lap * d["total_frames"] / config.NUM_LAPS, color="#555", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xlabel("Frame", color="#aaa"); ax.set_ylabel("Error (m)", color="#aaa")
    ax.tick_params(colors="#888", labelsize=8)
    ax.legend(loc="upper right", fontsize=9, facecolor="#141824", edgecolor=PALETTE["panel_edge"], labelcolor="#eee")
    ax.grid(True, alpha=0.15, color="white")
    fig.tight_layout()
    p = f"{out_prefix}_error.png"; fig.savefig(p, dpi=dpi, facecolor=fig.get_facecolor()); plt.close(fig)
    paths["error"] = p

    XX, YY, ZZ, extent_m, resolution, particle_offsets_m = d["last_grid"]
    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax3 = fig.add_subplot(111, projection="3d")
    ax3.set_facecolor(PALETTE["bg"])
    surf = ax3.plot_surface(XX, YY, ZZ, cmap="plasma", edgecolor="none", alpha=0.92, antialiased=True)
    ax3.scatter(particle_offsets_m[:, 0], particle_offsets_m[:, 1], np.zeros(len(particle_offsets_m)),
                color=PALETTE["kf"], s=6, alpha=0.5, depthshade=False, label="PF particles")
    ax3.scatter([0], [0], [1.0], color="#ffcc00", s=90, marker="^", depthshade=False, label="Car (belief peak)")
    ax3.set_xlim(-extent_m, extent_m); ax3.set_ylim(-extent_m, extent_m); ax3.set_zlim(0, 1.05)
    ax3.set_title(f"Adaptive 3D Confidence Grid — extent ±{extent_m:.1f} m, resolution {resolution}×{resolution}",
                  color=PALETTE["subtitle"], fontweight="bold", fontsize=12, pad=14)
    ax3.set_xlabel("East offset (m)", color="#aaa"); ax3.set_ylabel("North offset (m)", color="#aaa")
    ax3.set_zlabel("Confidence", color="#aaa")
    ax3.tick_params(colors="#888", labelsize=7)
    ax3.view_init(elev=32, azim=45)
    fig.colorbar(surf, ax=ax3, shrink=0.6, pad=0.08, label="Confidence (normalized)")
    fig.tight_layout()
    p = f"{out_prefix}_grid3d.png"; fig.savefig(p, dpi=dpi, facecolor=fig.get_facecolor()); plt.close(fig)
    paths["grid3d"] = p

    print("Saved static figures:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    return paths


# =====================================================================
# 6. LIVE ANIMATION (unchanged core logic, new high-contrast palette)
# =====================================================================

def create_iiser_car_simulation(config=IISERMohaliConfig, save_path=None):
    true_lats, true_lons = build_patrol_route(num_laps=config.NUM_LAPS, steps_per_segment=15)
    total_frames = len(true_lats)

    np.random.seed(7)
    measured_lats = true_lats + np.random.normal(0, config.GPS_ERROR_DEG, total_frames)
    measured_lons = true_lons + np.random.normal(0, config.GPS_ERROR_DEG, total_frames)

    dlat = np.gradient(true_lats)
    dlon = np.gradient(true_lons)
    headings = np.degrees(np.arctan2(dlat * config.M_PER_DEG_LAT, dlon * config.M_PER_DEG_LON))

    kf = StandardKalmanFilter(process_noise=config.KF_PROCESS_NOISE, measurement_noise=config.KF_MEASUREMENT_NOISE)
    ekf = ExtendedKalmanFilter(process_noise=config.EKF_PROCESS_NOISE, measurement_noise=config.EKF_MEASUREMENT_NOISE)
    pf = ParticleFilter(num_particles=300, process_noise_m=config.PF_PROCESS_NOISE_M,
                        measurement_noise_m=config.PF_MEASUREMENT_NOISE_M,
                        m_per_deg_lat=config.M_PER_DEG_LAT, m_per_deg_lon=config.M_PER_DEG_LON)

    for filt in (kf, ekf):
        filt.X[0, 0] = true_lons[0]
        filt.X[1, 0] = true_lats[0]
    pf.particles[:, 0] = true_lons[0] + np.random.normal(0, 0.0002, 300)
    pf.particles[:, 1] = true_lats[0] + np.random.normal(0, 0.0002, 300)

    # ---------------- Figure layout ----------------
    fig = plt.figure(figsize=config.FIGURE_SIZE)
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle('IISER Mohali Campus - Car Patrol Tracking Simulation', fontsize=16, fontweight='bold', color=PALETTE["title"])

    ax_map = plt.subplot2grid((2, 3), (0, 0), rowspan=2)
    ax_error = plt.subplot2grid((2, 3), (0, 1))
    ax_info = plt.subplot2grid((2, 3), (0, 2))
    ax_grid3d = plt.subplot2grid((2, 3), (1, 1), colspan=2, projection='3d')

    ax_map.set_facecolor(PALETTE["bg"])
    ax_map.set_title('Live Campus Tracking Map', color=PALETTE["title"], fontweight='bold', fontsize=12)

    for road_coords in IISERMohaliCampus.ROAD_NETWORK.values():
        road_coords = np.array(road_coords)
        ax_map.plot(road_coords[:, 1], road_coords[:, 0], color=PALETTE["road"], linewidth=6, alpha=0.45, zorder=1)

    for name, info in IISERMohaliCampus.BUILDINGS.items():
        ax_map.plot(info['lon'], info['lat'], marker='s', color=info['color'], markersize=8,
                    markeredgecolor='white', markeredgewidth=0.8, zorder=3)
        ax_map.text(info['lon'] + 0.0002, info['lat'] - 0.0002, name, color='#e0e0e0', fontsize=6, zorder=4)

    meas_scatter = ax_map.scatter([], [], s=4, color=PALETTE["measured"], alpha=0.35, zorder=2, label='Measured GPS (noisy)')
    ax_map.plot(true_lons, true_lats, color='#ffffff', linestyle='--', linewidth=1.0, alpha=0.35,
                label='True Road Route', zorder=2)

    line_kf, = ax_map.plot([], [], color=PALETTE["kf"], linewidth=LINEWIDTHS["kf"], linestyle=LINESTYLES["kf"],
                            label='KF Track', zorder=6, alpha=0.95, path_effects=OUTLINE, solid_capstyle='round')
    line_ekf, = ax_map.plot([], [], color=PALETTE["ekf"], linewidth=LINEWIDTHS["ekf"], linestyle=LINESTYLES["ekf"],
                             label='EKF Track', zorder=5, alpha=0.95, path_effects=OUTLINE)
    line_pf, = ax_map.plot([], [], color=PALETTE["pf"], linewidth=LINEWIDTHS["pf"], linestyle=LINESTYLES["pf"],
                            label='PF Track', zorder=4, alpha=0.95, path_effects=OUTLINE)

    car_true, = ax_map.plot([], [], marker=car_marker(0), color=PALETTE["true"], markersize=16,
                             markeredgecolor=PALETTE["true_edge"], markeredgewidth=1.5, zorder=7, label='Car (true GPS track)')
    car_ekf, = ax_map.plot([], [], marker='o', color=PALETTE["ekf"], markersize=9,
                            markeredgecolor='white', markeredgewidth=1.5, zorder=6, label='EKF estimate')

    ax_map.set_xlim(config.CAMPUS_LON_MIN, config.CAMPUS_LON_MAX)
    ax_map.set_ylim(config.CAMPUS_LAT_MIN, config.CAMPUS_LAT_MAX)
    ax_map.set_xlabel('Longitude', fontsize=9, color='#888')
    ax_map.set_ylabel('Latitude', fontsize=9, color='#888')
    ax_map.tick_params(colors='#666', labelsize=8)
    ax_map.legend(loc='upper left', fontsize=8, facecolor='#141824', edgecolor=PALETTE["panel_edge"], labelcolor='#ccc')
    ax_map.grid(True, alpha=0.1, color='white')

    ax_error.set_facecolor(PALETTE["bg"])
    ax_error.set_title('Tracking Error (meters)', color='#ffff88', fontweight='bold', fontsize=10)
    line_err_kf, = ax_error.plot([], [], color=PALETTE["kf"], linewidth=2.2, linestyle=LINESTYLES["kf"], label='KF')
    line_err_ekf, = ax_error.plot([], [], color=PALETTE["ekf"], linewidth=2.2, linestyle=LINESTYLES["ekf"], label='EKF')
    line_err_pf, = ax_error.plot([], [], color=PALETTE["pf"], linewidth=2.2, linestyle=LINESTYLES["pf"], label='PF')
    ax_error.set_xlabel('Frame', fontsize=9, color='#888')
    ax_error.set_ylabel('Error (m)', fontsize=9, color='#888')
    ax_error.tick_params(colors='#666', labelsize=8)
    ax_error.legend(loc='upper right', fontsize=8, facecolor='#141824', edgecolor=PALETTE["panel_edge"], labelcolor='#ccc')
    ax_error.grid(True, alpha=0.2, color='white')
    ax_error.set_xlim(0, total_frames)
    ax_error.set_ylim(0, 40)

    ax_info.set_facecolor(PALETTE["bg"])
    ax_info.axis('off')
    info_text = ax_info.text(0.05, 0.95, '', transform=ax_info.transAxes, fontsize=11, color=PALETTE["title"],
                              verticalalignment='top', fontfamily='monospace',
                              bbox=dict(boxstyle='round', facecolor=PALETTE["bg"], edgecolor=PALETTE["title"], alpha=0.8))

    ax_grid3d.set_facecolor(PALETTE["bg"])
    ax_grid3d.set_title('Adaptive 3D Confidence Grid (around car)', color=PALETTE["subtitle"], fontweight='bold', fontsize=10)
    ax_grid3d.set_xlabel('East offset (m)', fontsize=8, color='#888')
    ax_grid3d.set_ylabel('North offset (m)', fontsize=8, color='#888')
    ax_grid3d.set_zlabel('Confidence', fontsize=8, color='#888')
    ax_grid3d.tick_params(colors='#666', labelsize=7)
    ax_grid3d.set_zlim(0, 1.05)

    hist_kf, hist_ekf, hist_pf = [], [], []
    errors_kf, errors_ekf, errors_pf = [], [], []

    def to_m(dx_deg, dy_deg):
        return np.sqrt((dx_deg * config.M_PER_DEG_LON) ** 2 + (dy_deg * config.M_PER_DEG_LAT) ** 2)

    def init():
        line_kf.set_data([], [])
        line_ekf.set_data([], [])
        line_pf.set_data([], [])
        return line_kf, line_ekf, line_pf

    def animate(frame):
        true_lat, true_lon = true_lats[frame], true_lons[frame]
        meas_lat, meas_lon = measured_lats[frame], measured_lons[frame]

        kf.predict()
        kf_pos = kf.update((meas_lon, meas_lat))
        ekf.predict()
        ekf_pos = ekf.update((meas_lon, meas_lat))
        pf.predict()
        pf_pos = pf.update((meas_lon, meas_lat))

        hist_kf.append(kf_pos); hist_ekf.append(ekf_pos); hist_pf.append(pf_pos)

        err_kf = to_m(kf_pos[0] - true_lon, kf_pos[1] - true_lat)
        err_ekf = to_m(ekf_pos[0] - true_lon, ekf_pos[1] - true_lat)
        err_pf = to_m(pf_pos[0] - true_lon, pf_pos[1] - true_lat)
        errors_kf.append(err_kf); errors_ekf.append(err_ekf); errors_pf.append(err_pf)

        kf_arr, ekf_arr, pf_arr = np.array(hist_kf), np.array(hist_ekf), np.array(hist_pf)
        line_kf.set_data(kf_arr[:, 0], kf_arr[:, 1])
        line_ekf.set_data(ekf_arr[:, 0], ekf_arr[:, 1])
        line_pf.set_data(pf_arr[:, 0], pf_arr[:, 1])
        meas_scatter.set_offsets(np.column_stack([measured_lons[:frame + 1], measured_lats[:frame + 1]]))

        car_true.set_data([true_lon], [true_lat])
        car_true.set_marker(car_marker(headings[frame]))
        car_ekf.set_data([ekf_pos[0]], [ekf_pos[1]])

        frames_arr = np.arange(len(errors_kf))
        line_err_kf.set_data(frames_arr, errors_kf)
        line_err_ekf.set_data(frames_arr, errors_ekf)
        line_err_pf.set_data(frames_arr, errors_pf)
        max_err = max(max(errors_kf), max(errors_ekf), max(errors_pf), 1.0)
        ax_error.set_ylim(0, max_err * 1.2)

        nearest, best_d = None, 1e18
        for name, info in IISERMohaliCampus.BUILDINGS.items():
            d = to_m(ekf_pos[0] - info['lon'], ekf_pos[1] - info['lat'])
            if d < best_d:
                nearest, best_d = name, d

        # --- Adaptive 3D confidence grid, rebuilt from the PF's real belief ---
        XX, YY, ZZ, extent_m, resolution, particle_offsets_m = compute_adaptive_confidence_grid(pf, config)
        ax_grid3d.cla()
        ax_grid3d.set_facecolor(PALETTE["bg"])
        ax_grid3d.plot_surface(XX, YY, ZZ, cmap='plasma', edgecolor='none', alpha=0.9, antialiased=True)
        ax_grid3d.scatter(particle_offsets_m[:, 0], particle_offsets_m[:, 1],
                           np.zeros(len(particle_offsets_m)), color=PALETTE["kf"], s=3, alpha=0.4, depthshade=False)
        ax_grid3d.scatter([0], [0], [1.0], color='#ffcc00', s=60, marker='^', depthshade=False)  # car at the peak
        ax_grid3d.set_xlim(-extent_m, extent_m)
        ax_grid3d.set_ylim(-extent_m, extent_m)
        ax_grid3d.set_zlim(0, 1.05)
        ax_grid3d.set_title(f'Adaptive 3D Confidence Grid  (extent \u00b1{extent_m:.1f}m, res {resolution}x{resolution})',
                             color=PALETTE["subtitle"], fontweight='bold', fontsize=10)
        ax_grid3d.set_xlabel('East offset (m)', fontsize=8, color='#888')
        ax_grid3d.set_ylabel('North offset (m)', fontsize=8, color='#888')
        ax_grid3d.set_zlabel('Confidence', fontsize=8, color='#888')
        ax_grid3d.tick_params(colors='#666', labelsize=7)
        ax_grid3d.view_init(elev=32, azim=(frame * 0.6) % 360)  # slow cinematic rotation

        lap = frame // (total_frames // config.NUM_LAPS) + 1
        mean_kf = np.mean(errors_kf[-10:])
        mean_ekf = np.mean(errors_ekf[-10:])
        mean_pf = np.mean(errors_pf[-10:])
        best = min([('KF', mean_kf), ('EKF', mean_ekf), ('PF', mean_pf)], key=lambda t: t[1])[0]

        info_text.set_text(
            f"CAR PATROL - Lap {lap}/{config.NUM_LAPS}\n\n"
            f"Position:\n"
            f"  Lat: {true_lat:.6f}\n"
            f"  Lon: {true_lon:.6f}\n\n"
            f"Nearest building:\n  {nearest} ({best_d:.0f} m)\n\n"
            f"Tracking error (10-frame avg):\n"
            f"  KF:  {mean_kf:5.1f} m\n"
            f"  EKF: {mean_ekf:5.1f} m\n"
            f"  PF:  {mean_pf:5.1f} m\n\n"
            f"Best filter now: {best}\n\n"
            f"Frame {frame + 1}/{total_frames}"
        )

        # PROGRESS FIX: a full render is ~4-5 minutes and previously gave
        # zero feedback, which looks identical to "hung/broken". Print
        # progress every 30 frames when writing to a file.
        if save_path and (frame % 30 == 0 or frame == total_frames - 1):
            print(f"  rendering frame {frame + 1}/{total_frames}...")

        return (line_kf, line_ekf, line_pf, car_true, car_ekf,
                line_err_kf, line_err_ekf, line_err_pf, info_text)

    anim = FuncAnimation(fig, animate, frames=total_frames, init_func=init,
                          blit=False, interval=config.ANIMATION_INTERVAL, repeat=True)

    plt.tight_layout()

    if save_path:
        # FFMPEG FIX: the old code always used writer='ffmpeg' for .mp4
        # paths with no fallback. If ffmpeg isn't installed on the
        # user's machine this raises RuntimeError and the whole thing
        # crashes with a cryptic message. Fall back to a GIF (pillow
        # writer ships with matplotlib, no external binary needed) so
        # the user always gets an output file either way.
        if save_path.endswith('.mp4'):
            import shutil
            try:
                if shutil.which('ffmpeg') is None:
                    raise RuntimeError("ffmpeg not found on PATH")
                anim.save(save_path, writer='ffmpeg', fps=15, dpi=110,
                          extra_args=['-movflags', '+faststart', '-pix_fmt', 'yuv420p'])
            except Exception as e:
                fallback = save_path.rsplit('.', 1)[0] + '.gif'
                print(f"Could not save MP4 ({e}); ffmpeg is required for .mp4 output.")
                print(f"Falling back to GIF: {fallback}")
                anim.save(fallback, writer='pillow', fps=15, dpi=110)
                save_path = fallback
        else:
            anim.save(save_path, writer='pillow', fps=15, dpi=110)
        print(f"Saved animation to {save_path}")
    else:
        plt.show()

    return anim


if __name__ == "__main__":
    import sys

    if "--static" in sys.argv:
        # Fastest path: just the 3 final PNGs, no animation render.
        export_static_figures()
    elif "--show" in sys.argv:
        # Force an interactive window even if auto-detection guessed
        # headless. Will raise a clear matplotlib error if there is
        # truly no display, instead of silently doing nothing.
        create_iiser_car_simulation(save_path=None)
    else:
        # ENTRY-POINT FIX: previously this branch always called
        # create_iiser_car_simulation(save_path=None), which -- on the
        # unconditionally-forced Agg backend -- just calls plt.show()
        # and does nothing visible: no window, no file, no error. Now:
        # if there's a real display, show it live; otherwise always
        # write an actual output file so something is produced.
        if _HAS_DISPLAY:
            create_iiser_car_simulation(save_path=None)
        else:
            out = "/mnt/user-data/outputs/campus_tracking_animation.mp4"
            print("No display detected -- saving the animation to a file instead "
                  "of opening a window (pass --show to force a live window, or "
                  "--static for just the 3 final PNGs, which is much faster: "
                  "~5s vs ~4-5 min for the full animation).")
            create_iiser_car_simulation(save_path=out)

# ---- test harness for live animation function (not in original file) ----