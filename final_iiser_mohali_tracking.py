#!/usr/bin/env python3
"""
================================================================================
IISER MOHALI CAMPUS CAR TRACKING SIMULATION
Kalman Filter / Extended Kalman Filter / Particle Filter comparison
================================================================================
A car drives continuous laps around the IISER Mohali main campus loop road.
Its GPS position is noisy; three filters (KF, EKF, PF) each try to recover
the true smooth path in real time. The animation renders the campus map,
the live filter tracks, a rotating car icon, a tracking-error chart, and a
live info panel (nearest building, current error, etc).

FIXES APPLIED vs the original script:
  1. ParticleFilter._measurement_model mixed units: distances were computed
     in raw lat/lon degrees while measurement_noise was tuned assuming meters.
     This made the Gaussian likelihood underflow to exactly 0.0 for every
     particle within a few frames -> ZeroDivisionError in np.average().
     FIX: convert both distance and noise sigma to meters before computing
     the likelihood.
  2. ParticleFilter._resample() could walk its cumulative-weight index `j`
     out of bounds when floating point error left cum_weights[-1] slightly
     under 1.0. FIX: clip j to valid range.
  3. Scenario is now a continuous multi-lap patrol around the real road
     network (not a single one-way walk), so it behaves like an actual car
     driving around campus rather than a single point-to-point animation.
  4. Runs headless (Agg backend) and saves an MP4, since this environment
     has no display; a live plt.show() window still works fine on a normal
     desktop with a GUI backend if you switch it back.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# 1. CAMPUS CONFIGURATION
# =====================================================================

class IISERMohaliConfig:
    CAMPUS_LAT_MIN = 30.6580
    CAMPUS_LAT_MAX = 30.6700
    CAMPUS_LON_MIN = 76.7200
    CAMPUS_LON_MAX = 76.7380

    # Meters-per-degree scale factors at this latitude (used consistently everywhere)
    M_PER_DEG_LAT = 111000.0
    M_PER_DEG_LON = 96000.0

    FIGURE_SIZE = (18, 11)
    ANIMATION_INTERVAL = 60

    KF_PROCESS_NOISE = 5e-6
    KF_MEASUREMENT_NOISE = 1e-5
    EKF_PROCESS_NOISE = 8e-6
    EKF_MEASUREMENT_NOISE = 1.2e-5
    PF_PROCESS_NOISE = 1e-5
    PF_MEASUREMENT_NOISE_M = 15.0   # in METERS now (was mismatched with degrees before)

    GPS_ERROR_DEG = 0.00012        # ~ +/-12m GPS jitter (more realistic than the original 0.0003 ~30m)
    NUM_LAPS = 2


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

    # The car drives laps around this loop road (closes back on itself)
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
    """Standard Linear Kalman Filter (constant velocity model)"""

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
    """EKF with a turning-rate + acceleration motion model, good for road curves"""

    def __init__(self, dt=1.0, process_noise=8e-6, measurement_noise=1.2e-5):
        self.dt = dt
        self.X = np.zeros((6, 1))  # [x, y, vx, vy, omega, accel]
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
    """
    Particle Filter for non-linear/non-Gaussian tracking.

    FIX: the likelihood must be computed in a single consistent unit
    (meters). The original version compared raw lat/lon-degree distances
    against a measurement_noise tuned for meters, which made every
    particle's weight underflow to 0.0 -> ZeroDivisionError.
    """

    def __init__(self, num_particles=300, dt=1.0, process_noise=1e-5,
                 measurement_noise_m=15.0,
                 m_per_deg_lat=111000.0, m_per_deg_lon=96000.0):
        self.num_particles = num_particles
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise_m = measurement_noise_m
        self.m_per_deg_lat = m_per_deg_lat
        self.m_per_deg_lon = m_per_deg_lon

        self.particles = np.zeros((num_particles, 4))
        self.weights = np.ones(num_particles) / num_particles
        self.X = np.zeros((4, 1))

    def _motion_model(self, particles):
        n = self.num_particles
        particles_new = particles.copy()
        particles_new[:, 0] += particles[:, 2] * self.dt + np.random.normal(0, self.process_noise, n)
        particles_new[:, 1] += particles[:, 3] * self.dt + np.random.normal(0, self.process_noise, n)
        particles_new[:, 2] *= 0.95 + np.random.normal(0, self.process_noise, n)
        particles_new[:, 3] *= 0.95 + np.random.normal(0, self.process_noise, n)
        return particles_new

    def _measurement_model(self, particles, measurement):
        z = np.array(measurement)
        # convert lon/lat differences to METERS before comparing to measurement_noise_m
        dx_m = (particles[:, 0] - z[0]) * self.m_per_deg_lon
        dy_m = (particles[:, 1] - z[1]) * self.m_per_deg_lat
        distances_m = np.sqrt(dx_m ** 2 + dy_m ** 2)
        likelihood = np.exp(-distances_m ** 2 / (2 * self.measurement_noise_m ** 2))
        total = np.sum(likelihood)
        if total < 1e-300:
            # every particle is implausibly far from the measurement (e.g. after
            # a bad init) -> fall back to uniform weights instead of crashing
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
        """Systematic resampling. FIX: clip index j so floating-point rounding
        in cum_weights can't push it out of bounds (was an IndexError risk)."""
        indices = np.argsort(self.weights)[::-1]
        cum_weights = np.cumsum(self.weights[indices])
        cum_weights[-1] = 1.0  # guard against floating point drift
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
    """Concatenate the main loop road `num_laps` times so the car keeps
    driving around campus continuously, like a real patrol/shuttle car."""
    waypoints = np.array(IISERMohaliCampus.MAIN_LOOP)

    lats, lons = [], []
    for lap in range(num_laps):
        for i in range(len(waypoints) - 1):
            lats.extend(np.linspace(waypoints[i][0], waypoints[i + 1][0], steps_per_segment, endpoint=False))
            lons.extend(np.linspace(waypoints[i][1], waypoints[i + 1][1], steps_per_segment, endpoint=False))
    return np.array(lats), np.array(lons)


def car_marker(heading_deg):
    """A simple triangular 'car' marker rotated to face its direction of travel."""
    return MarkerStyle('^', transform=Affine2D().rotate_deg(heading_deg - 90))


# =====================================================================
# 4. BUILD & RUN THE SIMULATION
# =====================================================================

def create_iiser_car_simulation(config=IISERMohaliConfig, save_path=None):
    true_lats, true_lons = build_patrol_route(num_laps=config.NUM_LAPS, steps_per_segment=15)
    total_frames = len(true_lats)

    np.random.seed(7)
    measured_lats = true_lats + np.random.normal(0, config.GPS_ERROR_DEG, total_frames)
    measured_lons = true_lons + np.random.normal(0, config.GPS_ERROR_DEG, total_frames)

    # headings (deg) for rotating the car icon, based on the true path direction
    dlat = np.gradient(true_lats)
    dlon = np.gradient(true_lons)
    headings = np.degrees(np.arctan2(dlat * config.M_PER_DEG_LAT, dlon * config.M_PER_DEG_LON))

    kf = StandardKalmanFilter(process_noise=config.KF_PROCESS_NOISE, measurement_noise=config.KF_MEASUREMENT_NOISE)
    ekf = ExtendedKalmanFilter(process_noise=config.EKF_PROCESS_NOISE, measurement_noise=config.EKF_MEASUREMENT_NOISE)
    pf = ParticleFilter(num_particles=300, process_noise=config.PF_PROCESS_NOISE,
                        measurement_noise_m=config.PF_MEASUREMENT_NOISE_M,
                        m_per_deg_lat=config.M_PER_DEG_LAT, m_per_deg_lon=config.M_PER_DEG_LON)

    for filt in (kf, ekf):
        filt.X[0, 0] = true_lons[0]
        filt.X[1, 0] = true_lats[0]
    pf.particles[:, 0] = true_lons[0] + np.random.normal(0, 0.0002, 300)
    pf.particles[:, 1] = true_lats[0] + np.random.normal(0, 0.0002, 300)

    # ---------------- Figure layout ----------------
    fig = plt.figure(figsize=config.FIGURE_SIZE)
    fig.patch.set_facecolor('#0a0e27')
    fig.suptitle('IISER Mohali Campus - Car Patrol Tracking Simulation', fontsize=16, fontweight='bold', color='#00ff88')

    ax_map = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
    ax_error = plt.subplot2grid((2, 2), (0, 1))
    ax_info = plt.subplot2grid((2, 2), (1, 1))

    ax_map.set_facecolor('#0a0e27')
    ax_map.set_title('Live Campus Tracking Map', color='#00ff88', fontweight='bold', fontsize=12)

    for road_coords in IISERMohaliCampus.ROAD_NETWORK.values():
        road_coords = np.array(road_coords)
        ax_map.plot(road_coords[:, 1], road_coords[:, 0], color='#1a4d7a', linewidth=6, alpha=0.5, zorder=1)

    for name, info in IISERMohaliCampus.BUILDINGS.items():
        ax_map.plot(info['lon'], info['lat'], marker='s', color=info['color'], markersize=8,
                    markeredgecolor='white', markeredgewidth=0.8, zorder=3)
        ax_map.text(info['lon'] + 0.0002, info['lat'] - 0.0002, name, color='#e0e0e0', fontsize=6, zorder=4)

    ax_map.plot(true_lons, true_lats, color='#00bfff', linestyle='--', linewidth=1.2, alpha=0.35,
                label='Road Route', zorder=2)

    line_kf, = ax_map.plot([], [], color='#00ff88', linewidth=2, label='KF Track', zorder=4)
    line_ekf, = ax_map.plot([], [], color='#00bfff', linewidth=2, label='EKF Track', zorder=4)
    line_pf, = ax_map.plot([], [], color='#ee82ee', linewidth=2, label='PF Track', zorder=4)

    car_true, = ax_map.plot([], [], marker=car_marker(0), color='#ffffff', markersize=16,
                             markeredgecolor='#ffcc00', markeredgewidth=1.5, zorder=6, label='Car (true GPS track)')
    car_ekf, = ax_map.plot([], [], marker='o', color='#00bfff', markersize=9,
                            markeredgecolor='white', markeredgewidth=1.5, zorder=5, label='EKF estimate')

    ax_map.set_xlim(config.CAMPUS_LON_MIN, config.CAMPUS_LON_MAX)
    ax_map.set_ylim(config.CAMPUS_LAT_MIN, config.CAMPUS_LAT_MAX)
    ax_map.set_xlabel('Longitude', fontsize=9, color='#888')
    ax_map.set_ylabel('Latitude', fontsize=9, color='#888')
    ax_map.tick_params(colors='#666', labelsize=8)
    ax_map.legend(loc='upper left', fontsize=8, facecolor='#141824', edgecolor='#333', labelcolor='#ccc')
    ax_map.grid(True, alpha=0.1, color='white')

    ax_error.set_facecolor('#0a0e27')
    ax_error.set_title('Tracking Error (meters)', color='#ffff88', fontweight='bold', fontsize=10)
    line_err_kf, = ax_error.plot([], [], color='#00ff88', linewidth=2, label='KF')
    line_err_ekf, = ax_error.plot([], [], color='#00bfff', linewidth=2, label='EKF')
    line_err_pf, = ax_error.plot([], [], color='#ee82ee', linewidth=2, label='PF')
    ax_error.set_xlabel('Frame', fontsize=9, color='#888')
    ax_error.set_ylabel('Error (m)', fontsize=9, color='#888')
    ax_error.tick_params(colors='#666', labelsize=8)
    ax_error.legend(loc='upper right', fontsize=8, facecolor='#141824', edgecolor='#333', labelcolor='#ccc')
    ax_error.grid(True, alpha=0.2, color='white')
    ax_error.set_xlim(0, total_frames)
    ax_error.set_ylim(0, 40)

    ax_info.set_facecolor('#0a0e27')
    ax_info.axis('off')
    info_text = ax_info.text(0.05, 0.95, '', transform=ax_info.transAxes, fontsize=11, color='#00ff88',
                              verticalalignment='top', fontfamily='monospace',
                              bbox=dict(boxstyle='round', facecolor='#0a0e27', edgecolor='#00ff88', alpha=0.8))

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

        return (line_kf, line_ekf, line_pf, car_true, car_ekf,
                line_err_kf, line_err_ekf, line_err_pf, info_text)

    anim = FuncAnimation(fig, animate, frames=total_frames, init_func=init,
                          blit=False, interval=config.ANIMATION_INTERVAL, repeat=True)

    plt.tight_layout()

    if save_path:
        # Optional: pass save_path="something.mp4" to also export a video file.
        if save_path.endswith('.mp4'):
            anim.save(save_path, writer='ffmpeg', fps=15, dpi=110,
                      extra_args=['-movflags', '+faststart', '-pix_fmt', 'yuv420p'])
        else:
            anim.save(save_path, writer='pillow', fps=15, dpi=110)
        print(f"Saved animation to {save_path}")
    else:
        # Live interactive window (requires a real display / GUI backend)
        plt.show()

    return anim


if __name__ == "__main__":
    create_iiser_car_simulation(save_path=None)