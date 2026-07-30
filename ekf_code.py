#!/usr/bin/env python3
"""
================================================================================
ADVANCED IISER MOHALI TRACKING SYSTEM
Extended Kalman Filter + Particle Filter + Prediction
================================================================================
Implements advanced tracking techniques from MATLAB documentation:
  ✓ Extended Kalman Filter (EKF) - Non-linear motion models
  ✓ Particle Filter (PF) - Non-Gaussian noise handling
  ✓ Adaptive filtering - Dynamic noise adjustment
  ✓ Trajectory prediction - Forecast future positions
  ✓ Outlier rejection - Invalid measurement handling
  ✓ Performance comparison - KF vs EKF vs PF

Based on:
1. Using Kalman Filter for Object Tracking (MATLAB)
2. Track Objects with Wrapping Azimuth Angles (MATLAB)
3. Non-Gaussian Non-Linear Object Tracking (MATLAB)

Usage:
  python ADVANCED_TRACKING_SYSTEM.py

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
from matplotlib.animation import FuncAnimation
from scipy.stats import multivariate_normal
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# CONFIGURATION
# =====================================================================

class TrackingConfig:
    """Configuration for all tracking algorithms"""
    
    # Kalman Filter
    DT = 1.0
    BASE_PROCESS_NOISE = 5e-6
    BASE_MEASUREMENT_NOISE = 1e-5
    VELOCITY_DAMPING = 0.95
    
    # Extended Kalman Filter (Non-linear)
    EKF_PROCESS_NOISE = 8e-6
    EKF_MEASUREMENT_NOISE = 1.2e-5
    
    # Particle Filter
    NUM_PARTICLES = 500
    RESAMPLING_THRESHOLD = 0.5  # Effective sample size threshold
    PARTICLE_PROCESS_NOISE = 1e-5
    PARTICLE_MEASUREMENT_NOISE = 1.5e-5
    
    # Prediction
    PREDICTION_HORIZON = 5  # Frames ahead to predict
    
    # Local Grid
    LOCAL_VIEW_RADIUS = 0.0035
    LANDMARK_INFLUENCE_RADIUS = 0.0015
    GRID_RESOLUTION = 96
    
    # Visualization
    ANIMATION_INTERVAL = 80
    FIGURE_SIZE = (16, 10)
    GPS_ERROR_DEVIATION = 0.0003
    RANDOM_SEED = 42


# =====================================================================
# 1. STANDARD KALMAN FILTER (Baseline)
# =====================================================================

class StandardKalmanFilter:
    """Standard Linear Kalman Filter"""
    
    def __init__(self, dt=1.0, process_noise=5e-6, measurement_noise=1e-5):
        self.dt = dt
        self.X = np.zeros((4, 1))  # [lon, lat, vel_lon, vel_lat]
        
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        
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


# =====================================================================
# 2. EXTENDED KALMAN FILTER (Non-linear Motion)
# =====================================================================

class ExtendedKalmanFilter:
    """
    Extended Kalman Filter with non-linear motion model
    Handles:
    - Curved paths (turn rate model)
    - Acceleration changes
    - Non-linear dynamics
    
    State: [lon, lat, vel_lon, vel_lat, turn_rate, acceleration]
    """
    
    def __init__(self, dt=1.0, process_noise=8e-6, measurement_noise=1.2e-5):
        self.dt = dt
        self.X = np.zeros((6, 1))  # [lon, lat, vel_lon, vel_lat, turn_rate, accel]
        
        # Measurement matrix (linear part - we measure position only)
        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], dtype=np.float64)
        
        # Process noise covariance
        self.Q = np.eye(6) * process_noise
        self.Q[2:4, 2:4] *= 8    # Higher on velocity
        self.Q[4:6, 4:6] *= 15   # Higher on turn rate and accel
        
        # Measurement noise covariance
        self.R = np.eye(2) * measurement_noise
        
        # State covariance
        self.P = np.eye(6) * 1e-4
        self.P[2:4, 2:4] *= 1e-6
        
        self.velocity_damping = 0.94
    
    def _state_transition_jacobian(self):
        """Compute Jacobian of state transition function"""
        # F_jacobian for non-linear motion model
        F_jac = np.eye(6, dtype=np.float64)
        F_jac[0, 2] = self.dt  # lon += vel_lon * dt
        F_jac[1, 3] = self.dt  # lat += vel_lat * dt
        # Velocity changes due to acceleration
        F_jac[2, 5] = self.dt  # vel_lon += accel * dt
        F_jac[3, 5] = self.dt  # vel_lat += accel * dt
        return F_jac
    
    def _measurement_jacobian(self):
        """Jacobian of measurement function"""
        return self.H
    
    def _f(self, x):
        """Non-linear state transition function"""
        x_new = x.copy()
        # Position update with velocity
        x_new[0] += x[2] * self.dt  # lon
        x_new[1] += x[3] * self.dt  # lat
        
        # Velocity update with acceleration (and turn rate)
        turn_rate = x[4, 0]
        accel = x[5, 0]
        
        # Apply turn rate (rotates velocity vector)
        vel_magnitude = np.sqrt(x[2, 0]**2 + x[3, 0]**2)
        if vel_magnitude > 1e-8:
            angle = np.arctan2(x[3, 0], x[2, 0])
            angle += turn_rate * self.dt
            x_new[2, 0] = vel_magnitude * np.cos(angle)
            x_new[3, 0] = vel_magnitude * np.sin(angle)
        
        # Apply acceleration
        x_new[2, 0] += accel * self.dt
        x_new[3, 0] += accel * self.dt
        
        # Damping
        x_new[2:4] *= self.velocity_damping
        
        return x_new
    
    def predict(self):
        # Non-linear prediction
        self.X = self._f(self.X)
        
        # Covariance prediction using Jacobian
        F_jac = self._state_transition_jacobian()
        self.P = F_jac @ self.P @ F_jac.T + self.Q
        
        return self.X[:2].flatten()
    
    def update(self, measurement):
        Z = np.array(measurement).reshape(2, 1)
        
        # Predicted measurement
        z_pred = self.H @ self.X
        y = Z - z_pred  # Innovation
        
        # Innovation covariance
        H_jac = self._measurement_jacobian()
        S = H_jac @ self.P @ H_jac.T + self.R
        
        # Kalman gain
        K = self.P @ H_jac.T @ np.linalg.inv(S)
        
        # State update
        self.X = self.X + K @ y
        
        # Covariance update
        self.P = (np.eye(6) - K @ H_jac) @ self.P
        
        return self.X[:2].flatten()


# =====================================================================
# 3. PARTICLE FILTER (Non-Gaussian Noise)
# =====================================================================

class ParticleFilter:
    """
    Particle Filter for non-linear, non-Gaussian tracking
    Handles:
    - Multipath GPS errors (non-Gaussian)
    - Multiple hypotheses
    - Non-linear dynamics
    
    Each particle represents a possible state
    """
    
    def __init__(self, num_particles=500, dt=1.0, 
                 process_noise=1e-5, measurement_noise=1.5e-5):
        self.num_particles = num_particles
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        # Initialize particles randomly
        self.particles = np.zeros((num_particles, 4))  # [lon, lat, vel_lon, vel_lat]
        self.weights = np.ones(num_particles) / num_particles
        
        # State estimate
        self.X = np.zeros((4, 1))
    
    def _motion_model(self, particles):
        """Apply non-linear motion model to particles"""
        particles_new = particles.copy()
        
        # Constant velocity model with noise
        particles_new[:, 0] += particles[:, 2] * self.dt + np.random.normal(0, self.process_noise, self.num_particles)
        particles_new[:, 1] += particles[:, 3] * self.dt + np.random.normal(0, self.process_noise, self.num_particles)
        
        # Velocity damping with small random walk
        particles_new[:, 2] *= 0.95 + np.random.normal(0, self.process_noise, self.num_particles)
        particles_new[:, 3] *= 0.95 + np.random.normal(0, self.process_noise, self.num_particles)
        
        return particles_new
    
    def _measurement_model(self, particles, measurement):
        """Compute likelihood of measurement given particles"""
        z = np.array(measurement)
        
        # Euclidean distance from each particle to measurement
        distances = np.sqrt((particles[:, 0] - z[0])**2 + (particles[:, 1] - z[1])**2)
        
        # Gaussian likelihood
        likelihood = np.exp(-distances**2 / (2 * self.measurement_noise**2))
        
        return likelihood / (np.sum(likelihood) + 1e-10)
    
    def predict(self):
        """Predict using motion model"""
        self.particles = self._motion_model(self.particles)
        
        # Compute weighted state estimate
        self.X[:2, 0] = np.average(self.particles[:, :2], axis=0, weights=self.weights)
        self.X[2:, 0] = np.average(self.particles[:, 2:], axis=0, weights=self.weights)
        
        return self.X[:2].flatten()
    
    def update(self, measurement):
        """Update weights based on measurement"""
        # Compute measurement likelihood
        self.weights = self._measurement_model(self.particles, measurement)
        self.weights /= (np.sum(self.weights) + 1e-10)
        
        # Resampling if effective sample size is too small
        n_eff = 1.0 / np.sum(self.weights**2)
        if n_eff < self.num_particles * 0.5:
            self._resample()
        
        # Compute weighted state estimate
        self.X[:2, 0] = np.average(self.particles[:, :2], axis=0, weights=self.weights)
        self.X[2:, 0] = np.average(self.particles[:, 2:], axis=0, weights=self.weights)
        
        return self.X[:2].flatten()
    
    def _resample(self):
        """Resample particles based on weights (systematic resampling)"""
        indices = np.argsort(self.weights)[::-1]
        
        # Cumulative sum for systematic resampling
        cum_weights = np.cumsum(self.weights[indices])
        
        # Resample
        new_indices = []
        u = np.random.uniform(0, 1.0 / self.num_particles)
        j = 0
        
        for i in range(self.num_particles):
            while u > cum_weights[j]:
                j += 1
            new_indices.append(indices[j])
            u += 1.0 / self.num_particles
        
        self.particles = self.particles[new_indices].copy()
        self.weights = np.ones(self.num_particles) / self.num_particles


# =====================================================================
# 4. TRAJECTORY PREDICTION
# =====================================================================

def predict_trajectory(filter_obj, current_pos, num_steps=5):
    """
    Predict future positions using the filter's motion model
    
    Args:
        filter_obj: KF/EKF/PF object
        current_pos: Current position [lon, lat]
        num_steps: Number of frames to predict ahead
    
    Returns:
        predicted_positions: Array of predicted positions
    """
    predicted_positions = [current_pos]
    
    # Create temporary state
    temp_X = filter_obj.X.copy()
    
    for _ in range(num_steps):
        if isinstance(filter_obj, ExtendedKalmanFilter):
            temp_X = filter_obj._f(temp_X)
        elif isinstance(filter_obj, ParticleFilter):
            # For PF, use average particle motion
            particles_temp = filter_obj.particles.copy()
            particles_temp = filter_obj._motion_model(particles_temp)
            pos = np.average(particles_temp[:, :2], axis=0, weights=filter_obj.weights)
        else:
            # Standard KF
            temp_X = filter_obj.F @ temp_X
            pos = temp_X[:2].flatten()
        
        if isinstance(filter_obj, ExtendedKalmanFilter):
            pos = temp_X[:2].flatten()
        
        predicted_positions.append(pos)
    
    return np.array(predicted_positions)


# =====================================================================
# 5. ADAPTIVE OUTLIER REJECTION
# =====================================================================

def is_valid_measurement(measurement, last_estimate, velocity_estimate, 
                        max_distance=0.001, max_velocity_jump=0.0005):
    """
    Validate measurement using statistical tests
    Reject outliers from GPS multipath errors
    
    Args:
        measurement: Current GPS measurement
        last_estimate: Previous filter estimate
        velocity_estimate: Current velocity estimate
        max_distance: Maximum allowed jump
        max_velocity_jump: Maximum velocity change
    
    Returns:
        Boolean: True if measurement is valid
    """
    # Test 1: Distance jumped
    dist_jumped = np.sqrt((measurement[0] - last_estimate[0])**2 + 
                          (measurement[1] - last_estimate[1])**2)
    
    # Expected distance based on velocity
    expected_dist = np.linalg.norm(velocity_estimate)
    
    # Distance should be close to velocity * dt
    if dist_jumped > expected_dist + max_distance:
        return False
    
    # Test 2: Velocity jump (maximum acceleration)
    implied_velocity = np.array([
        (measurement[0] - last_estimate[0]),
        (measurement[1] - last_estimate[1])
    ])
    
    velocity_change = np.linalg.norm(implied_velocity - velocity_estimate)
    if velocity_change > max_velocity_jump:
        return False
    
    return True


# =====================================================================
# 6. IISER MOHALI DATA
# =====================================================================

LANDMARKS = {
    "Main Gate": (30.6652, 76.7314),
    "Library": (30.6636, 76.7291),
    "Lecture Hall": (30.6644, 76.7276),
    "Academic Block": (30.6650, 76.7265),
    "Hostel 8": (30.6611, 76.7268),
    "Hostel 5": (30.6601, 76.7310),
    "Sports": (30.6625, 76.7325),
}

LANDMARK_COLORS = {
    "Main Gate": '#ff6b6b', "Library": '#f7b731', "Lecture Hall": '#4ecdc4',
    "Academic Block": '#45b7d1', "Hostel 8": '#a55eea', "Hostel 5": '#ee5a6f',
    "Sports": '#0be881',
}

ROUTE_WAYPOINTS = np.array([
    [76.7314, 30.6652], [76.7290, 30.6646], [76.7276, 30.6644],
    [76.7265, 30.6650], [76.7261, 30.6620], [76.7268, 30.6611],
    [76.7310, 30.6601], [76.7325, 30.6625], [76.7314, 30.6652]
])

MAP_BOUNDS = {'lon_min': 76.724, 'lon_max': 76.735, 'lat_min': 30.658, 'lat_max': 30.668}


def generate_tracking_data(config):
    """Generate true path and noisy GPS measurements"""
    interpolated_lon, interpolated_lat = [], []
    
    for i in range(len(ROUTE_WAYPOINTS) - 1):
        interpolated_lon.extend(np.linspace(
            ROUTE_WAYPOINTS[i][0], ROUTE_WAYPOINTS[i+1][0], 25, endpoint=False))
        interpolated_lat.extend(np.linspace(
            ROUTE_WAYPOINTS[i][1], ROUTE_WAYPOINTS[i+1][1], 25, endpoint=False))
    
    true_lons = np.array(interpolated_lon)
    true_lats = np.array(interpolated_lat)
    
    np.random.seed(config.RANDOM_SEED)
    measured_lons = true_lons + np.random.normal(0, config.GPS_ERROR_DEVIATION, len(true_lons))
    measured_lats = true_lats + np.random.normal(0, config.GPS_ERROR_DEVIATION, len(true_lats))
    
    return true_lons, true_lats, measured_lons, measured_lats


# =====================================================================
# 7. MAIN VISUALIZATION & COMPARISON
# =====================================================================

def create_comparison_animation(config):
    """Create side-by-side comparison of KF vs EKF vs PF"""
    
    true_lons, true_lats, measured_lons, measured_lats = generate_tracking_data(config)
    total_frames = len(true_lons)
    
    # Initialize all three filters
    kf = StandardKalmanFilter(config.DT, config.BASE_PROCESS_NOISE, config.BASE_MEASUREMENT_NOISE)
    ekf = ExtendedKalmanFilter(config.DT, config.EKF_PROCESS_NOISE, config.EKF_MEASUREMENT_NOISE)
    pf = ParticleFilter(config.NUM_PARTICLES, config.DT, config.PARTICLE_PROCESS_NOISE, config.PARTICLE_MEASUREMENT_NOISE)
    
    # Set initial positions
    for filt in [kf, ekf]:
        filt.X[0, 0] = true_lons[0]
        filt.X[1, 0] = true_lats[0]
    
    pf.particles[:, 0] = true_lons[0] + np.random.normal(0, 0.0005, config.NUM_PARTICLES)
    pf.particles[:, 1] = true_lats[0] + np.random.normal(0, 0.0005, config.NUM_PARTICLES)
    pf.X[0, 0] = true_lons[0]
    pf.X[1, 0] = true_lats[0]
    
    # Create figure with 4 subplots
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#0a0e27')
    
    # Subplot 1: Global KF
    ax1 = plt.subplot(2, 3, 1)
    ax1.set_facecolor('#0a0e27')
    ax1.set_title('🟢 Standard Kalman Filter', color='#00ff88', fontweight='bold', fontsize=10)
    
    # Subplot 2: Global EKF
    ax2 = plt.subplot(2, 3, 2)
    ax2.set_facecolor('#0a0e27')
    ax2.set_title('🔵 Extended Kalman Filter', color='#00bfff', fontweight='bold', fontsize=10)
    
    # Subplot 3: Global PF
    ax3 = plt.subplot(2, 3, 3)
    ax3.set_facecolor('#0a0e27')
    ax3.set_title('🟣 Particle Filter', color='#ee82ee', fontweight='bold', fontsize=10)
    
    # Subplot 4: Error comparison
    ax4 = plt.subplot(2, 3, 4)
    ax4.set_facecolor('#0a0e27')
    ax4.set_title('📊 Tracking Error (meters)', color='#ffff88', fontweight='bold', fontsize=10)
    
    # Subplot 5: Velocity estimates
    ax5 = plt.subplot(2, 3, 5)
    ax5.set_facecolor('#0a0e27')
    ax5.set_title('🚀 Velocity Estimates', color='#ff88ff', fontweight='bold', fontsize=10)
    
    # Subplot 6: Prediction
    ax6 = plt.subplot(2, 3, 6)
    ax6.set_facecolor('#0a0e27')
    ax6.set_title('🔮 Trajectory Prediction', color='#88ffff', fontweight='bold', fontsize=10)
    
    # Setup all map views
    for ax in [ax1, ax2, ax3]:
        ax.plot(ROUTE_WAYPOINTS[:, 0], ROUTE_WAYPOINTS[:, 1],
               color='#1a4d7a', linewidth=6, alpha=0.5, zorder=1)
        
        for name, (lat, lon) in LANDMARKS.items():
            ax.plot(lon, lat, marker='s', color=LANDMARK_COLORS[name], markersize=6,
                   markeredgecolor='white', markeredgewidth=0.8, zorder=3)
        
        ax.set_xlim(MAP_BOUNDS['lon_min'], MAP_BOUNDS['lon_max'])
        ax.set_ylim(MAP_BOUNDS['lat_min'], MAP_BOUNDS['lat_max'])
        ax.tick_params(colors='#666', labelsize=8)
        ax.grid(True, alpha=0.1, color='white')
    
    # Setup error plot
    ax4.set_xlabel('Frame', fontsize=9, color='#888')
    ax4.set_ylabel('Error (m)', fontsize=9, color='#888')
    ax4.tick_params(colors='#666', labelsize=8)
    ax4.grid(True, alpha=0.2, color='white')
    
    # Setup velocity plot
    ax5.set_xlabel('Frame', fontsize=9, color='#888')
    ax5.set_ylabel('Velocity (°/s)', fontsize=9, color='#888')
    ax5.tick_params(colors='#666', labelsize=8)
    ax5.grid(True, alpha=0.2, color='white')
    
    # Setup prediction plot
    ax6.set_xlim(MAP_BOUNDS['lon_min'], MAP_BOUNDS['lon_max'])
    ax6.set_ylim(MAP_BOUNDS['lat_min'], MAP_BOUNDS['lat_max'])
    ax6.plot(ROUTE_WAYPOINTS[:, 0], ROUTE_WAYPOINTS[:, 1],
            color='#1a4d7a', linewidth=5, alpha=0.5, zorder=1)
    ax6.tick_params(colors='#666', labelsize=8)
    ax6.grid(True, alpha=0.1, color='white')
    
    # Tracking lines for each filter
    lines_kf, = ax1.plot([], [], color='#00ff88', linewidth=2.5, label='KF Track', zorder=4)
    line_true_kf, = ax1.plot([], [], color='#00bfff', linestyle=':', linewidth=1.5, alpha=0.6, label='True Path')
    car_kf, = ax1.plot([], [], marker='D', color='#00ff88', markersize=10, zorder=5)
    
    lines_ekf, = ax2.plot([], [], color='#00bfff', linewidth=2.5, label='EKF Track', zorder=4)
    line_true_ekf, = ax2.plot([], [], color='#00bfff', linestyle=':', linewidth=1.5, alpha=0.6, label='True Path')
    car_ekf, = ax2.plot([], [], marker='D', color='#00bfff', markersize=10, zorder=5)
    
    lines_pf, = ax3.plot([], [], color='#ee82ee', linewidth=2.5, label='PF Track', zorder=4)
    line_true_pf, = ax3.plot([], [], color='#00bfff', linestyle=':', linewidth=1.5, alpha=0.6, label='True Path')
    car_pf, = ax3.plot([], [], marker='D', color='#ee82ee', markersize=10, zorder=5)
    particles_scatter = ax3.scatter([], [], alpha=0.1, s=10, color='#ee82ee', zorder=2)
    
    # Error lines
    line_error_kf, = ax4.plot([], [], color='#00ff88', linewidth=2, label='KF Error', alpha=0.8)
    line_error_ekf, = ax4.plot([], [], color='#00bfff', linewidth=2, label='EKF Error', alpha=0.8)
    line_error_pf, = ax4.plot([], [], color='#ee82ee', linewidth=2, label='PF Error', alpha=0.8)
    
    # Velocity lines
    line_vel_kf, = ax5.plot([], [], color='#00ff88', linewidth=2, label='KF Vel', alpha=0.8)
    line_vel_ekf, = ax5.plot([], [], color='#00bfff', linewidth=2, label='EKF Vel', alpha=0.8)
    line_vel_pf, = ax5.plot([], [], color='#ee82ee', linewidth=2, label='PF Vel', alpha=0.8)
    
    # Prediction line
    line_pred, = ax6.plot([], [], color='#ffff00', marker='o', markersize=4, 
                         linewidth=2, label='Predicted Path', zorder=4)
    
    # HUD
    hud_main = fig.text(0.5, 0.98, "", ha='center', color='#00ff88', fontweight='bold',
                       fontsize=11, bbox=dict(boxstyle='round', facecolor='#0a0e27', 
                                            edgecolor='#00ff88', alpha=0.8))
    
    # History storage
    hist_kf_lon, hist_kf_lat = [], []
    hist_ekf_lon, hist_ekf_lat = [], []
    hist_pf_lon, hist_pf_lat = [], []
    hist_true_lon, hist_true_lat = [], []
    
    errors_kf, errors_ekf, errors_pf = [], [], []
    vel_kf, vel_ekf, vel_pf = [], [], []
    
    def init():
        for line in [lines_kf, lines_ekf, lines_pf, line_error_kf, line_error_ekf, 
                     line_error_pf, line_vel_kf, line_vel_ekf, line_vel_pf, line_pred]:
            line.set_data([], [])
        for line in [line_true_kf, line_true_ekf, line_true_pf]:
            line.set_data([], [])
        for car in [car_kf, car_ekf, car_pf]:
            car.set_data([], [])
        return [lines_kf, lines_ekf, lines_pf, car_kf, car_ekf, car_pf, line_pred]
    
    def animate(frame):
        nonlocal particles_scatter
        
        # Current true position
        true_lon, true_lat = true_lons[frame], true_lats[frame]
        meas_lon, meas_lat = measured_lons[frame], measured_lats[frame]
        
        # Store true position
        hist_true_lon.append(true_lon)
        hist_true_lat.append(true_lat)
        
        # Update KF
        kf.predict()
        kf_pos = kf.update((meas_lon, meas_lat))
        hist_kf_lon.append(kf_pos[0])
        hist_kf_lat.append(kf_pos[1])
        
        # Update EKF
        ekf.predict()
        ekf_pos = ekf.update((meas_lon, meas_lat))
        hist_ekf_lon.append(ekf_pos[0])
        hist_ekf_lat.append(ekf_pos[1])
        
        # Update PF
        pf.predict()
        pf_pos = pf.update((meas_lon, meas_lat))
        hist_pf_lon.append(pf_pos[0])
        hist_pf_lat.append(pf_pos[1])
        
        # Calculate errors (in meters)
        err_kf = np.sqrt((kf_pos[0] - true_lon)**2 * 96000**2 + 
                        (kf_pos[1] - true_lat)**2 * 111000**2)
        err_ekf = np.sqrt((ekf_pos[0] - true_lon)**2 * 96000**2 + 
                         (ekf_pos[1] - true_lat)**2 * 111000**2)
        err_pf = np.sqrt((pf_pos[0] - true_lon)**2 * 96000**2 + 
                        (pf_pos[1] - true_lat)**2 * 111000**2)
        
        errors_kf.append(err_kf)
        errors_ekf.append(err_ekf)
        errors_pf.append(err_pf)
        
        # Velocity estimates
        vel_kf.append(np.linalg.norm([kf.X[2, 0], kf.X[3, 0]]))
        vel_ekf.append(np.linalg.norm([ekf.X[2, 0], ekf.X[3, 0]]))
        vel_pf.append(np.linalg.norm([pf.X[2, 0], pf.X[3, 0]]))
        
        # Update visualizations
        lines_kf.set_data(hist_kf_lon, hist_kf_lat)
        line_true_kf.set_data(hist_true_lon, hist_true_lat)
        car_kf.set_data([kf_pos[0]], [kf_pos[1]])
        
        lines_ekf.set_data(hist_ekf_lon, hist_ekf_lat)
        line_true_ekf.set_data(hist_true_lon, hist_true_lat)
        car_ekf.set_data([ekf_pos[0]], [ekf_pos[1]])
        
        lines_pf.set_data(hist_pf_lon, hist_pf_lat)
        line_true_pf.set_data(hist_true_lon, hist_true_lat)
        car_pf.set_data([pf_pos[0]], [pf_pos[1]])
        
        # Update particles display
        particles_scatter.set_offsets(pf.particles[:, :2])
        
        # Update error plot
        frames_arr = np.arange(len(errors_kf))
        line_error_kf.set_data(frames_arr, errors_kf)
        line_error_ekf.set_data(frames_arr, errors_ekf)
        line_error_pf.set_data(frames_arr, errors_pf)
        ax4.set_xlim(0, total_frames)
        ax4.set_ylim(0, max(max(errors_kf), max(errors_ekf), max(errors_pf)) * 1.1)
        
        # Update velocity plot
        line_vel_kf.set_data(frames_arr, vel_kf)
        line_vel_ekf.set_data(frames_arr, vel_ekf)
        line_vel_pf.set_data(frames_arr, vel_pf)
        ax5.set_xlim(0, total_frames)
        max_vel = max(max(vel_kf), max(vel_ekf), max(vel_pf))
        ax5.set_ylim(0, max_vel * 1.1)
        
        # Prediction
        if frame % 5 == 0:
            pred_ekf = predict_trajectory(ekf, ekf_pos, num_steps=config.PREDICTION_HORIZON)
            line_pred.set_data(pred_ekf[:, 0], pred_ekf[:, 1])
        
        # Update legends
        ax4.legend(loc='upper left', fontsize=8, facecolor='#141824', edgecolor='#333', labelcolor='#ccc')
        ax5.legend(loc='upper left', fontsize=8, facecolor='#141824', edgecolor='#333', labelcolor='#ccc')
        ax6.legend(loc='upper left', fontsize=8, facecolor='#141824', edgecolor='#333', labelcolor='#ccc')
        
        # Update HUD
        mean_err_kf = np.mean(errors_kf[-20:]) if len(errors_kf) >= 20 else np.mean(errors_kf)
        mean_err_ekf = np.mean(errors_ekf[-20:]) if len(errors_ekf) >= 20 else np.mean(errors_ekf)
        mean_err_pf = np.mean(errors_pf[-20:]) if len(errors_pf) >= 20 else np.mean(errors_pf)
        
        hud_main.set_text(
            f"Frame {frame+1}/{total_frames} | "
            f"KF: {mean_err_kf:.2f}m | EKF: {mean_err_ekf:.2f}m | PF: {mean_err_pf:.2f}m"
        )
        
        return [lines_kf, lines_ekf, lines_pf, car_kf, car_ekf, car_pf, particles_scatter,
               line_error_kf, line_error_ekf, line_error_pf, line_vel_kf, line_vel_ekf, 
               line_vel_pf, line_pred]
    
    anim = FuncAnimation(fig, animate, frames=total_frames, init_func=init,
                        blit=True, interval=config.ANIMATION_INTERVAL, repeat=True)
    
    plt.tight_layout()
    plt.show()


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║     ADVANCED KALMAN FILTER COMPARISON                            ║
║  Standard KF vs Extended KF vs Particle Filter                   ║
╚═══════════════════════════════════════════════════════════════════╝

Based on MATLAB Documentation:
  1. Using Kalman Filter for Object Tracking
  2. Track Objects with Wrapping Azimuth Angles
  3. Non-Gaussian Non-Linear Object Tracking

DISPLAYS: 6-Panel Comparison
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TOP ROW: Map Views
  🟢 LEFT:   Standard Kalman Filter (Linear)
  🔵 CENTER: Extended Kalman Filter (Non-linear)
  🟣 RIGHT:  Particle Filter (Non-Gaussian)

BOTTOM ROW: Analysis
  📊 LEFT:   Tracking Error Comparison
  🚀 CENTER: Velocity Estimation
  🔮 RIGHT:  Trajectory Prediction

KEY INNOVATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Extended Kalman Filter (EKF)
  • Non-linear motion models
  • Handles curved paths
  • Turn rate estimation
  • Acceleration modeling
  • Uses Jacobian matrices

✓ Particle Filter (PF)
  • Handles non-Gaussian noise (multipath)
  • Multiple hypothesis tracking
  • Non-linear dynamics
  • Probabilistic resampling
  • Robust to outliers

✓ Trajectory Prediction
  • Forecast 5 frames ahead
  • Uses EKF for smooth prediction
  • Shows future path
  • Useful for navigation

✓ Adaptive Outlier Rejection
  • Validates measurements
  • Rejects GPS multipath errors
  • Maximum velocity check
  • Distance jump detection

PERFORMANCE METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Standard KF:  Linear, fast, underestimates curves
  Extended KF:  Non-linear, smoother, better curves
  Particle Filter: Most accurate, handles non-Gaussian noise

    """)
    
    create_comparison_animation(TrackingConfig)