# 🔍 HOW THE ADVANCED TRACKING SYSTEM WORKS
## Complete Code Walkthrough & Mechanics

---

## 📋 TABLE OF CONTENTS

1. [Overall Architecture](#overall-architecture)
2. [Data Flow](#data-flow)
3. [Standard Kalman Filter (KF)](#standard-kalman-filter-mechanics)
4. [Extended Kalman Filter (EKF)](#extended-kalman-filter-mechanics)
5. [Particle Filter (PF)](#particle-filter-mechanics)
6. [Animation Loop](#animation-loop-detailed)
7. [State Management](#state-management)
8. [Comparison Logic](#comparison-logic)

---

## 📐 OVERALL ARCHITECTURE

```
┌─────────────────────────────────────────────────────────┐
│               ADVANCED TRACKING SYSTEM                   │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
    ┌───▼────┐          ┌───▼────┐         ┌───▼─────┐
    │   KF   │          │  EKF   │         │   PF    │
    │(Linear)│          │(NonLin)│         │(Particl)│
    └───┬────┘          └───┬────┘         └───┬─────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                ┌───────────▼───────────┐
                │  Comparison Analysis  │
                │  - Error tracking     │
                │  - Velocity estimate  │
                │  - Predictions        │
                └───────────┬───────────┘
                            │
                ┌───────────▼──────────────┐
                │  6-Panel Visualization  │
                │  - 3 maps (top row)     │
                │  - 3 plots (bottom row) │
                └────────────────────────┘
```

---

## 🔄 DATA FLOW DIAGRAM

```
STEP 1: Load Configuration
┌──────────────────────┐
│ TrackingConfig       │
│ - DT = 1.0          │
│ - NUM_PARTICLES=500 │
│ - All noise params  │
└──────┬───────────────┘
       │
STEP 2: Generate Data
┌──────▼────────────────────┐
│ generate_tracking_data()  │
│ Returns:                  │
│ - true_lons/lats         │
│ - measured_lons/lats     │
│ (with GPS noise)         │
└──────┬────────────────────┘
       │
STEP 3: Initialize Filters
┌──────▼──────────────────────────┐
│ kf = StandardKalmanFilter(...)   │
│ ekf = ExtendedKalmanFilter(...)  │
│ pf = ParticleFilter(...)         │
│ Set X[0,0] = true_lons[0]       │
└──────┬──────────────────────────┘
       │
STEP 4: Create Visualization (6 panels)
┌──────▼──────────────────────────┐
│ create_comparison_animation()    │
│ Sets up:                         │
│ - ax1, ax2, ax3 (maps)          │
│ - ax4, ax5, ax6 (analysis)      │
└──────┬──────────────────────────┘
       │
STEP 5: Animate (Loop for each frame)
┌──────▼──────────────────────────┐
│ For frame = 0 to total_frames:  │
│                                  │
│ 1. Get measurement [lon, lat]    │
│ 2. For each filter:              │
│    - predict()                   │
│    - update(measurement)         │
│ 3. Calculate error metrics       │
│ 4. Update plots                  │
│ 5. Display HUD                   │
└──────┬──────────────────────────┘
       │
STEP 6: Output
┌──────▼──────────────────────────┐
│ Real-time 6-panel display       │
│ - Tracks all three              │
│ - Shows error comparison         │
│ - Displays predictions           │
└──────────────────────────────────┘
```

---

## 🎯 STANDARD KALMAN FILTER MECHANICS

### **Code Location:**
```
Lines 45-75: StandardKalmanFilter class
```

### **State Vector:**
```python
X = [longitude, latitude, velocity_lon, velocity_lat]^T

Example:
X = [[76.7314],       # Longitude
     [30.6652],       # Latitude
     [0.00001],       # Lon velocity (deg/s)
     [0.00001]]       # Lat velocity (deg/s)
```

### **How It Works: Step-by-Step**

#### **PREDICT STEP:**
```python
def predict(self):
    self.X = self.F @ self.X
    self.P = self.F @ self.P @ self.F.T + self.Q
```

**What happens:**
```
F = [[1, 0, dt, 0],      # lon_new = lon_old + vel_lon * dt
     [0, 1, 0, dt],      # lat_new = lat_old + vel_lat * dt
     [0, 0, 1, 0],       # vel_lon stays same
     [0, 0, 0, 1]]       # vel_lat stays same

Multiply F @ X:
┌──────────────────────────────────────┐
│ X_new[0] = X[0] + X[2] * dt          │
│ X_new[1] = X[1] + X[3] * dt          │
│ X_new[2] = X[2]                      │
│ X_new[3] = X[3]                      │
└──────────────────────────────────────┘

Example (dt=1.0):
OLD state: X = [76.7314, 30.6652, 0.00001, 0.00001]
NEW state: X = [76.73141, 30.66521, 0.00001, 0.00001]
           (position moved by velocity amount)

Covariance Update:
P_new = F @ P_old @ F^T + Q
        ↑                   ↑
   Spread velocity    Add process noise
   uncertainty to position   (models uncertainties)
```

#### **UPDATE STEP:**
```python
def update(self, measurement):
    Z = np.array(measurement).reshape(2, 1)  # GPS reading
    
    # Calculate innovation (what we actually measured vs predicted)
    y = Z - self.H @ self.X
    
    # Compute Kalman gain (how much to trust measurement)
    S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
    K = self.P @ self.H.T @ np.linalg.inv(S) # Kalman gain
    
    # Update state (blend prediction with measurement)
    self.X = self.X + K @ y
    
    # Update covariance (reduce uncertainty after measurement)
    self.P = (np.eye(4) - K @ self.H) @ self.P
```

**What each part does:**

```
H @ X = [[1, 0, 0, 0],      Extract just position (lon, lat)
         [0, 1, 0, 0]] @ X   from full state

EXAMPLE:
Predicted state: X = [76.7314, 30.6652, 0.00001, 0.00001]
H @ X = [76.7314, 30.6652]  (position only)

Measurement (GPS): Z = [76.7315, 30.6653]  (noisy GPS)

Innovation (difference):
y = Z - H @ X = [0.0001, 0.0001]  (GPS says we're slightly off)

Kalman Gain K:
- If GPS is very accurate (low R): K will be large → trust GPS more
- If GPS is noisy (high R): K will be small → trust prediction more

State Update:
X_new = X_old + K @ y

If K = 0.7 (trust GPS 70%):
  X_lon = 76.7314 + 0.7 * 0.0001 = 76.73147  (move toward GPS)
  X_lat = 30.6652 + 0.7 * 0.0001 = 30.66527

If K = 0.2 (trust GPS 20%, use prediction 80%):
  X_lon = 76.7314 + 0.2 * 0.0001 = 76.73142  (less movement)
```

**Visual flow:**
```
PREDICT:
  Old Position (76.7314, 30.6652)
       ↓
  Add velocity (0.00001, 0.00001)
       ↓
  Predicted Position (76.73141, 30.66521)
       ↓
UPDATE:
  Predicted (76.73141, 30.66521)
  GPS Measurement (76.7315, 30.6653)
       ↓
  Kalman Gain K calculates how much to blend
       ↓
  Final Estimate (76.73145, 30.66525)  ← Between prediction and GPS
```

---

## 🔄 EXTENDED KALMAN FILTER MECHANICS

### **Code Location:**
```
Lines 80-170: ExtendedKalmanFilter class
```

### **Key Difference from KF: Non-linear Motion**

#### **State Vector (6D):**
```python
X = [longitude, latitude, vel_lon, vel_lat, turn_rate, acceleration]^T

This allows modeling:
- Curved paths (turn_rate)
- Speed changes (acceleration)
- More realistic vehicle motion
```

#### **Non-linear Motion Function:**
```python
def _f(self, x):
    """Non-linear state transition"""
    x_new = x.copy()
    
    # Position updated by velocity
    x_new[0] += x[2] * self.dt  # lon += vel_lon * dt
    x_new[1] += x[3] * self.dt  # lat += vel_lat * dt
    
    # Handle turn rate (rotate velocity vector)
    vel_magnitude = np.sqrt(x[2]**2 + x[3]**2)
    if vel_magnitude > 1e-8:
        angle = np.arctan2(x[3], x[2])  # Current direction
        angle += x[4] * self.dt         # Add turn rate
        x_new[2] = vel_magnitude * np.cos(angle)  # New vel_lon
        x_new[3] = vel_magnitude * np.sin(angle)  # New vel_lat
    
    # Acceleration applied to velocity
    x_new[2] += x[5] * self.dt  # vel_lon += accel * dt
    x_new[3] += x[5] * self.dt  # vel_lat += accel * dt
    
    # Damping
    x_new[2:4] *= self.velocity_damping  # Reduce sudden changes
    
    return x_new
```

**Visual example of turn modeling:**
```
Without turn rate (Standard KF):
  Velocity: [0.00001, 0]  (moving East)
  Next frame: Still [0.00001, 0]
  Result: Straight line only ❌

With turn rate (EKF):
  Velocity: [0.00001, 0]  (moving East)
  Turn rate: 0.1 rad/frame
  Angle changes from 0° to 0.1 rad (5.7°)
  New velocity: [0.0000998, 0.00000573]  (rotated vector!)
  Result: Vehicle turns! ✅

Visualization:
  Without turn rate:         With turn rate:
     ──────────────────         ╱──
     ──────────────────        ╱──
     ──────────────────       ╱──
     (straight line)         (curved path)
```

#### **Jacobian Matrices:**
```python
def _state_transition_jacobian(self):
    """Linearize non-linear function around current state"""
    F_jac = np.eye(6)
    F_jac[0, 2] = self.dt  # ∂(lon_new)/∂(vel_lon) = dt
    F_jac[1, 3] = self.dt  # ∂(lat_new)/∂(vel_lat) = dt
    F_jac[2, 5] = self.dt  # ∂(vel_lon_new)/∂(accel) = dt
    F_jac[3, 5] = self.dt  # ∂(vel_lat_new)/∂(accel) = dt
    return F_jac
```

**Why Jacobians matter:**
```
Non-linear function f(x) can't be handled like linear F matrix
So we compute LOCAL approximation (linearization):

f(x) ≈ f(x_current) + F_jacobian * (x - x_current)

This allows using Kalman filter math on non-linear systems
```

#### **Predict in EKF:**
```python
def predict(self):
    # Non-linear prediction
    self.X = self._f(self.X)  ← Apply non-linear motion
    
    # Linear covariance update using Jacobian
    F_jac = self._state_transition_jacobian()
    self.P = F_jac @ self.P @ F_jac.T + self.Q
```

**Why this works better:**
```
KF on curves: Assumes straight line
  Predicted path: ─────────────
  Actual path:    ╱╱╱╱╱ (curved)
  Error: Big! ❌

EKF on curves: Models turns
  Predicted path: ╱╱╱╱╱
  Actual path:    ╱╱╱╱╱
  Error: Small! ✅
```

---

## 💫 PARTICLE FILTER MECHANICS

### **Code Location:**
```
Lines 175-250: ParticleFilter class
```

### **Core Concept:**
Instead of tracking ONE estimate (mean), track MANY particles (hypotheses)

```
Standard KF:          Particle Filter:
  Estimate: μ           Particles: {x₁, x₂, ..., x₅₀₀}
  Covariance: σ²        Weights: {w₁, w₂, ..., w₅₀₀}
  (One Gaussian)        (Multi-modal distribution)

Visual:
KF:        ┌──────┐              PF:  ●●●
           │ Mean │                   ●●●●●
           └──────┘                   ●●●●●
           (single point)             (500 particles)
```

### **Particle Initialization:**
```python
def __init__(self, num_particles=500, ...):
    # Start with 500 particles clustered around true position
    self.particles = np.zeros((num_particles, 4))
    # Each row: [lon, lat, vel_lon, vel_lat]
    
    self.weights = np.ones(num_particles) / num_particles
    # Each particle equally likely at start
```

**Example state:**
```
particles[0] = [76.7314, 30.6652, 0.00001, 0.00001]  ← Particle 1
particles[1] = [76.7315, 30.6653, 0.00002, 0.00001]  ← Particle 2
particles[2] = [76.7313, 30.6651, 0.00000, 0.00002]  ← Particle 3
...
particles[499] = [76.7314, 30.6652, 0.00001, 0.00001] ← Particle 500

weights[0] = 0.002  (1/500)
weights[1] = 0.002
...
weights[499] = 0.002
(All equal initially)
```

### **Predict Step:**
```python
def predict(self):
    self.particles = self._motion_model(self.particles)
    self.X[:2, 0] = np.average(self.particles[:, :2], 
                               axis=0, weights=self.weights)
```

**What happens:**
```
Motion model applied to EACH particle:
  For particle i:
    particles[i, 0] += particles[i, 2] * dt  (move by velocity)
    particles[i, 1] += particles[i, 3] * dt
    
    particles[i, 2] *= 0.95  (damping)
    particles[i, 3] *= 0.95
    
    particles[i, :] += noise  (add random walk)

State estimate = weighted average:
  X_lon = Σ(w_i * particles[i, 0])  (weighted sum)
  X_lat = Σ(w_i * particles[i, 1])

Example:
particles = [[76.7314, 30.6652],
             [76.7315, 30.6653],
             [76.7313, 30.6651],
             ...]

weights = [0.5, 0.3, 0.2, ...]  (some particles more likely)

X_lon = 0.5*76.7314 + 0.3*76.7315 + 0.2*76.7313 = 76.73141
X_lat = 0.5*30.6652 + 0.3*30.6653 + 0.2*30.6651 = 30.66521
```

### **Update Step (Key Magic):**
```python
def update(self, measurement):
    # Calculate measurement likelihood for each particle
    self.weights = self._measurement_model(self.particles, measurement)
    self.weights /= (np.sum(self.weights) + 1e-10)
    
    # Resample if needed
    n_eff = 1.0 / np.sum(self.weights**2)
    if n_eff < self.num_particles * 0.5:
        self._resample()
```

**Likelihood calculation:**
```python
def _measurement_model(self, particles, measurement):
    z = np.array(measurement)
    
    # How far each particle is from measurement
    distances = np.sqrt((particles[:, 0] - z[0])**2 + 
                       (particles[:, 1] - z[1])**2)
    
    # Gaussian likelihood (closer = higher)
    likelihood = np.exp(-distances**2 / (2 * self.measurement_noise**2))
    return likelihood

Example:
Measurement: [76.7315, 30.6653]

Particle 1: [76.7314, 30.6652]
  Distance: √(0.0001² + 0.0001²) = 0.000141
  Likelihood: exp(-0.000141²/2*1e-10) ≈ 1.0 (HIGH!) ✅

Particle 2: [76.7310, 30.6650]
  Distance: √(0.0005² + 0.0003²) = 0.000583
  Likelihood: exp(-0.000583²/2*1e-10) ≈ 0.5 (MEDIUM)

Particle 3: [76.7325, 30.6625]
  Distance: √(0.0010² + 0.0028²) = 0.00299
  Likelihood: exp(-0.00299²/2*1e-10) ≈ 0.01 (LOW!) ❌

Normalized weights:
w₁ = 1.0 / (1.0 + 0.5 + 0.01) ≈ 0.66  (Particle 1 most likely)
w₂ = 0.5 / 1.51 ≈ 0.33
w₃ = 0.01 / 1.51 ≈ 0.01
```

**Visual of weight update:**
```
BEFORE measurement:
  ●●●●●●●
  ●●●●●●●  (all equally likely)
  ●●●●●●●

AFTER measurement at [76.7315, 30.6653]:
  ●●●●●●●
  ●●●●●●●  (particles near measurement darker = higher weight)
  ●●○●●●●
```

### **Resampling (Critical Step):**
```python
def _resample(self):
    """Eliminate low-weight particles, duplicate high-weight ones"""
    
    # Get indices sorted by weight (highest first)
    indices = np.argsort(self.weights)[::-1]
    
    # Systematic resampling
    cum_weights = np.cumsum(self.weights[indices])
    
    new_indices = []
    u = np.random.uniform(0, 1.0 / self.num_particles)
    j = 0
    
    for i in range(self.num_particles):
        while u > cum_weights[j]:
            j += 1
        new_indices.append(indices[j])  # Keep this particle
        u += 1.0 / self.num_particles
    
    # Replace particles
    self.particles = self.particles[new_indices].copy()
    self.weights = np.ones(self.num_particles) / self.num_particles
```

**Why resampling matters:**
```
BEFORE resampling:
  Particle 1: weight = 0.66
  Particle 2: weight = 0.33
  Particle 3: weight = 0.01
  ...
  Particle 500: weight = 0.00000001

Problem: Particle 500 was sampled once 500 frames ago!
         Multiple copies of low-weight particles waste computation

AFTER resampling:
  Particle 1: weight = 0.002  (appeared ~330 times)
  Particle 2: weight = 0.002  (appeared ~165 times)
  Particle 3: weight = 0.002  (appeared ~5 times)
  ...
  (All weights equal again, but particles represent distribution better)

Visual:
Before:  ●●●●●●●
         ●●●●●●●  (mix of new and old)
         ●●●●●●●

After:   ●●●●●●●
         ●●●●●●●  (mostly good particles, few bad ones)
         ○●●●●●●
```

---

## 🎬 ANIMATION LOOP DETAILED

### **Code Location:**
```
Lines 330-620: create_comparison_animation function
```

### **Frame-by-Frame Execution:**

```
Frame 0:
  ├─ Get measurement: measured_lons[0], measured_lats[0]
  ├─ KF update:
  │  ├─ kf.predict()      # Linear prediction
  │  └─ kf.update(meas)   # Gaussian update
  ├─ EKF update:
  │  ├─ ekf.predict()     # Non-linear prediction
  │  └─ ekf.update(meas)  # Jacobian-based update
  ├─ PF update:
  │  ├─ pf.predict()      # Motion model on all particles
  │  └─ pf.update(meas)   # Likelihood weighting
  ├─ Calculate errors
  ├─ Store in history
  └─ Update plots

Frame 1:
  (Repeat for next measurement)

Frame 2:
  (Repeat...)

...continue until last frame
```

### **Code breakdown:**
```python
def animate(frame):
    # 1. GET CURRENT MEASUREMENT
    true_lon, true_lat = true_lons[frame], true_lats[frame]
    meas_lon, meas_lat = measured_lons[frame], measured_lats[frame]
    
    # 2. STORE TRUE POSITION
    hist_true_lon.append(true_lon)
    hist_true_lat.append(true_lat)
    
    # 3. UPDATE KALMAN FILTER
    kf.predict()                    # ← Predicts next position
    kf_pos = kf.update((meas_lon, meas_lat))  # ← Updates with measurement
    hist_kf_lon.append(kf_pos[0])
    hist_kf_lat.append(kf_pos[1])
    
    # 4. UPDATE EXTENDED KALMAN FILTER
    ekf.predict()                   # ← Non-linear predict
    ekf_pos = ekf.update((meas_lon, meas_lat))
    hist_ekf_lon.append(ekf_pos[0])
    hist_ekf_lat.append(ekf_pos[1])
    
    # 5. UPDATE PARTICLE FILTER
    pf.predict()                    # ← All 500 particles predicted
    pf_pos = pf.update((meas_lon, meas_lat))  # ← Weighted by likelihood
    hist_pf_lon.append(pf_pos[0])
    hist_pf_lat.append(pf_pos[1])
    
    # 6. CALCULATE ERRORS (in meters)
    err_kf = np.sqrt((kf_pos[0] - true_lon)**2 * 96000**2 + 
                    (kf_pos[1] - true_lat)**2 * 111000**2)
    err_ekf = np.sqrt((ekf_pos[0] - true_lon)**2 * 96000**2 + 
                     (ekf_pos[1] - true_lat)**2 * 111000**2)
    err_pf = np.sqrt((pf_pos[0] - true_lon)**2 * 96000**2 + 
                    (pf_pos[1] - true_lat)**2 * 111000**2)
    
    errors_kf.append(err_kf)
    errors_ekf.append(err_ekf)
    errors_pf.append(err_pf)
    
    # 7. UPDATE VISUALIZATION
    lines_kf.set_data(hist_kf_lon, hist_kf_lat)  # Draw line
    lines_ekf.set_data(hist_ekf_lon, hist_ekf_lat)
    lines_pf.set_data(hist_pf_lon, hist_pf_lat)
    
    car_kf.set_data([kf_pos[0]], [kf_pos[1]])    # Draw car
    car_ekf.set_data([ekf_pos[0]], [ekf_pos[1]])
    car_pf.set_data([pf_pos[0]], [pf_pos[1]])
    
    # 8. UPDATE ERROR PLOT
    line_error_kf.set_data(frames_arr, errors_kf)
    line_error_ekf.set_data(frames_arr, errors_ekf)
    line_error_pf.set_data(frames_arr, errors_pf)
    
    # 9. UPDATE HUD
    hud_main.set_text(f"Frame {frame+1}/{total_frames} | "
                      f"KF: {mean_err_kf:.2f}m | "
                      f"EKF: {mean_err_ekf:.2f}m | "
                      f"PF: {mean_err_pf:.2f}m")
```

### **Timeline Example (Frame 50):**

```
TIME t=50 (5 seconds in, each frame is 0.1s)

True position: (30.6652, 76.7314)
GPS measurement: (30.6653, 76.7315)  + 0.0001 noise

STEP 1 - KF Predict:
  X_old = [76.7314, 30.6651, 0.00001, 0.00002]
  X_pred = X_old + velocity * dt
  X_pred = [76.73141, 30.66512, 0.00001, 0.00002]

STEP 2 - KF Update:
  Kalman gain K calculated
  X_new = X_pred + K * (Z - H*X_pred)
  X_new = [76.73145, 30.66514, 0.00001, 0.00002]

STEP 3 - EKF Predict:
  X_old = [76.7314, 30.6651, 0.00001, 0.00002, 0.01, 0.001]
  Apply non-linear function _f()
  Handle turn rate rotation
  X_pred = [76.73141, 30.66512, 0.0000099, 0.0000198, 0.01, 0.001]
  (More realistic due to turn modeling)

STEP 4 - EKF Update:
  Jacobian used for covariance
  X_new = [76.73146, 30.66515, 0.0000099, 0.0000198, 0.01, 0.001]

STEP 5 - PF Predict:
  For each of 500 particles:
    Apply motion model
    Add noise
  particles[:, 0] += particles[:, 2] * 1.0 + noise
  particles[:, 1] += particles[:, 3] * 1.0 + noise

STEP 6 - PF Update:
  For each particle:
    distance = ||particle - measurement||
    weight = exp(-distance² / (2*R²))
  Particles near measurement get higher weights
  State = weighted average of particles

STEP 7 - Error Calculation:
  err_kf = distance from true to KF estimate
  err_ekf = distance from true to EKF estimate
  err_pf = distance from true to PF estimate
  
  Typically: err_pf < err_ekf < err_kf

STEP 8 - Display:
  Plot updated KF line (green)
  Plot updated EKF line (blue)
  Plot updated PF line (purple)
  Draw car at new position
  Update error plot (should see lines going down)
```

---

## 📊 STATE MANAGEMENT

### **Data Structures Maintained:**

```python
# For each filter:
kf.X           # 4×1 state vector
kf.P           # 4×4 covariance matrix
kf.F, kf.H     # Transition & measurement matrices
kf.Q, kf.R     # Process & measurement noise

ekf.X          # 6×1 state vector (includes turn, accel)
ekf.P          # 6×6 covariance matrix
ekf._f()       # Non-linear function
ekf._state_transition_jacobian()  # Linearization

pf.particles   # 500×4 array of states
pf.weights     # 500×1 array of importance weights
pf.X           # Weighted average state
pf._motion_model()     # Applies to all particles
pf._measurement_model()  # Likelihood calculation
```

### **History Storage (for plotting):**
```python
hist_kf_lon, hist_kf_lat         # All KF positions
hist_ekf_lon, hist_ekf_lat       # All EKF positions
hist_pf_lon, hist_pf_lat         # All PF positions
hist_true_lon, hist_true_lat     # Ground truth

errors_kf, errors_ekf, errors_pf # Tracking errors
vel_kf, vel_ekf, vel_pf          # Velocity estimates
```

### **Memory per frame:**
```
KF:  4 floats * 2 (state + covariance diag) = ~8 bytes
EKF: 6 floats * 2 = ~12 bytes
PF:  500 * 4 floats (particles) + 500 weights = ~8 KB
     (PF uses most memory)

For 200 frames:
KF:  200 * 2 pos * 8 bytes = 3.2 KB
PF:  200 * 8 KB = 1.6 MB
```

---

## ⚖️ COMPARISON LOGIC

### **How filters are compared:**

```python
# Calculate errors (all in meters)
err_kf = distance(kf_estimate, true_position)
err_ekf = distance(ekf_estimate, true_position)
err_pf = distance(pf_estimate, true_position)

# Track statistics
mean_kf = average(errors_kf)
mean_ekf = average(errors_ekf)
mean_pf = average(errors_pf)

# Plot on same graph
# X-axis: Frame number
# Y-axis: Error in meters
# 3 lines: KF (green), EKF (blue), PF (purple)

Expected plot shape:
    Error (m)
    │
 20 │ ╱ KF (green) - worst, higher error
    │╱╲
 10 │  ╲╱╲ EKF (blue) - medium, lower error
    │   ╲
  5 │    ╲_╱╲ PF (purple) - best, lowest error
    │      ╲
  0 └──────╲─────────── Frame
           └─ All should be decreasing initially
              then flatten (stabilize)
```

### **Why comparison matters:**
```
Single filter: Don't know if it's doing well
Three filters: Can see relative performance

KF baseline:  shows what linear model does
EKF middle:   shows improvement with non-linearity
PF best:      shows limit of accuracy with good noise handling

This teaches you:
  • When to use each filter
  • How much accuracy costs
  • Speed vs accuracy tradeoff
```

---

## 🎯 KEY INSIGHTS

### **Why KF does worst:**
```
Assumes X(k+1) = F * X(k)  (linear)
Assumes noise is Gaussian

On curves:
  Reality:  ╱╱╱╱
  KF:       ─────
  Error: Big!
```

### **Why EKF does better:**
```
Uses Jacobian linearization: F_jac * X(k)
Handles turn rate + acceleration

On curves:
  Reality: ╱╱╱╱
  EKF:     ╱╱╱╱
  Error: Small!
```

### **Why PF does best:**
```
Multiple particles represent distribution
No Gaussian assumption

On noisy data:
  Multi-modal uncertainty handled
  Robust to outliers
  Most accurate overall

Cost: 10-100x slower than KF
```

---

## 🔧 TUNING AFFECTS FILTERS DIFFERENTLY

```python
# Increase Q (trust velocity more)
Q = 1e-5  (was 5e-6)

KF:  Becomes smoother (lag increases)
EKF: Better turn tracking (less lag)
PF:  Particles spread more (less sharp turns)

# Increase R (trust GPS less)
R = 5e-5  (was 1e-5)

KF:  Relies more on prediction (lag increases)
EKF: Same effect as KF
PF:  Higher measurement noise threshold
```

---

## 📈 COMPUTATIONAL COMPLEXITY

```
Per frame:
KF:   4×4 matrices: O(16) operations
EKF:  6×6 matrices: O(36) operations, Jacobian calc
PF:   500 particles: O(2000) operations

Speed comparison:
KF:   ✅✅✅✅✅ (fastest)
EKF:  ✅✅✅✅   (2x slower)
PF:   ✅✅       (20x slower)

On 200 frames:
KF:   < 1ms
EKF:  ~2ms
PF:   ~50-100ms
```

---

## 🎉 SUMMARY

### **The flow:**
```
Load config
    ↓
Generate GPS data with noise
    ↓
Initialize 3 filters
    ↓
For each frame:
  Get measurement
    ↓
  For each filter: predict() → update()
    ↓
  Calculate error
    ↓
  Update plots
    ↓
  Display HUD
    ↓
  Next frame
    ↓
Display comparison (EKF wins on curves, PF best overall)
```

### **What makes each filter different:**

| Aspect | KF | EKF | PF |
|--------|----|----|-----|
| **Motion model** | Linear F matrix | Non-linear _f() | Random particles |
| **Covariance** | Matrix P | Jacobian linear approx | Particle spread |
| **Update** | Gaussian | Gaussian | Likelihood weighting |
| **Speed** | Fastest | Medium | Slowest |
| **Accuracy** | Lowest | Medium | Highest |
| **Best for** | Straight lines | Curves | Noisy non-Gaussian |

---

**The code demonstrates professional-grade tracking used in real-world autonomous systems!** 🚀