import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation

# =====================================================================
# 1. KALMAN FILTER CLASS
# =====================================================================
class KalmanFilter2D:
    def __init__(self, dt, process_noise=0.1, measurement_noise=2.0):
        self.dt = dt
        
        # State Vector: [x, y, vx, vy]^T
        self.X = np.zeros((4, 1))
        
        # State Transition Matrix (Physics Model: pos = pos + vel * dt)
        self.F = np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1]
        ])
        
        # Measurement Matrix (We only directly measure x and y positions)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process Noise Covariance (Uncertainty in systemic acceleration/physics)
        self.Q = np.eye(4) * process_noise
        
        # Measurement Noise Covariance (Sensor variance / GPS inaccuracy)
        self.R = np.eye(2) * measurement_noise
        
        # State Covariance Matrix (Initial confidence setup)
        self.P = np.eye(4) * 10.0

    def predict(self):
        # X_prior = F * X
        self.X = np.dot(self.F, self.X)
        # P_prior = F * P * F^T + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.X[:2].flatten()

    def update(self, measurement):
        # Z: Current measurement matrix [x, y]^T
        Z = np.array(measurement).reshape(2, 1)
        
        # Innovation/Residual covariance: S = H * P * H^T + R
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        
        # Optimal Kalman Gain: K = P * H^T * inv(S)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # Update state estimate: X = X + K * (Z - H * X)
        self.X = self.X + np.dot(K, (Z - np.dot(self.H, self.X)))
        
        # Update state covariance: P = (I - K * H) * P
        I = np.eye(4)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
        return self.X[:2].flatten()


# =====================================================================
# 2. ADAPTIVE GRID GENERATION (QUADTREE)
# =====================================================================
def generate_density_matrix(rows, cols, target_r, target_c, sigma=6.0):
    """Generates a gaussian density grid centered on the tracked car target."""
    y_idx, x_idx = np.ogrid[:rows, :cols]
    matrix = np.exp(-((x_idx - target_c)**2 + (y_idx - target_r)**2) / (2 * sigma**2))
    return matrix

def compute_adaptive_grid(mat, r, c, h, w, threshold, min_size, grid_cells):
    """Recursive Quadtree subdivision based on localized spatial variance."""
    block = mat[r:r+h, c:c+w]
    if block.size == 0:
        return
    
    variation = block.max() - block.min()
    if variation <= threshold or h <= min_size or w <= min_size:
        grid_cells.append((r, c, h, w))
        return
    
    half_h, half_w = h // 2, w // 2
    compute_adaptive_grid(mat, r,          c,          half_h,     half_w,     threshold, min_size, grid_cells)
    compute_adaptive_grid(mat, r,          c + half_w, half_h,     w - half_w, threshold, min_size, grid_cells)
    compute_adaptive_grid(mat, r + half_h, c,          h - half_h, half_w,     threshold, min_size, grid_cells)
    compute_adaptive_grid(mat, r + half_h, c + half_w, h - half_h, w - half_w, threshold, min_size, grid_cells)


# =====================================================================
# 3. ENVIRONMENT AND SIMULATION SETUP
# =====================================================================
ROWS, COLS = 100, 100
TOTAL_FRAMES = 120
DT = 1.0

# Instantiating the Tracker
kf = KalmanFilter2D(dt=DT, process_noise=0.05, measurement_noise=15.0)

# Generate a true movement path (Circular Trajectory inside the grid bounds)
t = np.linspace(0, 2 * np.pi, TOTAL_FRAMES)
center_r, center_c = ROWS / 2, COLS / 2
radius = 35
true_cols = center_c + radius * np.cos(t)
true_rows = center_r + radius * np.sin(t)

# Initialize Kalman State close to where the vehicle spawns
kf.X[0, 0] = true_cols[0]
kf.X[1, 0] = true_rows[0]

# Pre-generate measurements with artificial Gaussian noise
np.random.seed(42) # Consistent noise profiles
noise_magnitude = 3.5
measured_cols = true_cols + np.random.normal(0, noise_magnitude, TOTAL_FRAMES)
measured_rows = true_rows + np.random.normal(0, noise_magnitude, TOTAL_FRAMES)


# =====================================================================
# 4. ANIMATION & VISUALIZATION ENGINE
# =====================================================================
fig, ax = plt.subplots(figsize=(9, 9))
fig.patch.set_facecolor('#0d0d1a')
ax.set_facecolor('#0d0d1a')

# Storage arrays to render path histories
history_true_x, history_true_y = [], []
history_meas_x, history_meas_y = [], []
history_kf_x, history_kf_y = [], []

# Creating UI element containers
rect_patches = []
img_display = ax.imshow(np.zeros((ROWS, COLS)), cmap='plasma', origin='upper', vmin=0, vmax=1, alpha=0.6)

# Trace lines
line_true, = ax.plot([], [], color='#00ffcc', linestyle='-', linewidth=1.5, label='True Path')
line_kf,   = ax.plot([], [], color='#ffcc00', linestyle='-', linewidth=2.0, label='Kalman Filter Track')
scatter_meas, = ax.plot([], [], color='#ff3333', marker='o', linestyle='None', markersize=6, alpha=0.7, label='Noisy GPS Reading')

# Dynamic points representing the current state
point_true, = ax.plot([], [], color='#00ffcc', marker='o', markersize=9, markeredgecolor='white')
point_kf,   = ax.plot([], [], color='#ffcc00', marker='X', markersize=11, markeredgecolor='black')

ax.legend(loc='upper right', facecolor='#1a1a2e', framealpha=0.9, labelcolor='white', edgecolor='#555566')
ax.set_xlim(-5, COLS + 5)
ax.set_ylim(ROWS + 5, -5) # Kept inverted to mimic standard matrix orientation (Row 0 at top)
ax.set_title("Car Tracking: Adaptive Quadtree Grid + Kalman Filtering", color='white', pad=15)

def init_anim():
    line_true.set_data([], [])
    line_kf.set_data([], [])
    scatter_meas.set_data([], [])
    point_true.set_data([], [])
    point_kf.set_data([], [])
    return line_true, line_kf, scatter_meas, point_true, point_kf

def update_frame(frame):
    global rect_patches
    
    # 1. Clear out old quadtree rectangles from previous visual updates
    for patch in rect_patches:
        patch.remove()
    rect_patches.clear()
    
    # 2. Get true position and noisy data point
    r_true, c_true = true_rows[frame], true_cols[frame]
    r_meas, c_meas = measured_rows[frame], measured_cols[frame]
    
    # 3. Apply Kalman Filter Step
    kf.predict()
    c_kf, r_kf = kf.update((c_meas, r_meas)) # Feed measurement to update prediction
    
    # Append values to coordinate history buffers
    history_true_x.append(c_true)
    history_true_y.append(r_true)
    history_meas_x.append(c_meas)
    history_meas_y.append(r_meas)
    history_kf_x.append(c_kf)
    history_kf_y.append(r_kf)
    
    # 4. Update Grid Resolution surrounding the filtered Kalman tracking position
    density_matrix = generate_density_matrix(ROWS, COLS, target_r=r_kf, target_c=c_kf, sigma=8.0)
    img_display.set_data(density_matrix)
    
    grid_cells = []
    compute_adaptive_grid(density_matrix, 0, 0, ROWS, COLS, threshold=0.05, min_size=2, grid_cells=grid_cells)
    
    # 5. Populate updated map subdivision meshes
    for (r, c, h, w) in grid_cells:
        rect = Rectangle((c - 0.5, r - 0.5), w, h, fill=False, edgecolor='#ffffff', linewidth=0.6, alpha=0.3)
        ax.add_patch(rect)
        rect_patches.append(rect)
        
    # 6. Synchronize trace line variables
    line_true.set_data(history_true_x, history_true_y)
    line_kf.set_data(history_kf_x, history_kf_y)
    scatter_meas.set_data(history_meas_x, history_meas_y)
    
    point_true.set_data([c_true], [r_true])
    point_kf.set_data([c_kf], [r_kf])
    
    return [line_true, line_kf, scatter_meas, point_true, point_kf] + rect_patches

anim = FuncAnimation(fig, update_frame, frames=TOTAL_FRAMES, init_func=init_anim, blit=True, interval=50, repeat=True)
plt.show()