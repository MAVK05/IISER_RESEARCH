import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation

# =====================================================================
# 1. GEOGRAPHIC 2D KALMAN FILTER
# =====================================================================
class GeoKalmanFilter:
    def __init__(self, dt=1.0, process_noise=1e-7, measurement_noise=2e-5):
        self.dt = dt
        # State vector: [Longitude, Latitude, Longitude_Velocity, Latitude_Velocity]^T
        self.X = np.zeros((4, 1))
        
        # State transition (Constant velocity model in coordinate degrees)
        self.F = np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1]
        ])
        # Measurement matrix (Direct GPS reading of Lon and Lat)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        self.Q = np.eye(4) * process_noise
        self.R = np.eye(2) * measurement_noise
        self.P = np.eye(4) * 1e-4

    def predict(self):
        self.X = np.dot(self.F, self.X)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.X[:2].flatten()

    def update(self, measurement):
        Z = np.array(measurement).reshape(2, 1)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.X = self.X + np.dot(K, (Z - np.dot(self.H, self.X)))
        self.P = np.dot((np.eye(4) - np.dot(K, self.H)), self.P)
        return self.X[:2].flatten()


# =====================================================================
# 2. ADAPTIVE GEOGRAPHIC GRID MESH
# =====================================================================
def build_geo_density(lon_grid, lat_grid, target_lon, target_lat, sigma=0.0006):
    """Generates a tracking density mask based on spatial coordinate distance."""
    matrix = np.exp(-((lon_grid - target_lon)**2 + (lat_grid - target_lat)**2) / (2 * sigma**2))
    return matrix

def compute_adaptive_mesh(mat, lon_idx, lat_idx, lon_size, lat_size, threshold, min_size, mesh_accumulator):
    """Subdivides coordinates recursively depending on tracking intensity variance."""
    block = mat[lat_idx:lat_idx+lat_size, lon_idx:lon_idx+lon_size]
    if block.size == 0: return
    
    variation = block.max() - block.min()
    if variation <= threshold or lat_size <= min_size or lon_size <= min_size:
        mesh_accumulator.append((lon_idx, lat_idx, lon_size, lat_size))
        return
    
    half_lon, half_lat = lon_size // 2, lat_size // 2
    compute_adaptive_mesh(mat, lon_idx,            lat_idx,            half_lon,          half_lat,          threshold, min_size, mesh_accumulator)
    compute_adaptive_mesh(mat, lon_idx + half_lon,  lat_idx,            lon_size - half_lon, half_lat,          threshold, min_size, mesh_accumulator)
    compute_adaptive_mesh(mat, lon_idx,            lat_idx + half_lat,  half_lon,          lat_size - half_lat, threshold, min_size, mesh_accumulator)
    compute_adaptive_mesh(mat, lon_idx + half_lon,  lat_idx + half_lat,  lon_size - half_lon, lat_size - half_lat, threshold, min_size, mesh_accumulator)


# =====================================================================
# 3. REAL IISER MOHALI MAP COORDINATES & WAYPOINTS
# =====================================================================
# Actual GPS locations on campus
landmarks = {
    "Main Gate (Sector 81)": (30.6652, 76.7314),
    "Lecture Hall Complex":  (30.6644, 76.7276),
    "Academic Block 1":      (30.6650, 76.7265),
    "Central Library":       (30.6636, 76.7291),
    "Hostel 8":              (30.6611, 76.7268),
    "Hostel 5":              (30.6601, 76.7310),
    "Sports Complex":        (30.6625, 76.7325),
}

# Driving loop path along the peripheral loop and main avenues (Longitude, Latitude)
route_waypoints = np.array([
    [76.7314, 30.6652],  # Main Gate
    [76.7290, 30.6646],  # Library Junction
    [76.7276, 30.6644],  # Past LHC
    [76.7265, 30.6650],  # Academic Blocks North Road
    [76.7261, 30.6620],  # Turning South along the boundary toward H7/H8
    [76.7268, 30.6611],  # Hostel 8 Residential sector
    [76.7310, 30.6601],  # Heading East past Hostel 5 / Dining Hall
    [76.7325, 30.6625],  # Moving North along Sports Fields / Faculty Housing
    [76.7314, 30.6652]   # Loop complete at Main Gate
])

# Generate steps by interpolating smoothly between the map waypoints
interpolated_lon = []
interpolated_lat = []
steps_per_segment = 25

for i in range(len(route_waypoints) - 1):
    interpolated_lon.extend(np.linspace(route_waypoints[i][0], route_waypoints[i+1][0], steps_per_segment, endpoint=False))
    interpolated_lat.extend(np.linspace(route_waypoints[i][1], route_waypoints[i+1][1], steps_per_segment, endpoint=False))

true_lons = np.array(interpolated_lon)
true_lats = np.array(interpolated_lat)
total_frames = len(true_lons)

# Simulating standard GPS multi-path reflection errors (bouncing off concrete structures)
np.random.seed(42)
gps_error_deviation = 0.0003  # Roughly translates to 20-30 meter jitter spikes
measured_lons = true_lons + np.random.normal(0, gps_error_deviation, total_frames)
measured_lats = true_lats + np.random.normal(0, gps_error_deviation, total_frames)


# =====================================================================
# 4. ENVIRONMENT MATRICES AND PLOTTING SIMULATION
# =====================================================================
# Map Bounds defining the bounding view around IISER Mohali
MIN_LON, MAX_LON = 76.724, 76.735
MIN_LAT, MAX_LAT = 30.658, 30.668

RESOLUTION = 128  # Resolution for spatial decomposition
lon_space = np.linspace(MIN_LON, MAX_LON, RESOLUTION)
lat_space = np.linspace(MAX_LAT, MIN_LAT, RESOLUTION)  # Top-down tracking orientation
lon_grid, lat_grid = np.meshgrid(lon_space, lat_space)

geo_kf = GeoKalmanFilter(dt=1.0)
geo_kf.X[0, 0] = true_lons[0]
geo_kf.X[1, 0] = true_lats[0]

fig, ax = plt.subplots(figsize=(10, 10))
fig.patch.set_facecolor('#14161d')
ax.set_facecolor('#14161d')

# Map elements
ax.plot(route_waypoints[:, 0], route_waypoints[:, 1], color='#292f3d', linewidth=6, alpha=0.7, label='Campus Road Network', zorder=1)

for name, (lat_val, lon_val) in landmarks.items():
    ax.plot(lon_val, lat_val, marker='h', color='#3b4354', markersize=7)
    ax.text(lon_val + 0.0001, lat_val + 0.0001, name, color='#8da0ba', fontsize=8, fontweight='semibold')

# Tracking UI components
history_lon, history_lat = [], []
history_gps_lon, history_gps_lat = [], []
history_kf_lon, history_kf_lat = [], []

mesh_patches = []
density_plot = ax.imshow(np.zeros((RESOLUTION, RESOLUTION)), cmap='plasma', alpha=0.2, 
                         extent=[MIN_LON, MAX_LON, MIN_LAT, MAX_LAT], origin='upper', zorder=2)

line_route, = ax.plot([], [], color='#00bfff', linestyle=':', linewidth=1.2, label='Intended Path Route')
line_track, = ax.plot([], [], color='#00ff88', linestyle='-', linewidth=2.5, label='Kalman Filtered Tracking Pos')
scatter_gps, = ax.plot([], [], color='#ff4a4a', marker='.', linestyle='None', markersize=4, alpha=0.5, label='Raw GPS Logs')
car_cursor, = ax.plot([], [], color='#00ff88', marker='o', markersize=10, markeredgecolor='white', zorder=10)

ax.set_xlim(MIN_LON, MAX_LON)
ax.set_ylim(MIN_LAT, MAX_LAT)
ax.axis('off')

ax.legend(loc='lower left', facecolor='#1b1e26', edgecolor='#2d3340', labelcolor='#ccd6e3')
status_hud = ax.text(0.5, 0.95, "", transform=ax.transAxes, color='#ffffff', weight='bold', fontsize=11, ha='center')

def run_init():
    line_route.set_data([], [])
    line_track.set_data([], [])
    scatter_gps.set_data([], [])
    car_cursor.set_data([], [])
    status_hud.set_text("")
    return line_route, line_track, scatter_gps, car_cursor, status_hud

def advance_simulation(frame):
    # Sweep out previous iteration rectangles
    for patch in mesh_patches:
        patch.remove()
    mesh_patches.clear()
    
    curr_lon, curr_lat = true_lons[frame], true_lats[frame]
    gps_lon,  gps_lat  = measured_lons[frame], measured_lats[frame]
    
    # State update
    geo_kf.predict()
    kf_lon, kf_lat = geo_kf.update((gps_lon, gps_lat))
    
    history_lon.append(curr_lon)
    history_lat.append(curr_lat)
    history_gps_lon.append(gps_lon)
    history_gps_lat.append(gps_lat)
    history_kf_lon.append(kf_lon)
    history_kf_lat.append(kf_lat)
    
    # Re-map Quadtree tracking based on dynamic positional tracking density matrix
    density_map = build_geo_density(lon_grid, lat_grid, target_lon=kf_lon, target_lat=kf_lat)
    density_plot.set_data(density_map)
    
    accumulated_cells = []
    compute_adaptive_mesh(density_map, 0, 0, RESOLUTION, RESOLUTION, threshold=0.08, min_size=2, mesh_accumulator=accumulated_cells)
    
    # Overlay the adaptive grid patches translated into geo-coordinates bounding frames
    for (lon_idx, lat_idx, lon_w, lat_h) in accumulated_cells:
        geo_x = lon_space[lon_idx]
        geo_y = lat_space[lat_idx]
        geo_w = lon_space[min(lon_idx + lon_w, RESOLUTION-1)] - geo_x
        geo_y_bottom = lat_space[min(lat_idx + lat_h, RESOLUTION-1)]
        geo_h = geo_y_bottom - geo_y
        
        rect = Rectangle((geo_x, geo_y), geo_w, geo_h, fill=False, edgecolor='#424a5d', linewidth=0.4, alpha=0.3, zorder=3)
        ax.add_patch(rect)
        mesh_patches.append(rect)
        
    # Synchronize tracking lines
    line_route.set_data(history_lon, history_lat)
    line_track.set_data(history_kf_lon, history_kf_lat)
    scatter_gps.set_data(history_gps_lon, history_gps_lat)
    car_cursor.set_data([kf_lon], [kf_lat])
    
    # Proximity calculation to real campus landmarks
    proxities = {lbl: np.hypot((kf_lat - pos[0])*111000, (kf_lon - pos[1])*96000) for lbl, pos in landmarks.items()}
    closest = min(proxities, key=proxities.get)
    
    status_hud.set_text(f"IISER Mohali Navigation Track | Passing: {closest} ({proxities[closest]:.1f}m away)")
    
    return [line_route, line_track, scatter_gps, car_cursor, status_hud] + mesh_patches

anim = FuncAnimation(fig, advance_simulation, frames=total_frames, init_func=run_init, blit=True, interval=70, repeat=True)
plt.show()
