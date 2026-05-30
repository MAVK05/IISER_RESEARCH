import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches

# ============================================================
# k-WAVE STYLE SPHERE STRUCTURES IN PYTHON
# + INTEGRATED WITH 3D OCTREE
# ============================================================

def get_positive_int(prompt):
    while True:
        try:
            value = int(input(prompt))
            if value > 0:
                return value
            print("  ✗ Please enter a positive integer.")
        except ValueError:
            print("  ✗ Invalid input. Please enter a whole number.")

print("\n=== k-Wave Style 3D Sphere + Octree ===\n")
rows = get_positive_int("Enter rows   (suggest 30): ")
cols = get_positive_int("Enter cols   (suggest 30): ")
deps = get_positive_int("Enter depth  (suggest 30): ")

# ------------------------------------------------------------
# 1. BUILD 3D GAUSSIAN MATRIX (same as before)
# ------------------------------------------------------------

z_idx, y_idx, x_idx = np.ogrid[:deps, :rows, :cols]

sigma = max(min(rows, cols, deps) * 0.20, 4)

blobs_3d = [
    (rows*0.25, cols*0.25, deps*0.25, sigma, 1.00),
    (rows*0.75, cols*0.30, deps*0.65, sigma, 0.85),
    (rows*0.45, cols*0.75, deps*0.45, sigma, 0.90),
]

matrix_3d = np.zeros((deps, rows, cols))
for (cr, cc, cd, sg, amp) in blobs_3d:
    matrix_3d += amp * np.exp(
        -((x_idx - cc)**2
        + (y_idx - cr)**2
        + (z_idx - cd)**2) / (2 * sg**2)
    )
matrix_3d = (matrix_3d - matrix_3d.min()) / (matrix_3d.max() - matrix_3d.min() + 1e-10)

# ------------------------------------------------------------
# 2. BUILD SPHERE STRUCTURES (k-Wave style)
# ------------------------------------------------------------

cx = rows // 2
cy = cols // 2
cz = deps // 2
radius = min(rows, cols, deps) // 3

# coordinate grids
xg, yg, zg = np.mgrid[:rows, :cols, :deps]

# distance of every voxel from sphere centre
dist = np.sqrt((xg - cx)**2 + (yg - cy)**2 + (zg - cz)**2)

# ── Structure 1 : FILLED BALL (Image 1 style) ────────────────
# every voxel INSIDE sphere = True
filled_ball = dist <= radius

# ── Structure 2 : SURFACE SHELL (Image 2 style) ──────────────
# only voxels ON THE SURFACE = True
shell_thickness = 1.5
surface_shell   = np.abs(dist - radius) <= shell_thickness
surface_points  = np.argwhere(surface_shell)   # list of (x,y,z) coords

print(f"\n  Sphere centre  : ({cx}, {cy}, {cz})")
print(f"  Sphere radius  : {radius}")
print(f"  Filled voxels  : {filled_ball.sum()}")
print(f"  Surface points : {len(surface_points)}")

# ------------------------------------------------------------
# 3. OCTREE ON GAUSSIAN MATRIX
# ------------------------------------------------------------

THRESHOLD    = 0.05
MIN_SIZE     = 4
octree_cells = []

def adaptive_octree(mat, r, c, d, h, w, dp, threshold, min_size):
    block     = mat[d:d+dp, r:r+h, c:c+w]
    variation = block.max() - block.min()
    if (variation <= threshold
            or h  <= min_size
            or w  <= min_size
            or dp <= min_size):
        octree_cells.append((r, c, d, h, w, dp))
        return
    half_h  = h  // 2
    half_w  = w  // 2
    half_dp = dp // 2
    for (dr, new_h)  in [(0, half_h),  (half_h,  h  - half_h )]:
        for (dc, new_w)  in [(0, half_w),  (half_w,  w  - half_w )]:
            for (dd, new_dp) in [(0, half_dp), (half_dp, dp - half_dp)]:
                adaptive_octree(mat,
                                r+dr, c+dc, d+dd,
                                new_h, new_w, new_dp,
                                threshold, min_size)

adaptive_octree(matrix_3d, 0, 0, 0, rows, cols, deps, THRESHOLD, MIN_SIZE)

all_volumes   = [h*w*dp for (r,c,d,h,w,dp) in octree_cells]
vol_threshold = np.percentile(all_volumes, 40)

print(f"\n  Octree cells   : {len(octree_cells)}")

# ------------------------------------------------------------
# 4. VISUALISATION — 4 panels
# ------------------------------------------------------------

fig = plt.figure(figsize=(22, 16))
fig.patch.set_facecolor('#0d0d1a')

# ── Panel 1 : FILLED BALL — Image 1 style ────────────────────
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.set_facecolor('#0d0d1a')

# subsample to avoid slow rendering
step = max(1, rows * cols * deps // 5000)
xf   = xg[filled_ball][::step]
yf   = yg[filled_ball][::step]
zf   = zg[filled_ball][::step]

ax1.scatter(xf, yf, zf,
            c='#cccc00',          # yellow like k-Wave
            s=8,
            alpha=0.4,
            edgecolors='black',
            linewidths=0.2)

ax1.set_title('① Filled Ball\n(k-Wave makeBall style)',
              color='white', fontsize=11, pad=8)
ax1.set_xlabel('X [voxels]', color='#ffffff', fontsize=9)
ax1.set_ylabel('Y [voxels]', color='#ffffff', fontsize=9)
ax1.set_zlabel('Z [voxels]', color='#ffffff', fontsize=9)
ax1.tick_params(colors='#ffffff', labelsize=8)

# ── Panel 2 : SURFACE SHELL — Image 2 style ──────────────────
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax2.set_facecolor('#0d0d1a')

# subsample surface points
sp = surface_points[::max(1, len(surface_points)//800)]
ax2.scatter(sp[:, 0], sp[:, 1], sp[:, 2],
            c='black',            # black dots like k-Wave
            s=15,
            alpha=0.9)

ax2.set_title('② Surface Shell\n(k-Wave sensor mask style)',
              color='white', fontsize=11, pad=8)
ax2.set_xlabel('X [voxels]', color='#ffffff', fontsize=9)
ax2.set_ylabel('Y [voxels]', color='#ffffff', fontsize=9)
ax2.set_zlabel('Z [voxels]', color='#ffffff', fontsize=9)
ax2.tick_params(colors='#ffffff', labelsize=8)

# ── Panel 3 : XY slice with sphere boundary ──────────────────
pd0, pr0, pc0 = cz, cx, cy   # slice through sphere centre

ax3 = fig.add_subplot(2, 2, 3)
ax3.set_facecolor('#0d0d1a')
ax3.imshow(matrix_3d[pd0, :, :],
           cmap='turbo', origin='upper', vmin=0, vmax=1)

# draw octree cells
for (r, c, d, h, w, dp) in octree_cells:
    if d <= pd0 < d + dp:
        color = 'red' if h*w*dp <= vol_threshold else 'white'
        ax3.add_patch(patches.Rectangle(
            (c-0.5, r-0.5), w, h,
            fill=False, edgecolor=color, linewidth=0.8, alpha=0.7
        ))

# draw sphere circle boundary on this slice
theta  = np.linspace(0, 2*np.pi, 200)
circ_x = cy + radius * np.cos(theta)
circ_y = cx + radius * np.sin(theta)
ax3.plot(circ_x, circ_y,
         color='yellow', linewidth=2.0,
         linestyle='--', label='Sphere boundary')

ax3.legend(fontsize=9, labelcolor='white',
           facecolor='#1a1a2e', framealpha=0.7)
ax3.set_title(f'③ XY Slice at depth={pd0} + Sphere Boundary\nRed=inside sphere  White=outside',
              color='white', fontsize=11, pad=8)
ax3.set_xlabel('Column', color='#ffffff', fontsize=9)
ax3.set_ylabel('Row',    color='#ffffff', fontsize=9)
ax3.tick_params(colors='#ffffff', labelsize=8)

# ── Panel 4 : 3D octree coloured by inside/outside sphere ────
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
ax4.set_facecolor('#0d0d1a')

step   = max(1, len(octree_cells) // 500)
sample = octree_cells[::step]

inside_x, inside_y, inside_z   = [], [], []
outside_x, outside_y, outside_z = [], [], []

for (r, c, d, h, w, dp) in sample:
    # centre of this octree cell
    cx_cell = c + w/2
    cy_cell = r + h/2
    cz_cell = d + dp/2

    # check if cell centre is inside sphere (correct axis correspondence)
    cell_dist = np.sqrt((cx_cell - cx)**2 + (cy_cell - cy)**2 + (cz_cell - cz)**2)

    if cell_dist <= radius:
        inside_x.append(cx_cell)
        inside_y.append(cy_cell)
        inside_z.append(cz_cell)
    else:
        outside_x.append(cx_cell)
        outside_y.append(cy_cell)
        outside_z.append(cz_cell)

ax4.scatter(inside_x,  inside_y,  inside_z,
            c='red',   s=12, alpha=0.6,
            label=f'Inside sphere ({len(inside_x)})')
ax4.scatter(outside_x, outside_y, outside_z,
            c='cyan',  s=6,  alpha=0.3,
            label=f'Outside sphere ({len(outside_x)})')

# draw sphere wireframe
u = np.linspace(0, 2*np.pi, 30)
v = np.linspace(0, np.pi,   20)
sx = cx + radius * np.outer(np.cos(u), np.sin(v))
sy = cy + radius * np.outer(np.sin(u), np.sin(v))
sz = cz + radius * np.outer(np.ones(np.size(u)), np.cos(v))
ax4.plot_wireframe(sx, sy, sz,
                   color='yellow', alpha=0.15,
                   linewidth=0.5)

ax4.set_title('④ Octree Cells Inside vs Outside Sphere\nRed=inside  Cyan=outside',
              color='white', fontsize=11, pad=8)
ax4.set_xlabel('Col',   color='#ffffff', fontsize=9)
ax4.set_ylabel('Row',   color='#ffffff', fontsize=9)
ax4.set_zlabel('Depth', color='#ffffff', fontsize=9)
ax4.tick_params(colors='#ffffff', labelsize=8)
ax4.legend(fontsize=9, labelcolor='white',
           facecolor='#1a1a2e', framealpha=0.7)

# formatting
for ax in [ax1, ax2, ax4]:
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#333355')
    ax.yaxis.pane.set_edgecolor('#333355')
    ax.zaxis.pane.set_edgecolor('#333355')

for ax in [ax3]:
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')

# place suptitle and adjust margins so nothing is clipped
fig.suptitle(
    f"k-Wave Style Sphere Structures + 3D Octree  |  "
    f"{rows}×{cols}×{deps}  |  r={radius}  |  "
    f"Filled={filled_ball.sum()}  Surface={len(surface_points)}",
    color='white', fontsize=11, y=0.94
)
fig.subplots_adjust(top=0.88, left=0.06, right=0.98, hspace=0.36, wspace=0.28)

plt.show()
