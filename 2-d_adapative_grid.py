import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

def get_positive_int(prompt):
    while True:
        try:
            value = int(input(prompt))
            if value > 0:
                return value
            print("   ✗ Please enter a positive integer.")
        except ValueError:
            print("   ✗ Invalid input. Please enter a whole number.")

print("\n=== Adaptive Grid — 3 Distinct Blobs ===\n")
rows = get_positive_int("Enter number of rows    : ")
cols = get_positive_int("Enter number of columns : ")

y_idx, x_idx = np.ogrid[:rows, :cols]

sigma = max(min(rows, cols) * 0.00000000001, 4)  # Blob size control(relative to grid size)

blobs = [
    (rows * 0.20, cols * 0.20, sigma, 0.90),
    (rows * 0.78, cols * 0.25, sigma, 0.80),
    (rows * 0.48, cols * 0.78, sigma, 0.60),
]

matrix = np.zeros((rows, cols))
for (cr, cc, sg, amp) in blobs:
    matrix += amp * np.exp(
        -((x_idx - cc)**2 + (y_idx - cr)**2) / (2 * sg**2)
    )
matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min() + 1e-10)

# Peak detection
temp_mat        = matrix.copy()
peaks           = []
suppress_radius = max(rows, cols) // 5

for _ in range(3):
    idx    = np.argmax(temp_mat)
    pr, pc = np.unravel_index(idx, temp_mat.shape)
    peaks.append((pr, pc, matrix[pr, pc]))
    r0, r1 = max(0, pr - suppress_radius), min(rows, pr + suppress_radius)
    c0, c1 = max(0, pc - suppress_radius), min(cols, pc + suppress_radius)
    temp_mat[r0:r1, c0:c1] = 0

print("\n  --- Top-3 Peaks ---")
for i, (pr, pc, val) in enumerate(peaks):
    print(f"  Blob {i+1}: I={val:.4f}  row={pr}  col={pc}")

# Quadtree
THRESHOLD  = 0.04
MIN_SIZE   = 1
grid_cells = []

def adaptive_grid(mat, r, c, h, w, threshold, min_size):
    block     = mat[r:r+h, c:c+w]
    variation = block.max() - block.min()
    if variation <= threshold or h <= min_size or w <= min_size:
        grid_cells.append((r, c, h, w))
        return
    half_h = h // 2
    half_w = w // 2
    adaptive_grid(mat, r,          c,          half_h,     half_w,     threshold, min_size)
    adaptive_grid(mat, r,          c + half_w, half_h,     w - half_w, threshold, min_size)
    adaptive_grid(mat, r + half_h, c,          h - half_h, half_w,     threshold, min_size)
    adaptive_grid(mat, r + half_h, c + half_w, h - half_h, w - half_w, threshold, min_size)

adaptive_grid(matrix, 0, 0, rows, cols, THRESHOLD, MIN_SIZE)

cell_lookup = np.empty((rows, cols), dtype=object)
for (r, c, h, w) in grid_cells:
    cell_lookup[r:r+h, c:c+w] = f"{h}×{w}"

print(f"\n  Grid  : {rows}×{cols}   Cells : {len(grid_cells)}")
print(f"  Threshold={THRESHOLD}   MinSize={MIN_SIZE}\n")

# Plot
blob_peak_colors = ['#ff4422', '#33dd33', '#2255ff']
blob_labels      = ['Blob 1 — Red', 'Blob 2 — Green', 'Blob 3 — Blue']
peak_markers     = ['*', 'D', '^']

fig, ax = plt.subplots(figsize=(9, 9))
fig.patch.set_facecolor('#0d0d1a')
ax.set_facecolor('#0d0d1a')

# Raw heatmap — no mosaic
ax.imshow(matrix, cmap='turbo', origin='upper',
          vmin=0, vmax=1, interpolation='nearest')

# Subgrid lines
for (r, c, h, w) in grid_cells:
    ax.add_patch(Rectangle(
        (c - 0.5, r - 0.5), w, h,
        fill=False, edgecolor='white', linewidth=1.4, alpha=0.85
    ))

# Peak markers
for i, (pr, pc, val) in enumerate(peaks):
    ax.plot(pc, pr,
            marker          = peak_markers[i],
            color           = blob_peak_colors[i],
            markersize      = 14,
            markeredgecolor = 'white',
            markeredgewidth = 1.2,
            zorder          = 6,
            linestyle       = 'None')

cbar = plt.colorbar(
    plt.cm.ScalarMappable(cmap='turbo', norm=plt.Normalize(0, 1)),
    ax=ax, fraction=0.046, pad=0.04
)
cbar.set_label('Intensity', color='#cccccc')
cbar.ax.yaxis.set_tick_params(color='#cccccc')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#cccccc')

legend_handles = []
for i in range(3):
    legend_handles.append(
        plt.Line2D([0], [0],
                   marker=peak_markers[i],
                   color='w',
                   markerfacecolor=blob_peak_colors[i],
                   markeredgecolor='white',
                   markersize=10,
                   linestyle='None',
                   label=blob_labels[i] + f'   I={peaks[i][2]:.3f}')
    )
legend_handles.append(
    mpatches.Patch(facecolor='none', edgecolor='white',
                   linewidth=1.5, label='Subgrid cells')
)

ax.legend(handles=legend_handles, loc='upper right', fontsize=8.5,
          facecolor='#1a1a2e', framealpha=0.85,
          labelcolor='white', edgecolor='#555566')

ax.set_title(
    f"Adaptive Quadtree  {rows}×{cols}   "
    f"Threshold={THRESHOLD}   MinCell={MIN_SIZE}   Cells={len(grid_cells)}",
    color='white', fontsize=10, pad=10
)

max_ticks = min(rows, cols, 20)
x_step = max(1, cols // max_ticks)
y_step = max(1, rows // max_ticks)
ax.set_xticks(np.arange(0, cols, x_step))
ax.set_yticks(np.arange(0, rows, y_step))
ax.set_xticklabels(np.arange(0, cols, x_step), color='#aaaaaa', fontsize=8)
ax.set_yticklabels(np.arange(0, rows, y_step), color='#aaaaaa', fontsize=8)
ax.set_xlabel('Column', color='#aaaaaa')
ax.set_ylabel('Row',    color='#aaaaaa')
ax.tick_params(colors='#aaaaaa')
for spine in ax.spines.values():
    spine.set_edgecolor('#333355')

# Hover tooltip
annot = ax.annotate(
    '', xy=(0, 0), xytext=(14, 14),
    textcoords='offset points',
    bbox=dict(boxstyle='round,pad=0.5', fc='#111111', ec='white', alpha=0.88),
    color='white', fontsize=9, fontfamily='monospace'
)
annot.set_visible(False)

hline = ax.axhline(y=0, color='white', linewidth=0.7, alpha=0.4, linestyle='--')
vline = ax.axvline(x=0, color='white', linewidth=0.7, alpha=0.4, linestyle='--')
hline.set_visible(False)
vline.set_visible(False)

def on_hover(event):
    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        annot.set_visible(False)
        hline.set_visible(False)
        vline.set_visible(False)
        fig.canvas.draw_idle()
        return

    c_pos = int(round(event.xdata))
    r_pos = int(round(event.ydata))

    if 0 <= r_pos < rows and 0 <= c_pos < cols:
        intensity = matrix[r_pos, c_pos]
        cell_info = cell_lookup[r_pos, c_pos] or "—"
        dists     = [np.sqrt((r_pos-pr)**2 + (c_pos-pc)**2) for (pr,pc,_) in peaks]
        nearest   = int(np.argmin(dists))

        annot.xy = (event.xdata, event.ydata)
        annot.set_text(
            f" Row       : {r_pos}\n"
            f" Col       : {c_pos}\n"
            f" Intensity : {intensity:.4f}\n"
            f" Blob      : {blob_labels[nearest]}\n"
            f" Cell size : {cell_info}"
        )
        annot.set_visible(True)
        hline.set_ydata([r_pos])
        vline.set_xdata([c_pos])
        hline.set_visible(True)
        vline.set_visible(True)
    else:
        annot.set_visible(False)
        hline.set_visible(False)
        vline.set_visible(False)

    fig.canvas.draw_idle()

fig.canvas.mpl_connect('motion_notify_event', on_hover)

plt.tight_layout()
plt.show()