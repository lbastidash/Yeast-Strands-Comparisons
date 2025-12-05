import tifffile
import numpy as np
import matplotlib.pyplot as plt
import os

# Plotting style and colors
plt.rcParams.update({
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
    "savefig.facecolor": "#FFFFFF",
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "text.color": "#000000",
    "legend.edgecolor": "#000000",
    "grid.color": "#CCCCCC"
})

colors = ['#FF5A5F', '#007991', '#B79CED', '#FFD275']

# Group configuration
group_crops = {
    "y2": 8,
    "y4": 9,
    "y5": 7,
    "y7": 5
}

label_map = {
    "y2": "LH4409",
    "y4": "LH4591",
    "y5": "LH4592",
    "y7": "LH4832"
}

frametime = 30
treshold = 0.0001
frame_limit = 170

def analyze_ratio(filename, frametime=30, treshold=0.1, frame_limit=170):
    with tifffile.TiffFile(filename) as tif:
        stack = tif.asarray()
        num_frames = min(stack.shape[0], frame_limit)

    ratio = []
    ratio_error = []

    for frame_idx in range(num_frames):
        frame = stack[frame_idx]
        flattened = frame.ravel()
        n_pixels = flattened.size

        k = max(1, int(n_pixels * treshold))
        top_pixels = np.partition(flattened, -k)[-k:]
        top_mean = np.mean(top_pixels)
        top_std = np.std(top_pixels)

        k_bottom = max(1, int(n_pixels * (1 - treshold)))
        bottom_pixels = np.partition(flattened, k_bottom)[:k_bottom]
        bottom_mean = 100
        bottom_std = 40

        r = top_mean / bottom_mean if bottom_mean != 0 else 0
        err = r * np.sqrt((top_std / top_mean)**2 + (bottom_std / bottom_mean)**2) if bottom_mean != 0 and top_mean != 0 else 0

        ratio.append(r)
        ratio_error.append(err)

    time = np.arange(0, num_frames * frametime, frametime) / 1000
    return time, np.array(ratio), np.array(ratio_error)

# Data containers
time_shared = None
group_ratios = {}
group_errors = {}

# Plot 1: Individual groups
fig1, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
axs = axs.ravel()

for idx, y in enumerate(group_crops):
    max_crop = group_crops[y]
    base_dir = f"2025_02_19_{y}_80mW30ms/Crops{y.upper()}"
    all_ratios = []
    all_errors = []

    for c in range(1, max_crop + 1):
        filename = os.path.join(base_dir, f"Crop{c}.ome")
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            continue

        time, ratio, error = analyze_ratio(filename, frametime, treshold, frame_limit)
        if time_shared is None:
            time_shared = time
        all_ratios.append(ratio)
        all_errors.append(error)

        axs[idx].errorbar(
            time, ratio, yerr=error, fmt='-', marker='o', markersize=3,
            linewidth=1.2, elinewidth=0.5, capsize=0, alpha=0.7, label=f'Crop{c}'
        )

    all_ratios = np.array(all_ratios)
    all_errors = np.array(all_errors)
    avg_ratio = np.mean(all_ratios, axis=0)
    avg_error = np.sqrt(np.sum(all_errors**2, axis=0)) / all_errors.shape[0]

    group_ratios[y] = avg_ratio
    group_errors[y] = avg_error

    axs[idx].set_title(label_map[y], fontsize=12)
    axs[idx].set_ylabel("Contrast")
    axs[idx].grid(True, linestyle='--', linewidth=0.5)
    axs[idx].legend(fontsize=7)

axs[2].set_xlabel("Time [s]")
axs[3].set_xlabel("Time [s]")
plt.suptitle("Contrast Across Time", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig1.savefig("crop_group_ratios.pdf", format="pdf")

# Plot 2: Average ratios per group
fig2, ax2 = plt.subplots(figsize=(10, 5))
for idx, y in enumerate(group_crops):
    ax2.errorbar(
        time_shared, group_ratios[y], yerr=group_errors[y], fmt='-', marker='o',
        color=colors[idx], markersize=4, linewidth=1.8, elinewidth=0.6,
        capsize=0, label=label_map[y]
    )

ax2.set_title("Averaged Contrast each yeast strain", fontsize=14)
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Contrast")
ax2.grid(True, linestyle='--', linewidth=0.5)
ax2.legend()
plt.tight_layout()
fig2.savefig("crop_group_averages.pdf", format="pdf")
plt.show()
