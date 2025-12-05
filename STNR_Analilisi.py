import tifffile
import numpy as np
import matplotlib.pyplot as plt
import os

# Set white background and custom styling
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

# Color palette
colors = ['#FF5A5F', '#007991', '#B79CED', '#FFD275']

def analyze_bright_pixels(filename, frametime=30, treshold=0.005):
    with tifffile.TiffFile(filename) as tif:
        stack = tif.asarray()
        num_frames = stack.shape[0]

    top_avgs, bottom_avgs = [], []
    top_stds, bottom_stds = [], []

    for frame_idx in range(num_frames):
        frame = stack[frame_idx]
        flattened = frame.ravel()
        n_pixels = flattened.size

        k_top = max(1, int(n_pixels * treshold))
        top_pixels = np.partition(flattened, -k_top)[-k_top:]
        top_avgs.append(np.mean(top_pixels))
        top_stds.append(np.std(top_pixels))

        k_bottom = max(1, int(n_pixels * (1 - treshold)))
        bottom_pixels = np.partition(flattened, k_bottom)[:k_bottom]
        bottom_avgs.append(np.mean(bottom_pixels))
        bottom_stds.append(np.std(bottom_pixels))

    time = np.arange(0, len(top_avgs) * frametime, frametime) / 1000
    return (
        time[:170],
        np.array(top_avgs[:170]),
        np.array(top_stds[:170]),
        np.array(bottom_avgs[:170]),
        np.array(bottom_stds[:170])
    )

frametime = 30
treshold = 0.0001
frame_limit = 170

base_root = "2025_02_19_y{}_80mW30ms"
base_name = "FluorecenseYeastFullSensor80mW_"
file_suffix = "_MMStack_Pos0.ome.tif"

# Label mapping
label_map = {
    "y2": "LH4409",
    "y4": "LH4591",
    "y5": "LH4592",
    "y7": "LH4832"
}

y_labels = ["y2", "y4", "y5", "y7"]
w_ranges = {
    "y2": [1, 2, 3],
    "y4": [1, 2, 3],
    "y5": [1, 2],
    "y7": [1, 2, 3],
}

group_ratio_avg = {}
group_ratio_err = {}
time_shared = None

# Figure 1: Subplots for each group
fig1, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
axs = axs.ravel()

for idx, y_label in enumerate(y_labels):
    ax = axs[idx]
    base_dir = base_root.format(y_label[1])
    ratio_all, ratio_errs_all = [], []

    for i, w in enumerate(w_ranges[y_label]):
        subfolder = f"{base_name}{w}"
        filepath = os.path.join(base_dir, subfolder, f"{subfolder}{file_suffix}")

        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue

        time, top_vals, top_errs, bottom_vals, bottom_errs = analyze_bright_pixels(filepath, frametime, treshold)
        if time_shared is None:
            time_shared = time

        ratio = top_vals / bottom_vals
        ratio_err = ratio * np.sqrt((top_errs / top_vals)**2 )

        ratio_all.append(ratio)
        ratio_errs_all.append(ratio_err)

        color = colors[i % len(colors)]

        ax.errorbar(
            time, ratio, yerr=ratio_err, fmt='-', marker='^',
            color=color, markersize=4, linewidth=1.5, elinewidth=0.5, capsize=0,
            label=f"Group {w} Ratio"
        )

    ratio_all = np.array(ratio_all)
    ratio_errs_all = np.array(ratio_errs_all)

    mean_ratio = np.mean(ratio_all, axis=0)
    mean_err = np.sqrt(np.sum(ratio_errs_all**2, axis=0)) / ratio_all.shape[0]

    group_ratio_avg[y_label] = mean_ratio
    group_ratio_err[y_label] = mean_err

    ax.set_title(f"{label_map[y_label]}", fontsize=12)
    ax.set_ylabel("Contrast")
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=8)

axs[2].set_xlabel("Time [s]")
axs[3].set_xlabel("Time [s]")
plt.tight_layout()
plt.suptitle("Contrast Across Time", fontsize=16, color='#000000')
plt.subplots_adjust(top=0.9)
fig1.savefig("figure_group_subplots.pdf", format="pdf", facecolor=fig1.get_facecolor())

# Figure 2: Averaged ratios across groups
fig2, ax2 = plt.subplots(figsize=(10, 5))
for i, y_label in enumerate(y_labels):
    color = colors[i % len(colors)]
    ax2.errorbar(
        time_shared, group_ratio_avg[y_label], yerr=group_ratio_err[y_label],
        fmt='-', marker='^', color=color, markersize=4, linewidth=2, elinewidth=0.5, capsize=0,
        label=f"{label_map[y_label]} Ratio"
    )

ax2.set_title("Averaged Contrast Across Time", fontsize=13)
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Contrast")
ax2.grid(True, linestyle='--', linewidth=0.5)
ax2.legend(fontsize=9)

plt.tight_layout()
fig2.savefig("figure_group_averages.pdf", format="pdf", facecolor=fig2.get_facecolor())
plt.show()