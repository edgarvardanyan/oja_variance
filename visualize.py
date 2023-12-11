from oja import load_from_json
import matplotlib.pyplot as plt
import os

corrs = [-0.3, -0.7, 0.3, 0.7]

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

plot_ids = [(0, 0, -0.3), (0, 1, -0.7), (1, 0, 0.3), (1, 1, 0.7)]

for coord0, coord1, corr in plot_ids:

    json_path = os.path.join(os.path.dirname(__file__), f"variances_({str(corr).replace('.', '_')}).json")
    alphas, variances = load_from_json(json_path)  home/edgar/phd/oja_python/variances.png
    axs[coord0, coord1].set_yscale("log")
    axs[coord0, coord1].set_xscale("log")
    axs[coord0, coord1].scatter(x=alphas, y=variances[:, 0], label=f"V")

    # Adding the straight line
    y_line = [alpha * (1 - corr**2) / (8 * abs(corr)) for alpha in alphas]
    axs[coord0, coord1].plot(alphas, y_line, color='red', linestyle='--', label=f'y=α(1-ρ²)/(8|ρ|))')

    axs[coord0, coord1].legend()
    axs[coord0, coord1].set_title(f"Correlation = {corr}")
    axs[coord0, coord1].set_xlabel("Learning rate")
    axs[coord0, coord1].set_ylabel("Variance")

plt.subplots_adjust(hspace=0.3, wspace=0.2)

png_path = os.path.join(os.path.dirname(__file__), "variance_plots.png")

plt.savefig(png_path)