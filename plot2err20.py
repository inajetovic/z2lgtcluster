import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.gridspec as gridspec

# Load and combine CSVs
seeds = [1,2, 3, 4,5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16, 17,18, 19, 20]
df_list = []

for i in seeds:
    df = pd.read_csv(f"SPSA{i}.csv")
    df_list.append(df)

df = pd.concat(df_list, ignore_index=True)
#df = df[df["L"] <= 2]
print("Original shape:", df.shape)

# Define ground state energies
gs_energies = {
    1: -14.232797293022875,
    2: -24.001449903462607,
    3: -33.811996098495314
}

# Filter for valid P values with known E_GS
df = df[df["P"].isin(gs_energies.keys())]

# Compute R = |1 - E / E_GS|
df["R"] = df.apply(lambda row: abs(1 - row["E"] / gs_energies[row["P"]]), axis=1)

# Group and aggregate R instead of E
group_cols = [col for col in df.columns if col not in ['E', 'R']]
df = df.groupby(group_cols)['R'].agg(['mean', 'min', 'max', 'std']).reset_index()

# Set plotting styles
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})
sns.set_palette("colorblind")
palette = sns.color_palette("colorblind")

# Get unique Ansatz values in order
unique_ansatz = df["Ansatz"].unique()
ansatz_color_map = {ansatz: palette[i % len(palette)] for i, ansatz in enumerate(unique_ansatz)}

# Plot per unique P
unique_Ps = sorted(df["P"].unique())
n_cols = len(unique_Ps)
fig = plt.figure(figsize=(5 * n_cols, 6))
gs = gridspec.GridSpec(2, n_cols, height_ratios=[2, 5], figure=fig)

# Plot
for i, (P, subdata) in enumerate(df.groupby("P")):
    ax = fig.add_subplot(gs[1, i])

    # Seaborn scatter
    sns.scatterplot(
        data=subdata, x="S", y="mean",
        hue="Ansatz", style="L",
        palette=ansatz_color_map, ax=ax,
        s=200, linewidth=1.2
    )

    # Add error bars with correct color
    for _, row in subdata.iterrows():
        lower = row["mean"] - row["min"]
        upper = row["max"] - row["mean"]
        if not np.isnan(lower) and not np.isnan(upper):
            color = ansatz_color_map.get(row["Ansatz"], "gray")
            ax.errorbar(
                row["S"], row["mean"],
                yerr=[[lower], [upper]],
                fmt='none', ecolor=color,
                elinewidth=1.5, capsize=5
            )

    ax.set_xlabel(r"$N_{shots}$", fontsize=18)
    if i == 0:
        ax.set_ylabel(r"$R = \left|1 - \frac{E}{E_{\mathrm{GS}}}\right|$", fontsize=18)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis='y', labelleft=False)  # Hides tick labels only, keeps ticks/grid

    ax.set_title(fr"$ P = {P}, N = {2 + 6 * P - (P-1)}$", fontsize=18)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_xticks(sorted(subdata["S"].unique()))
    ax.grid(True)
    ax.legend_.remove()

# Global legend
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(labels), fontsize=18)
plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.savefig(f"{len(seeds)}SPSA_relative_error_plot.pdf")
plt.show()
