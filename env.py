from data import *
M = 10
C = 8  # Attention sum starts from M-P (e.g., M-2)
ratio = 0.5 # ratio of cap/(cap+nuc)
T = int(C*ratio)  # Number of cap token
P = 4
num_sections, num_tokens_per_section, d_model = 4, 5, 64

import matplotlib.pyplot as plt

keys = prompt.keys()
fig, axes = plt.subplots(2, int(len(keys)/2), figsize=(20, 7))
table = {}
plots = {}
axes = axes.flatten()
for i, key in enumerate(keys):
    table[key] = False
    ax = axes[i]
    # 첫 번째 플롯
    ax.set_title(f"Prompt {key}")
    ax.set_xlabel("Total Token Length")
    ax.set_ylabel("Hit Ratio", fontsize=12, labelpad=1)
    ax.legend()
    ax.grid(True)

    ax.axvline(x=M, color="gray", linestyle="--", linewidth=1.5, label=f"M = {M}")
    plots[key] = ax
offset = 0.3 