# coding: utf-8

from src.datasets.neon_trees import evaluation as neon_eval
from src.datasets.cowc import evaluation as cowc_eval
from src.datasets.aerial_animal import evaluation as animal_eval
from src.datasets.xview2 import evaluation as xview_eval
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=2)

fig, axes = plt.subplots(ncols=4, nrows=1, figsize=(24, 6))
neon_eval("./data/NeonTreeEvaluation/gpt-4v-counting.jsonl", axes[0])
cowc_eval("./data/cowc-m/gpt-4v-combined.jsonl", axes[1])
animal_eval("./data/aerial-animal-population-4tu/gpt-4v-counting.jsonl", axes[2])
xview_eval("./data/xView2/gpt4-v-test.jsonl", axes[3])
axes[0].set_title("Neon Tree", fontsize="large")
axes[1].set_title("COWC Vehicle", fontsize="large")
axes[2].set_title("Aerial Animal", fontsize="large")
axes[3].set_title("xBD Building", fontsize="large")
for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    common_limit = max(xmax, ymax)
    ax.set_xlim([0, common_limit])
    ax.set_ylim([0, common_limit])

plt.tight_layout()
plt.savefig("./data/NeonTreeEvaluation/counting-scatter-comparison-4.pdf")
