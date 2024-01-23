import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

scores = {
    # accuracy
    "landmarks": {
        "GPT-4V": 0.671,
        "IB-T5-xxl": 0.400,
        "IB-Vicuna-13b": 0.301,
        "LLaVA-v1.5": 0.296,
        "Qwen-VL-Chat": 0.292
    },
    # RefCLIPScore
    "RSICD": {
        "GPT-4V": 0.754,
        "IB-T5-xxl": 0.776,
        "IB-Vicuna-13b": 0.787,
        "LLaVA-v1.5": 0.773,
        "Qwen-VL-Chat": 0.765
    },
    # F-1
    "PatternNet": {
        "GPT-4V": 0.71,
        "IB-T5-xxl": 0.66,
        "IB-Vicuna-13b": 0.60,
        "LLaVA-v1.5": 0.58,
        "Qwen-VL-Chat": 0.4
    },
    # mean IoU
    "DIOR-RSVG": {
        "GPT-4V": 0.158,
        # "IB-T5-xxl": np.nan,
        # "IB-Vicuna-13b": np.nan,
        "LLaVA-v1.5": 0.0,
        "Qwen-VL-Chat": 0.007
    },
    # R2
    "NEONTreeEvaluation": {
        "GPT-4V": 0.250,
        "IB-T5-xxl": 0.093,
        "IB-Vicuna-13b": 0.0,
        "LLaVA-v1.5": 0.353,
        "Qwen-VL-Chat": 0.0
    },
    "xView2": {
        "GPT-4V": (0.108 + 0.062 + 0.055 + 0.106) / 4,
        # "IB-T5-xxl": np.nan,
        # "IB-Vicuna-13b": np.nan,
        # "LLaVA-v1.5": np.nan,
        "Qwen-VL-Chat": 0.0
    },
}


metrics = {
    "landmarks": "Accuracy",
    "RSICD": "RefCLIPScore",
    # F-1
    "PatternNet": "F-1",
    # mean IoU
    "DIOR-RSVG": "mean IoU",
    # R2
    "NEONTreeEvaluation": "$R^2$",
    "xView2": "$R^2$",
}

scenarios = ["landmarks", "RSICD", "PatternNet", "DIOR-RSVG", "NEONTreeEvaluation", "xView2"]

if __name__ == "__main__":
    sns.set(font_scale=2, style="ticks")
    cmap = sns.color_palette("blend:#397CB8,#CFE2F3")

    models = sorted(scores["landmarks"].keys())

    fig, axes = plt.subplots(nrows=6, figsize=(6, 18))

    for ax, scenario_name in zip(axes, scenarios):
        print(scenario_name)
        scenario_scores = scores[scenario_name]

        xs = [scenario_scores[key] for key in models if key in scenario_scores]
        ys = [x for x in models if x in scenario_scores]
        print(xs)
        sns.barplot(x=xs, y=ys, palette=cmap, ax=ax)
        # plt.xticks(rotation=45)
        ax.set_xlim([0, 1])
        ax.set_title(metrics[scenario_name], fontsize="large", pad=10)

        # Remove background
        ax.set_facecolor('none')

        # Remove y-axis tick marks but keep the labels
        ax.yaxis.set_tick_params(which='both', length=0)

        # Remove axis spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    plt.tight_layout()
    plt.savefig("./data/teaser-results.png", dpi=500)
