import re

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error


def parse_digit_response(x: str):
    replace_dict = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                    "eight": 8, "nine": 9, "ten": 10}

    x = str(x).strip().lower()
    for old, new in replace_dict.items():
        x = x.replace(old, str(new))

    try:
        ret = int(re.search(r'\d+', x).group())
    except ValueError:
        ret = -1
    except AttributeError:
        ret = -1
    return ret


def calculate_counting_metrics(result_json, result_json_no_refusal):
    rr = (result_json["parsed_response"] == -1).mean()

    mape = mean_absolute_percentage_error(
        y_true=result_json["count"],
        y_pred=result_json["parsed_response"].replace(-1, 0)
    )
    mape_no_refusal = mean_absolute_percentage_error(
        y_true=result_json_no_refusal["count"],
        y_pred=result_json_no_refusal["parsed_response"]
    )

    r2 = np.corrcoef(result_json["count"], result_json["parsed_response"].replace(-1, 0))[0, 1] ** 2
    r2_no_refusal = np.corrcoef(
        result_json_no_refusal["count"], result_json_no_refusal["parsed_response"].replace(-1, 0)
    )[0, 1] ** 2

    return rr, (mape, mape_no_refusal), (r2, r2_no_refusal)


def plot_scatter(df, ax=None):
    import seaborn as sns

    df["Predicted Count"] = df["parsed_response"]
    df["True Count"] = df["count"]

    r2 = np.corrcoef(df["Predicted Count"], df["True Count"].replace(-1, 0))[0, 1] ** 2

    max_limit = max(df["parsed_response"].max(), df["count"].max())
    cmap = sns.color_palette("blend:#397CB8,#CFE2F3", as_cmap=True)

    if not ax:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_xlim(0, max_limit)
    ax.set_ylim(0, max_limit)

    sns.regplot(
        data=df,
        x="Predicted Count",
        y="True Count",
        ci=None,
        robust=True,
        truncate=False,
        scatter_kws=dict(marker="x", color="k"),
        line_kws=dict(color="red", linestyle="--"),
        ax=ax
    )
    sns.kdeplot(
        data=df,
        x="Predicted Count",
        y="True Count",
        levels=5,
        fill=True,
        alpha=0.6,
        cut=2,
        ax=ax,
        cmap=cmap
    )
    ax.text(
        0.95, 0.95, f'$R^2 = {r2:.2f}$',
        verticalalignment='top', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=24
    )
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return ax
