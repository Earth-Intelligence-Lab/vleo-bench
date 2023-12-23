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

    r2 = np.corrcoef(df["Predicted Count"], df["True Count"].replace(-1, 0))[0, 1] ** 2

    if not ax:
        fig, ax = plt.subplots(figsize=(6, 6))

    sns.regplot(
        data=df,
        x="Predicted Count",
        y="True Count",
        scatter_kws=dict(color="k"),
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
    )
    ax.text(
        0.95, 0.95, f'$R^2 = {r2:.2f}$',
        verticalalignment='top', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=12
    )
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    return ax
