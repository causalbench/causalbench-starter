"""
Copyright (C) 2021  Patrick Schwab, GlaxoSmithKline plc
Copyright (C) 2023  Mathieu Chevalley, GlaxoSmithKline plc
"""
import os
import json
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns


method_display_name_map = {
    "ges": "GES",
    "gies": "GIES",
    "random100": "Random (k=100)",
    "random1000": "Random (k=1000)",
    "random10000": "Random (k=10000)",
    "pc": "PC",
    "notears-mlp-sparse": "NOTEARS (MLP,L1)",
    "notears-lin-sparse": "NOTEARS (Linear,L1)",
    "notears-mlp": "NOTEARS (MLP)",
    "notears-lin": "NOTEARS (Linear)",
    "DCDI-DSF": "DCDI-DSF",
    "DCDI-G": "DCDI-G",
    "grnboost": "GRNBoost",
    "custom": "Custom",
}

sweep_title_display_map_int = {
    "partial_intervention_sweep_rpe1": "RPE1 (interventional - varying intervention set)"
}


def json2df_sweep(json_obj) -> pd.DataFrame:
    rows = []
    for method_name in json_obj.keys():
        for step in sorted(map(float, json_obj[method_name].keys())):
            prefix = (
                method_display_name_map[method_name]
                if method_name in method_display_name_map
                else method_name,
                step,
                "mean_wasserstein_distance",
            )
            data_points = json_obj[method_name][str(step)]
            for i, data_point in enumerate(data_points):
                rows.append(prefix + (i, data_point))

    df = pd.DataFrame(data=rows, columns=["method", "step", "metric", "index", "value"])
    return df


def plot_sweeps(df: pd.DataFrame, file_path: str, title_display_map: dict):
    # Computing area under curve
    def integrate(x, y):
        area = np.trapz(y=y, x=x)
        return area

    fig = plt.figure(constrained_layout=True, figsize=(10, 4))
    ax = fig.subplots(1, 1, sharex=True, sharey=False)
    palette = sns.color_palette("pastel")
    markers = ["8", "s", "p", "P", "*", "X", "D"]
    marker_sizes = [8, 8, 10, 10, 17, 14, 8]
    all_handles = []
    methods = sorted(set(df["method"]))
    auc_scores = dict()
    improvement_scores = dict()
    for j, (method, marker, marker_size) in \
            enumerate(zip(methods, markers, marker_sizes)):
        scatter = ax.scatter(
            df[
                (df["metric"] == "mean_wasserstein_distance") & (df["method"] == method)
            ]["step"]
            * 100,
            df[
                (df["metric"] == "mean_wasserstein_distance") & (df["method"] == method)
            ]["value"],
            color=palette[j],
            s=marker_size,
            marker=marker,
            edgecolors="black",
            linewidths=0.2,
            zorder=1,
        )
        subset = df[
            (df["metric"] == "mean_wasserstein_distance") & (df["method"] == method)
        ][["index", "step", "value"]]
        subset = subset.groupby("step").median().reset_index()
        auc = integrate(subset["step"], subset["value"])
        auc_scores[method] = auc
        if 1.00 in subset["step"].values and 0.25 in subset["step"].values:
            improvement_scores[method] = subset["value"][subset["step"] == 1.00].values[0] - subset["value"][subset["step"] == 0.25].values[0]
        else:
            improvement_scores[method] = -1
        ax.plot(
            subset["step"] * 100,
            subset["value"],
            color=palette[j],
            zorder=1,
            alpha=0.8,
            linewidth=1.0,
        )
        all_handles += [scatter]
    ax.grid(linestyle="--", color="grey", linewidth=0.25)
    ax.legend(all_handles, methods)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim([ymin * 0.9, ymax * 1.1])
    ax.set_xlim([0.0, 101])
    ax.set_ylabel("Mean Wasserstein\nDistance [unitless]")
    ax.set_xlabel("Fraction of Intervention Set [%]")

    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    ax.text(
        0.5,
        0.95,
        "RPE1 (interventional - varying intervention set)",
        horizontalalignment="center",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8, pad=2, edgecolor="white"),
    )
    plt.savefig(file_path)
    plt.clf()

    return auc_scores, improvement_scores


def generate_plots_int_sweeps(plot_dir: str, data_dir: str):
    files = [
        "partial_intervention_sweep_rpe1.json",
    ]
    all_dfs = []
    for file_name in files:
        with open(os.path.join(data_dir, file_name), "r") as fp:
            json_obj = json.load(fp)
        df = json2df_sweep(json_obj)
        if len(df) > 0:
            base_name = file_name[:-5]
            df["evidence"] = sweep_title_display_map_int[base_name]
            all_dfs.append(df)
    df = pd.concat(all_dfs)
    print(df)
    outfile_path = os.path.join(plot_dir, f"sweeps_partial_interventions.pdf")
    auc_scores, improvement_scores = plot_sweeps(df, outfile_path, sweep_title_display_map_int)
    scores = {"auc_scores": auc_scores, "improvements_scores": improvement_scores}

    with open(os.path.join(plot_dir, "scores.json"), "w") as output:
        json.dump(scores, output)


def generate_plots(plot_dir: str, data_dir: str):
    generate_plots_int_sweeps(plot_dir, data_dir)


def extract_metrics(data_dir: str, dataset_name: str, filename: str):
    outputs = dict()
    conf = set()
    for d in os.listdir(data_dir):
        arguments_file = os.path.join(data_dir, d, "arguments.json")
        metrics_file = os.path.join(data_dir, d, "metrics.json")
        if not os.path.exists(arguments_file) or not os.path.exists(metrics_file):
            continue

        with open(arguments_file) as f:
            data = json.load(f)
            seed = data["partial_intervention_seed"]
            if data["model_name"] == "custom":
                model_name = "Custom " + data["inference_function_file_path"]
            else:
                model_name = data["model_name"]
            tup = (
                model_name,
                data["dataset_name"],
                data["fraction_partial_intervention"],
                seed,
            )
            if (
                (data["training_regime"] == "PartialIntervational")
                and data["dataset_name"] == dataset_name
                and tup not in conf
            ):
                conf.add(tup)
                with open(metrics_file) as f2:
                    metrics = json.load(f2)
                    outputs.setdefault(model_name, dict()).setdefault(
                        data["fraction_partial_intervention"], []
                    ).append(
                        metrics["quantitative_test_evaluation"]["output_graph"][
                            "wasserstein_distance"
                        ]["mean"]
                    )
    with open(os.path.join(data_dir, filename + ".json"), "w") as output:
        json.dump(outputs, output)


if __name__ == "__main__":
    plots_dir, data_dir = sys.argv[1], sys.argv[2]
    extract_metrics(data_dir, "weissmann_rpe1", "partial_intervention_sweep_rpe1")
    generate_plots(plots_dir, data_dir)
