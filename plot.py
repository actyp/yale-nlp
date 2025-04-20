import matplotlib.pyplot as plt
import numpy as np
import json
import os

evaluation_path = "evaluation"
plot_path = "plots"

benches = [
    "sample_once",
    "sample_vote",
    "sample_eval",
    "sample_veco",
]

short_benches = {
    "sample_once": "Once",
    "sample_vote": "Vote",
    "sample_eval": "Eval",
    "sample_veco": "Veco",
}


def plot_bar(
    group_by,
    data,
    ylabel,
    title,
    legend_title,
    plot_file,
    gather_value="",
    stacked_keys=None,
    colors=None,
):

    models = sorted(data[benches[0]].keys())

    if group_by == "benchmarks":
        group_names = benches
        per_group_names = models
        data_per_group = data
        short_names = models
    elif group_by == "models":
        group_names = models
        per_group_names = benches
        data_per_group = {
            model: {bench: data[bench][model] for bench in benches} for model in models
        }
        short_names = short_benches

    # sort benchmarks and models for consistent ordering
    n_groups = len(group_names)
    n_per_group = len(per_group_names)

    # indices of the groups on the x‐axis
    idx = np.arange(n_groups)
    # width of each bar
    bar_w = 0.8 / n_per_group

    fig, ax = plt.subplots(figsize=(8, 5))

    # if stacked_keys is provided, we do a stacked bar per “per_group_val”
    if stacked_keys:

        for i, per_group_val in enumerate(per_group_names):
            bottoms = np.zeros(n_groups)
            xpos = idx + i * bar_w

            for key in stacked_keys:
                vals = [
                    data_per_group[grp].get(per_group_val, {}).get(key, 0)
                    for grp in group_names
                ]
                clr = colors.get(key) if colors else None
                ax.bar(
                    xpos,
                    vals,
                    bar_w,
                    bottom=bottoms,
                    label=key if i == 0 else None,
                    color=clr,
                    edgecolor="black",
                )
                bottoms += np.array(vals)

            # write the per_group_val under each individual bar
            for j, x in enumerate(xpos):
                ax.text(
                    x,
                    -0.01,
                    short_names[per_group_val],
                    transform=ax.get_xaxis_transform(),
                    rotation=45,
                    ha="center",
                    va="top",
                    fontsize="small",
                )

        ax.tick_params(axis="x", which="major", pad=20)
        plt.subplots_adjust(bottom=0.35)

    else:
        # original single‐value bars
        for i, per_group_val in enumerate(per_group_names):
            vals = [
                data_per_group[group].get(per_group_val, {}).get(gather_value, 0.0)
                for group in group_names
            ]
            ax.bar(
                idx + i * bar_w,
                vals,
                bar_w,
                label=per_group_val,
                edgecolor="black",
            )

    # labeling
    ax.set_xticks(idx + bar_w * (n_per_group - 1) / 2)
    ax.set_xticklabels(group_names, rotation=20)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(plot_file)


def get_data_eval():
    file_ext = "eval_results.json"
    # find files with file_ext in evaluation_path
    files = []
    for file in os.listdir(evaluation_path):
        if file.endswith(file_ext):
            files.append(file)

    data_eval = {}

    for bench in benches:
        data_eval[bench] = {}

    # read json files and get status
    for file in files:
        model_name = file.split("--")[0]

        bench = None
        for benchmark in benches:
            if benchmark in file:
                bench = benchmark
                break
        if bench is None:
            print(f"Unknown benchmark: {file}")
            continue

        with open(os.path.join(evaluation_path, file), "r") as f:
            data = json.load(f)

        pass_count = 0
        fail_count = 0

        for task_id, res in data["eval"].items():
            if res[0]["status"] == "pass":
                pass_count += 1
            else:
                fail_count += 1

        total_bench = pass_count + fail_count
        data_eval[bench][model_name] = {
            "pass": pass_count / total_bench,
            "fail": fail_count / total_bench,
            "total": total_bench,
        }
    return data_eval


def get_usage_data():
    file_ext = "usages.json"
    # find files with file_ext in evaluation_path
    files = []
    for file in os.listdir(evaluation_path):
        if file.endswith(file_ext):
            files.append(file)

    data_usage = {}
    for bench in benches:
        data_usage[bench] = {}

    # read json files and get status
    for file in files:
        model_name = file.split("--")[0]

        bench = None
        for benchmark in benches:
            if benchmark in file:
                bench = benchmark
                break
        if bench is None:
            print(f"Unknown benchmark: {file}")
            continue

        with open(os.path.join(evaluation_path, file), "r") as f:
            data = json.load(f)

        breakdown = data["uncached"]["breakdown"]
        model_key = list(breakdown.keys())[0]
        break_data = breakdown[model_key]
        retry_stats = break_data["retry_stats"]

        if "TemporaryLMError" in retry_stats["errors"]:
            errors = retry_stats["errors"]["TemporaryLMError"]
        else:
            errors = 0

        data_usage[bench][model_name] = {
            "prompt_tokens": break_data["prompt_tokens"],
            "completion_tokens": break_data["completion_tokens"],
            "total_tokens": break_data["total_tokens"],
            "num_requests": break_data["num_requests"],
            "num_occurences": retry_stats["num_occurences"],
            "total_wait_interval": retry_stats["total_wait_interval"],
            "total_call_interval": retry_stats["total_call_interval"],
            "errors": errors,
        }

    return data_usage


if __name__ == "__main__":
    data_eval = get_data_eval()
    data_usage = get_usage_data()
    models = sorted(data_eval[benches[0]].keys())

    # Plot Pass@k rates by benchmark
    plot_bar(
        group_by="benchmarks",
        data=data_eval,
        gather_value="pass",
        ylabel="Success Rate",
        title="Pass@k Rates by Benchmark",
        legend_title="Model",
        plot_file=os.path.join(plot_path, "benches_pass.png"),
    )

    plot_bar(
        group_by="models",
        data=data_eval,
        gather_value="pass",
        ylabel="Success Rate",
        title="Pass@k Rates by Model",
        legend_title="Benchmark",
        plot_file=os.path.join(plot_path, "models_pass.png"),
    )

    plot_bar(
        group_by="models",
        data=data_usage,
        gather_value="num_occurences",
        ylabel="Retries",
        title="Number of Retries per Model/Benchmark",
        legend_title="Benchmark",
        plot_file=os.path.join(plot_path, "models_occur.png"),
    )

    plot_bar(
        group_by="models",
        data=data_usage,
        ylabel="Tokens",
        title="Prompt vs Completion Token Usage per Model",
        legend_title="Token Type",
        plot_file=os.path.join(plot_path, "models_tokens.png"),
        stacked_keys=["prompt_tokens", "completion_tokens"],
        colors={
            "prompt_tokens": "skyblue",
            "completion_tokens": "orange",
        },
    )
