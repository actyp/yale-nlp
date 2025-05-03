import matplotlib.pyplot as plt
import numpy as np
import json
import os

evaluation_path = "evaluation"
plot_path = "plots"

# create plot_path if it doesn't exist
os.makedirs(plot_path, exist_ok=True)

methods = [
    "sample_once",
    "sample_vote",
    "sample_eval",
    "sample_veco",
]

short_methods = {
    "sample_once": "Once",
    "sample_vote": "Vote",
    "sample_eval": "Eval",
    "sample_veco": "Veco",
}

models = [
    "Artigenz-Coder-DS",
    "Nxcode-CQ",
    "CodeLlama",
    "Starcoder2",
    "Qwen2.5-Coder",
    "Deepseek-Coder",
]

model_params = {
    "Artigenz-Coder-DS": "6.7B",
    "Starcoder2": "15B",
    "CodeLlama": "13B",
    "Deepseek-Coder": "33B",
    "Nxcode-CQ": "7B",
    "Qwen2.5-Coder": "32B",
}

details_map = {
    True: {-1: "", 0: "", 1: "Verified solution immediately", 2: "", 3: "", 4: ""},
    False: {
        -1: "None analytical response found",
        0: "None verification analysis found",
        1: "None correction response found",
        2: "None verification analysis found",
        3: "",
        4: "",
    },
}


def plot_pies(data, plot_dir):
    # data: { model_name : {task_id : {"status": "pass"/"fail", "details":[(verify_bool,attempts),…]} } }

    for model in models:
        # First pie chart shows the pass/fail evaluation
        # percentage for each model (1 pie chart per model)
        # For Pass/Fail we also show if verification was successful
        # To be successful we just need 1 sample that was verified

        model_tasks = data[model]
        total = len(model_tasks)
        assert total == 1140

        # 1) aggregate counts
        pass_count = fail_count = 0
        pass_verify = pass_non = fail_verify = fail_non = 0

        for tid, stat_det in model_tasks.items():
            status, details = stat_det["status"], stat_det["details"]
            if status == "pass":
                pass_count += 1
                found_verified = False
                for verified, _ in details:
                    if verified:
                        pass_verify += 1
                        found_verified = True
                        # we need at least one verified solution
                        break
                if not found_verified:
                    pass_non += 1
            else:
                fail_count += 1
                found_verified = False
                for verified, _ in details:
                    if verified:
                        fail_verify += 1
                        found_verified = True
                        break
                if not found_verified:
                    fail_non += 1

        # 2) prepare sizes
        outer_sizes = [pass_count, fail_count]

        outer_sizes = [x / total for x in outer_sizes]
        inner_sizes = [pass_verify, pass_non, fail_verify, fail_non]

        inner_sizes = [x / total for x in inner_sizes]

        # 3) plot
        outer_colors = ["#4caf50", "#f44336"]  # green / red
        inner_colors = [
            "#6cce9e",  # two shades of green
            "#a8e6cf",
            "#f47c7d",  # two shades of red
            "#fcb5b5",
        ]

        fig, ax = plt.subplots(figsize=(5, 5))
        # outer ring (overall pass vs fail)
        # ax.pie(
        #     outer_sizes,
        #     radius=1.0,
        #     labels=["Pass", "Fail"],
        #     colors=outer_colors,
        #     startangle=90,
        #     labeldistance=0.85,
        #     wedgeprops=dict(width=0.3, edgecolor="white"),
        # )
        # inner ring (breakdown by verify/non‑verify)

        labels = [
            f"Pass Verified\n({inner_sizes[0]*100:.1f}%)",
            f"Pass Unverified\n({inner_sizes[1]*100:.1f}%)",
            f"Fail Verified\n({inner_sizes[2]*100:.1f}%)",
            f"Fail Unverified\n({inner_sizes[3]*100:.1f}%)",
        ]

        ax.pie(
            inner_sizes,
            radius=0.7,
            labels=labels,  # ["Pass ✓", "Pass ✗", "Fail ✓", "Fail ✗"]
            colors=inner_colors,
            startangle=90,
            labeldistance=1.1,
            # autopct=lambda pct: f"{pct:.1f}%",
            # pctdistance=1.1,
            wedgeprops=dict(width=0.3, edgecolor="white"),
        )
        ax.text(
            0,
            0,
            model,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

        ax.set(aspect="equal")
        # ax.set_title(f"Veco Evaluation", fontsize=14)

        fig.tight_layout()

        # save
        os.makedirs(plot_dir, exist_ok=True)
        fn = os.path.join(plot_dir, f"pie_{model}_eval.png")
        plt.tight_layout()
        plt.savefig(fn, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        # Second pie chart will gather (status, attempts) for every sample
        # in each task id for each model

        categories = {
            "UnV|-1": 0,
            "UnV|0": 0,
            "UnV|1": 0,
            "UnV|2": 0,
            "UnV|3": 0,
            "UnV|4": 0,
            "V|0": 0,
            "V|1": 0,
            "V|2": 0,
            "V|3": 0,
            "V|4": 0,
        }

        for tid, stat_det in model_tasks.items():
            status, details = stat_det["status"], stat_det["details"]
            # we don't care about evaluation status
            for sample in details:
                verified, attempts = sample
                if verified:
                    # verified
                    categories[f"V|{attempts}"] += 1
                else:
                    categories[f"UnV|{attempts}"] += 1

        # normalize for total
        total = sum(categories.values())
        sample_num = 8  # m = 8
        assert total == 1140 * sample_num
        for k in categories.keys():
            categories[k] = round(categories[k] / total, 2)

        # for colors we need 6 shades of red and 5 shades of green
        reds = ["#fcb5b5", "#f47c7d", "#e84343", "#d63031", "#c0392b", "#96281b"]
        greens = ["#a8e6cf", "#6cce9e", "#38b28b", "#2a9d8f", "#216e6f"]

        cat_colors = {
            "UnV|-1": reds[0],
            "UnV|0": reds[1],
            "UnV|1": reds[2],
            "UnV|2": reds[3],
            "UnV|3": reds[4],
            "UnV|4": reds[5],
            "V|0": greens[0],
            "V|1": greens[1],
            "V|2": greens[2],
            "V|3": greens[3],
            "V|4": greens[4],
        }

        # remove all keys for which value is 0
        categories = {k: v for k, v in categories.items() if v > 0.00}

        labels = []
        sizes = []

        for k, v in categories.items():
            lbl = f"{k}\n({v*100:.1f}%)"
            labels.append(lbl)
            sizes.append(v)

        colors = []
        for k in categories.keys():
            colors.append(cat_colors[k])

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(
            sizes,
            radius=0.7,
            labels=labels,
            colors=colors,
            startangle=90,
            labeldistance=1.1,
            wedgeprops=dict(width=0.3, edgecolor="white"),
            textprops={"fontsize": 8},
        )

        ax.text(
            0,
            0,  # data‐coords at center of pie
            model,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

        ax.set(aspect="equal")
        # ax.set_title(f"Verification Statistics", fontsize=14)

        fig.tight_layout(pad=0, w_pad=0, h_pad=0)

        # save
        os.makedirs(plot_dir, exist_ok=True)
        fn = os.path.join(plot_dir, f"pie_{model}_attempts.png")
        plt.tight_layout()
        plt.savefig(fn, bbox_inches="tight")
        plt.close(fig)


def plot_bar(
    group_by,
    data,
    ylabel,
    title,
    legend_title,
    plot_file,
    ylim=None,
    gather_value="",
    stacked_keys=None,
    colors=None,
):

    if group_by == "benchmarks":
        group_names = methods
        per_group_names = models
        data_per_group = data
        short_names = models
    elif group_by == "models":
        group_names = models
        per_group_names = methods
        data_per_group = {
            model: {bench: data[bench][model] for bench in methods} for model in models
        }
        short_names = short_methods

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
                    edgecolor="white",
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
                edgecolor="white",
            )

    # labeling
    ax.set_xticks(idx + bar_w * (n_per_group - 1) / 2)

    # Fix xlabels
    xlabels = group_names.copy()
    if group_by == "models":
        for lbl in xlabels:
            params = model_params.get(lbl, "")
            if params:
                xlabels[xlabels.index(lbl)] = f"{lbl}\n({params})"

    ax.set_xticklabels(xlabels, rotation=20, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title(title, fontsize=20)
    ax.legend(title=legend_title)
    if ylim:
        ax.set_ylim(0, ylim)

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

    for bench in methods:
        data_eval[bench] = {}

    # read json files and get status
    for file in files:
        model_name = file.split("--")[1].split("_")[0]

        for set_names in models:
            if set_names.lower() in model_name.lower():
                model_name = set_names
                break

        bench = None
        for benchmark in methods:
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
    for bench in methods:
        data_usage[bench] = {}

    # read json files and get status
    for file in files:
        model_name = file.split("--")[1].split("_")[0]

        for set_names in models:
            if set_names.lower() in model_name.lower():
                model_name = set_names
                break

        bench = None
        for benchmark in methods:
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

        num_requests = break_data["num_requests"]
        num_retries = retry_stats["num_occurences"]

        num_direct = num_requests - num_retries

        completion_tokens = break_data["completion_tokens"]
        direct_tokens = completion_tokens * num_direct / (num_requests**2)
        retry_tokens = completion_tokens * num_retries / (num_requests**2)

        if "TemporaryLMError" in retry_stats["errors"]:
            errors = retry_stats["errors"]["TemporaryLMError"]
        else:
            errors = 0

        data_usage[bench][model_name] = {
            "prompt_tokens": break_data["prompt_tokens"] / num_requests,
            "completion_tokens": break_data["completion_tokens"] / num_requests,
            "direct_tokens": direct_tokens,
            "retry_tokens": retry_tokens,
            "total_tokens": break_data["total_tokens"],
            "num_requests": num_requests,
            "num_occurences": num_retries,
            "total_wait_interval": retry_stats["total_wait_interval"],
            "total_call_interval": retry_stats["total_call_interval"],
            "errors": errors,
        }

    return data_usage


def get_task_id_details():
    file_ext = "sample_veco_m8_n4.jsonl"
    # find files with file_ext in evaluation_path
    files = []
    for file in os.listdir(evaluation_path):
        if file.endswith(file_ext):
            files.append(file)

    data_task = {}

    for file in files:
        model_name = file.split("--")[1].split("_")[0]

        for set_names in models:
            if set_names.lower() in model_name.lower():
                model_name = set_names
                break

        with open(os.path.join(evaluation_path, file), "r") as file:
            for line in file:
                try:
                    json_data = json.loads(line)

                    task_id = json_data["task_id"]
                    details = json_data["details"]
                    if model_name not in data_task:
                        data_task[model_name] = {}
                    if task_id not in data_task[model_name]:
                        data_task[model_name][task_id] = {}

                    data_task[model_name][task_id]["details"] = details

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line: {line.strip()} - {e}")

    # parse eval_results to get pass/fail for each task_id
    file_ext = "sample_veco_m8_n4-sanitized-calibrated_eval_results.json"
    files = []
    for file in os.listdir(evaluation_path):
        if file.endswith(file_ext):
            files.append(file)

    # read json files and get status
    for file in files:
        model_name = file.split("--")[1].split("_")[0]

        for set_names in models:
            if set_names.lower() in model_name.lower():
                model_name = set_names
                break

        bench = None
        for benchmark in methods:
            if benchmark in file:
                bench = benchmark
                break
        if bench is None:
            print(f"Unknown benchmark: {file}")
            continue

        with open(os.path.join(evaluation_path, file), "r") as f:
            data = json.load(f)

        for task_id, res in data["eval"].items():
            status = res[0]["status"]

            data_task[model_name][task_id]["status"] = status

    return data_task


if __name__ == "__main__":
    data_eval = get_data_eval()
    data_usage = get_usage_data()
    data_veco = get_task_id_details()

    # Plot Pass@k rates by benchmark
    plot_bar(
        group_by="benchmarks",
        data=data_eval,
        gather_value="pass",
        ylabel="Success Rate",
        title="Benchmark Success Rate",
        legend_title="Model",
        ylim=1,
        plot_file=os.path.join(plot_path, "methods_pass.png"),
    )

    plot_bar(
        group_by="models",
        data=data_eval,
        gather_value="pass",
        ylabel="Success Rate",
        title="Benchmark Success Rate",
        legend_title="Method",
        ylim=1,
        plot_file=os.path.join(plot_path, "models_pass.png"),
    )

    plot_bar(
        group_by="models",
        data=data_usage,
        gather_value="num_occurences",
        ylabel="Retry Attempts",
        title="Number of Retries",
        legend_title="Benchmark",
        plot_file=os.path.join(plot_path, "models_attempts.png"),
    )

    plot_bar(
        group_by="models",
        data=data_usage,
        ylabel="Tokens",
        title="Average Prompt vs Completion Token Usage",
        legend_title="Token Type",
        plot_file=os.path.join(plot_path, "models_tokens.png"),
        stacked_keys=["prompt_tokens", "direct_tokens", "retry_tokens"],
        colors={
            "prompt_tokens": "skyblue",
            "direct_tokens": "orange",
            "retry_tokens": "lightcoral",
        },
    )

    plot_pies(
        data_veco,
        plot_dir=plot_path,
    )
