from inference import sample_once, sample_vote, sample_eval, sample_veco
from local_models import SUPPORTED_MODELS
from data_parse import get_dataset
from typing import SupportsIndex
from threading import Lock
import pyglove as pg
import langfun as lf
import argparse
import json
import os


METHODS = [
    "sample_once",
    "sample_vote",
    "sample_eval",
    "sample_veco",
]


def process_row(
    row: SupportsIndex,
    lm: lf.LanguageModel,
    method: str,
    num_samples: int,
    num_retries: int,
    out_file: str,
    out_file_lock: Lock,
):
    funcargs = {
        "sample_once": (sample_once, [lm]),
        "sample_vote": (sample_vote, [lm, num_samples]),
        "sample_eval": (sample_eval, [lm, num_samples]),
        "sample_veco": (sample_veco, [lm, num_samples, num_retries]),
    }

    task_id = row["task_id"]
    complete_prompt = row["complete_prompt"]

    func, args = funcargs[method]
    dct = func(task_id, complete_prompt, *args)

    solution = dct["solution"]
    solution_source = solution.source if solution is not None else "None"

    save_dct = {
        "task_id": task_id,
        "solution": solution_source,
        "raw_solution": solution_source,
    }

    if "details" in dct:
        save_dct["details"] = dct["details"]

    with out_file_lock:
        with open(out_file, "a") as file:
            file.write(json.dumps(save_dct) + "\n")


def parser():
    parser = argparse.ArgumentParser(
        usage="%(prog)s [options]",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    available_models = [m.model_id for m in SUPPORTED_MODELS]
    parser.add_argument("--model_id", required=True,
                        help="The id of the model to use. "
                        f"Available: {available_models}")

    parser.add_argument("--num_samples", type=int, default=8,
                        help="The number of samples (m) to use.")

    parser.add_argument("--num_retries", type=int, default=4,
                        help="The number of retries (n) to use.")

    parser.add_argument("--methods", nargs='+',
                        default=METHODS, choices=METHODS,
                        help="list of methods to use for inference")

    parser.add_argument("--dataset_id", default="bigcode/bigcodebench",
                        help="The id of the dataset to use.")

    parser.add_argument("--dataset_split", default="v0.1.4",
                        help="The split of the dataset to use.")

    parser.add_argument("--out_dir", type=str, default="./evaluation",
                        help="The directory to store the results.")

    parser.add_argument("--max_workers", type=int, default=None,
                        help="Limit the number of concurrent workers")

    return parser


if __name__ == "__main__":
    args = parser().parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    dataset = get_dataset(args.dataset_id, args.dataset_split)

    model_id = args.model_id
    model_name = model_id.replace("/", "--")

    num_samples = args.num_samples
    num_retries = args.num_retries

    for method in args.methods:
        basename = f"{model_name}_{method}_m{num_samples}_n{num_retries}"

        out_file = os.path.join(out_dir, f"{basename}.jsonl")
        os.close(os.open(out_file, os.O_CREAT | os.O_TRUNC))

        lm = lf.LanguageModel.get(model_id)
        lock = Lock()

        with lf.track_usages(lm) as usages, lf.track_queries() as queries:
            lf.concurrent_execute(
                func=lambda row: process_row(
                    row, lm, method, num_samples, num_retries, out_file, lock
                ),
                parallel_inputs=dataset,
                max_workers=args.max_workers,
            )

        usages_file = os.path.join(out_dir, f"{basename}_usages.json")
        with open(usages_file, 'w') as file:
            json.dump(usages.to_json(), file)

        queries_file = os.path.join(out_dir, f"{basename}_queries.html")
        with open(queries_file, 'w') as file:
            file.write(str(pg.view(queries)))
