from inference import sample_once, sample_vote, sample_eval, sample_veco
from local_models import SUPPORTED_MODELS
from data_parse import get_dataset
from typing import SupportsIndex
from threading import Lock
import pyglove as pg
import langfun as lf
import traceback
import argparse
import logging
import json
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(thread)d): <%(funcName)s> %(message)s",
)
logger = logging.getLogger(__name__)

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
    queries_dir: str,
):
    funcargs = {
        "sample_once": (sample_once, [lm]),
        "sample_vote": (sample_vote, [lm, num_samples]),
        "sample_eval": (sample_eval, [lm, num_samples]),
        "sample_veco": (sample_veco, [lm, num_samples, num_retries]),
    }

    task_id = row["task_id"]
    complete_prompt = row["complete_prompt"]

    filename = f"{task_id.replace('/', '--')}.html"
    queries_file = os.path.join(queries_dir, filename)

    func, args = funcargs[method]
    try:
        with lf.track_queries() as queries:
            dct = func(task_id, complete_prompt, *args)

    except Exception:
        logger.warning(f"Exception during function: {traceback.format_exc()}")
        dct = {"solution": None, "details": "RaisedException"}

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

    try:
        with open(queries_file, 'w') as file:
            for query in queries:
                file.write(pg.to_html_str(query) + '\n')
    except Exception:
        logger.warning(f"Exception during query: {traceback.format_exc()}")


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

    parser.add_argument("--max_workers", type=int, default=20,
                        help="Number of concurrent workers (can be None)")

    parser.add_argument("--query_num", type=int, default=200,
                        help="Query number to randomly sample and store")

    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout of a single query")

    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling parameter temperature")

    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Sampling parameter top p")
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

        queries_dir = os.path.join(out_dir, f"{basename}_queries")
        os.makedirs(queries_dir, exist_ok=True)

        usages_file = os.path.join(out_dir, f"{basename}_usages.json")

        lm = lf.LanguageModel.get(
            model_id,
            timeout=args.timeout,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        lock = Lock()

        with lf.track_usages(lm) as usages:
            lf.concurrent_execute(
                func=lambda row: process_row(
                    row, lm, method, num_samples, num_retries,
                    out_file, lock, queries_dir,
                ),
                parallel_inputs=dataset,
                max_workers=args.max_workers,
            )

        with open(usages_file, 'w') as file:
            file.write(usages.to_json_str())
