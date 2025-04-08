from inference import sample_once, sample_vote, sample_eval, sample_veco
from typing import Iterable, Callable, SupportsIndex
from local_models import SUPPORTED_MODELS
from data_parse import get_dataset
import multiprocessing as mp
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


def parallel_process(
    par_iter: Iterable,
    par_fun: Callable[any, any],
    par_arg_gen: Callable[[object, mp.Lock], any],
    processes: int = None
):
    with mp.Manager() as manager:
        lock = manager.Lock()

        with mp.Pool(processes) as pool:
            for item in par_iter:
                pool.apply_async(
                    par_fun,
                    args=par_arg_gen(item, lock),
                    error_callback=lambda e: print(f"Error: {e}", flush=True),
                )

            pool.close()
            pool.join()


def process_row(
    row: SupportsIndex,
    lm: lf.LanguageModel,
    method: str,
    num_samples: int,
    num_retries: int,
    out_file: str,
    out_file_lock: mp.Lock,
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
    solution = func(task_id, complete_prompt, *args)

    solution_source = solution.source if solution is not None else "None"

    save_dict = {
        "task_id": task_id,
        "solution": solution_source,
        "raw_solution": solution_source,
    }

    with out_file_lock:
        with open(out_file, "a") as file:
            file.write(json.dumps(save_dict) + "\n")


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

    parser.add_argument("--processes", type=int, default=None,
                        help="Limit the number of parallel processes")

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
    processes = args.processes

    for method in args.methods:
        filename = f"{model_name}_{method}_m{num_samples}_n{num_retries}.jsonl"
        out_file = os.path.join(out_dir, filename)
        os.close(os.open(out_file, os.O_CREAT | os.O_TRUNC))

        lm = lf.LanguageModel.get(model_id)

        def process_row_arg_gen(row, lock):
            return (row, lm, method, num_samples, num_retries, out_file, lock)

        parallel_process(dataset, process_row, process_row_arg_gen, processes)
