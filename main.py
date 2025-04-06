from inference import sample_once, sample_eval, sample_verify, sample_vote
from local_models import SUPPORTED_MODELS
from data_parse import get_dataset
import langfun as lf
import argparse
import json
import os


METHODS = ["sample_once", "sample_vote", "sample_eval", "sample_verify"]


def main(model_id, dataset, method, num_samples, num_retries, out_file):
    # Load the model
    lm = lf.LanguageModel.get(model_id)

    funcargs = {
        "sample_once": (sample_once, [lm]),
        "sample_vote": (sample_vote, [lm, num_samples]),
        "sample_eval": (sample_eval, [lm, num_samples]),
        "sample_verify": (sample_verify, [lm, num_samples, num_retries]),
    }

    # Iterate through the dataset
    for idx, sample in enumerate(iter(dataset)):
        # Extract the task and test case from the sample
        task_id = sample["task_id"]
        complete_prompt = sample["complete_prompt"]

        # Perform SETS
        func, args = funcargs[method]
        solution = func(task_id, complete_prompt, *args)

        solution_source = solution.source if solution is not None else ""

        # Create jsonl file:
        save_dict = {
            "task_id": task_id,
            "solution": solution_source,
            "raw_solution": solution_source,
        }
        # dump save_dict to jsonl file -- in append mode
        with open(out_file, "a") as f:
            f.write(json.dumps(save_dict) + "\n")


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
        filename = f"{model_name}_{method}_m{num_samples}_n{num_retries}.jsonl"
        out_file = os.path.join(out_dir, filename)

        main(model_id, dataset, method, num_samples, num_retries, out_file)
