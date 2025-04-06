from inference import sample_once, sample_eval, sample_verify, sample_vote
from data_parse import get_dataset
import langfun as lf
import argparse
import json
import os


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

        # Create jsonl file:
        save_dict = {
            "task_id": task_id,
            "solution": solution.source,
            "raw_solution": solution.source,
        }
        # dump save_dict to jsonl file -- in append mode
        with open(out_file, "a+") as f:
            f.write(json.dumps(save_dict) + "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run SETS with model.")
    parser.add_argument("--model_id", type=str, required=True,
                        help="The ID of the model to use.")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="The number of samples (m) to use.")
    parser.add_argument("--num_retries", type=int, default=3,
                        help="The number of retries (n) to use.")
    args = parser.parse_args()

    cwd = os.getcwd()
    sol_dir = os.path.join(cwd, "evaluation")

    # Create directory if it doesn't exist
    os.makedirs(sol_dir, exist_ok=True)

    dataset_id = "bigcode/bigcodebench"

    methods = ["sample_once", "sample_vote", "sample_eval", "sample_verify"]

    # Load the dataset
    dataset = get_dataset(dataset_id, streaming=True, split="v0.1.4")

    num_samples = args.num_samples
    num_retries = args.num_retries

    model_id = args.model_id
    model_name = model_id.replace("/", "--")

    for method in methods:
        filename = f"{model_name}_{method}_m{num_samples}_n{num_retries}.jsonl"
        out_file = os.path.join(sol_dir, filename)

        # Create file if it doesn't exist
        if not os.path.exists(out_file):
            f = open(out_file, "w")
            f.close()

        main(model_id, dataset, method, num_samples, num_retries, out_file)
