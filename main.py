from data_parse import get_dataset
from inference import sample_eval, sample_verify, sample_vote
import argparse
import json
import langfun as lf
import os


def main(model_id, dataset, method, num_samples, num_retries, out_file):
    # Load the model
    lm = lf.LanguageModel.get(model_id)

    funcargs = {
        "sample_vote": (sample_vote, [lm, num_samples]),
        "sample_verify": (sample_verify, [lm, num_samples, num_retries]),
        "sample_eval": (sample_eval, [lm, num_samples]),
    }

    # Iterate through the dataset
    for idx, sample in enumerate(iter(dataset)):
        # Extract the task and test case from the sample
        task_id = sample["task_id"]
        complete_promt = sample["complete_prompt"]

        # Perform SETS
        func, args = funcargs[method]
        solution = func(complete_promt, *args)

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
    parser.add_argument(
        "--model_id", type=str, required=True, help="The ID of the model to use."
    )
    args = parser.parse_args()

    cwd = os.getcwd()
    sol_dir = os.path.join(cwd, "evaluation")

    # Create directory if it doesn't exist
    if not os.path.exists(sol_dir):
        os.makedirs(sol_dir)

    dataset_id = "bigcode/bigcodebench"

    methods = ["sample_vote", "sample_verify", "sample_eval"]

    # Load the dataset
    dataset = get_dataset(dataset_id, streaming=True, split="v0.1.4")

    num_samples = 3
    num_retries = 3

    model_id = args.model_id
    model_name = model_id.replace("/", "--")

    for method in methods:
        out_file = os.path.join(sol_dir, f"{model_name}_{method}.jsonl")

        # Create file if it doesn't exist
        if not os.path.exists(out_file):
            f = open(out_file, "w")
            f.close()

        main(model_id, dataset, method, num_samples, num_retries, out_file=out_file)
