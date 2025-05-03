# Evaluating [Google SETS](https://arxiv.org/abs/2501.19306) in Code Generation

## Environment Setup

The setup assumes access to an HPC cluster with Slurm manager and modules.

The necessary environments are:

- vllm: for the online vllm inference
- nlp: for the langfun client
- bcb: for the remote evaluation using the BigCodeBench API


To setup all the environments use:
```bash
bash setup_envs.sh
```

## Result Generation

To generate the results for a specific model:

1. Uncomment one or more lines from the bottom of generate.slurm starting with `run_main --model_id`
    - optionally add any additional arguments of main.py at the end of those lines
    - check the available ones using: `python main.py --help`
    - e.g. in order to run only the SETS method, add: `--methods sample_veco`

2. Run `generate.slurm`:

```bash

# either schedule the job in the cluster
sbatch generate.slurm

# or run the job interactively at current node (if it has GPU access)
bash generate.slurm
```


Upon successful completion, the results can be found in the `evaluation` directory.

The common basename format for a specific model and method is:
```bash
basename = {model_name}_{method}_m{num_samples}_n{num_retries}
```

And the resulting files are:

- `{basename}.jsonl`: a jsonl file with the model's solutions per task
- `{basename}_queries`: a directory with html files containing the queries required for a single task for manual inspection (needs much storage space)
- `{basename}_usages.json`: a json file with aggregate statistics over all tasks


## Result Evaluation

To evaluate the results of a specific model and method, use:
```bash
bash eval.sh evaluation/{basename}.jsonl
```

To evaluate many results in the background (assuming default n=4), use:
```bash
for f in $(ls evaluation/*_n4.jsonl); do bash eval.sh $f & done
```

These are the stages that the `evaluation/{basename}.jsonl` file goes through before being sent to the BigCodeBench remote API for evaluation:

1. Sanitization: Sanitizes `evaluation/{basename}.jsonl` and creates `evaluation/{basename}-sanitized-calibrated.jsonl`

2. Synchecking: Checks `evaluation/{basename}-sanitized-calibrated.jsonl` for non-compilable proposed solutions

3. Evaluation: Sends `evaluation/{basename}-sanitized-calibrated.jsonl` to the remote API of BigCodeBench for evaluation, which responds with these files:
    - `evaluation/{basename}-sanitized-calibrated_eval_results.json`: contains analytical evaluation details for each task

    - `evaluation/{basename}-sanitized-calibrated_pass_at_k.json`: contains some information about the evaluation parameters, pass@1 rate and ground truth pass rate

> Steps 1 and 2 will fail if something goes wrong. For example, it is possible that synchecking may require manual removal of problematic solutions in `evaluation/{basename}.jsonl` that were not removed/nullified in any prior step. Nevertheless, those cases should not be that many. Additionally, Step 3 requires waiting for the remote API to respond, which may need some time depending on its availability. Wait for the results and let the processes retry automatically.

## Result Visualization

After a successful evaluation, all `evaluation/*_eval_results.json` and `evaluation/*_usages.json`
files can be used to generate plots in the `plots` directory using:
```bash
python plot.py
```


## Result Availability

Available results can be found [here](https://yaleedu-my.sharepoint.com/:f:/g/personal/thanos_typaldos_yale_edu/EqU1sy1WSbJFv_uOIxHsCcoBPugiLygjlrdb6gDlDUYK4w?e=gcSI3K) divided into two zip files:

- `evaluation_json.zip`: contains everything related to the results and evaluation
- `evaluation_queries.zip`: contains the query directories for manual inspection
