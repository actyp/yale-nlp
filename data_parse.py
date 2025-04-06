from datasets import load_dataset


def get_dataset(dataset: str, split: str):
    """
    Load a dataset from the Hugging Face Hub.

    Args:
        dataset (str): The name of the dataset to load.
        split (str): The split of the dataset to load.

    Returns:
        datasets.DatasetDict: The loaded dataset.
    """
    return load_dataset(dataset, streaming=True, split=split)


if __name__ == "__main__":
    dataset = get_dataset("bigcode/bigcodebench", "v0.1.4")

    # print first sample
    iterator = iter(dataset)
    sample_0 = next(iterator)

    print(sample_0.keys())
