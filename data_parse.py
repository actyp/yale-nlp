from datasets import load_dataset


def get_dataset(dataset: str, streaming: bool, split: str):
    """
    Load a dataset from the Hugging Face Hub.

    Args:
        dataset (str): The name of the dataset to load.

    Returns:
        datasets.DatasetDict: The loaded dataset.
    """
    return load_dataset(dataset, streaming, split)


if __name__ == "__main__":
    dataset = get_dataset("bigcode/bigcodebench")

    # print first sample
    iterator = iter(dataset)
    sample_0 = next(iterator)

    print(sample_0.keys())
