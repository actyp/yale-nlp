from typing import Annotated
import pyglove as pg
import langfun as lf


class Solution(lf.PythonCode):
    pass


class AnalyticalResponse(pg.Object):
    "Response containing analytical constraints and final solution."

    analysis: Annotated[
        str, "List all the constraints in the problem."
    ]

    solution: Annotated[
        Solution, "The final solution that satisfies all the constraints."
    ]


class CorrectionResponse(pg.Object):
    "Response containing thoughts and new corrected solution."

    thought: Annotated[
        str, "Outline your step-by-step thought process for deriving a new solution."
    ]

    solution: Annotated[
        Solution, "The new solution that satisfies all the constraints."
    ]


examples = [
    lf.MappingExample(
        input="""\
import itertools
from random import shuffle

def task_func(numbers=list(range(1, 3))):
    \"\"\"
    Calculates the average of the sums of absolute differences between each pair of consecutive numbers
    for all permutations of a given list. Each permutation is shuffled before calculating the differences.

    Args:
    - numbers (list): A list of numbers. Default is numbers from 1 to 10.

    Returns:
    float: The average of the sums of absolute differences for each shuffled permutation of the list.

    Requirements:
    - itertools
    - random.shuffle

    Example:
    >>> result = task_func([1, 2, 3])
    >>> isinstance(result, float)
    True
    \"\"\"
""",
        schema=AnalyticalResponse,
        output=AnalyticalResponse(
            analysis="Analysis of the derived code",
            solution=Solution("""\
import itertools
from random import shuffle

def task_func(numbers=list(range(1, 3))):
    permutations = list(itertools.permutations(numbers))
    sum_diffs = 0

    for perm in permutations:
        perm = list(perm)
        shuffle(perm)
        diffs = [abs(perm[i] - perm[i+1]) for i in range(len(perm)-1)]
        sum_diffs += sum(diffs)

    avg_sum_diffs = sum_diffs / len(permutations)

    return avg_sum_diffs\
""")),
    )
]

task = ("""\
import collections
import random
import string

def task_func(length=100):
    \"\"\"
    Generate a random string of the specified length composed of uppercase and lowercase letters,
    and then count the occurrence of each character in this string.

    Parameters:
    length (int, optional): The number of characters in the generated string. Default is 100.

    Returns:
    dict: A dictionary where each key is a character from the generated string and the value
            is the count of how many times that character appears in the string.

    Requirements:
    - collections
    - random
    - string

    Raises:
    ValueError if the length is a negative number

    Example:
    >>> import random
    >>> random.seed(42)  # Ensures reproducibility for demonstration
    >>> task_func(10)
    {'h': 1, 'B': 2, 'O': 1, 'L': 1, 'm': 1, 'j': 1, 'u': 1, 'E': 1, 'V': 1}
    \"\"\"
""")

test_case_str = ("""\
import unittest
import string
class TestCases(unittest.TestCase):
    def setUp(self):
        # Prepare valid characters and set a random seed for reproducibility
        self.valid_chars = string.ascii_uppercase + string.ascii_lowercase
        random.seed(42)  # Ensuring reproducibility for tests
    def test_generated_string_properties(self):
        # Consolidated test for different lengths to check structure and correctness
        test_lengths = [10, 50, 100, 150, 5]
        for length in test_lengths:
            with self.subTest(length=length):
                result = task_func(length)
                self.assertTrue(len(result) <= length, "Length of result should be <= requested string length")
                self.assertEqual(sum(result.values()), length, f"Total counts should sum to {length}")
                self.assertTrue(all(char in self.valid_chars for char in result), "All characters should be valid letters")
    def test_zero_length(self):
        # Test edge case where length is zero
        result = task_func(0)
        self.assertEqual(len(result), 0, "Result should be empty for zero length")
        self.assertEqual(sum(result.values()), 0, "Sum of counts should be zero for zero length")
    def test_negative_length(self):
        # Test handling of negative length input
        with self.assertRaises(ValueError, msg="Negative length should raise an error"):
            task_func(-1)
""")
