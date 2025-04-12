from typing import Annotated
import pyglove as pg
import langfun as lf
import re


class Solution(lf.PythonCode):
    def clean_resp(resp):
        return clean_resp(resp, "Solution")


class AnalyticalResponse(pg.Object):
    "Response containing analytical constraints and final solution."

    analysis: Annotated[str, "List all the constraints in the problem."]

    solution: Annotated[
        Solution, "The final solution that satisfies all the constraints."
    ]

    def clean_resp(resp):
        return clean_resp(resp, "AnalyticalResponse")


class CorrectionResponse(pg.Object):
    "Response containing thoughts and new corrected solution."

    thought: Annotated[
        str, "Outline your step-by-step thought process for deriving a new solution."
    ]

    solution: Annotated[
        Solution, "The new solution that satisfies all the constraints."
    ]

    def clean_resp(resp):
        return clean_resp(resp, "CorrectionResponse")


class MultipleChoiceResponse(pg.Object):
    "Response containing thoughts and multiple choice solution."

    thought: Annotated[
        str,
        "Outline your step-by-step thought process for selecting the best solution.",
    ]

    solution: Annotated[
        Solution, "The best solution that satisfies all the constraints."
    ]

    def clean_resp(resp):
        return clean_resp(resp, "MultipleChoiceResponse")


examples = [
    lf.MappingExample(
        input="""\
import itertools
from random import shuffle

def task_func(numbers=list(range(1, 10))):
    \"\"\"
    Calculates the sum of multiplication between each pair of consecutive numbers for all permutations of a given list. Each permutation is shuffled before calculating the multiplications.

    Args:
    - numbers (list): A list of numbers. Default is numbers from 1 to 10.

    Returns:
    int: The sum of multiplications for each shuffled permutation of the list.

    Requirements:
    - itertools
    - random.shuffle

    Example:
    >>> result = task_func([1, 2, 3])
    >>> isinstance(result, int)
    True
    \"\"\"
""",
        schema=AnalyticalResponse,
        output=AnalyticalResponse(
            analysis="""\
            First, we take all permutations of provided list of numbers using itertools.permutations
            Second, we iterate through the permutation lists
            Third, in each iteration we do the following:
                - shuffle the permutation list
                - calculate the consecutive number multiplications
                - add the multiplications to a common variable
            Finally, after the iterations we return the common variable holding the accumulated sums
            """,

            solution=Solution(
                """\
import itertools
from random import shuffle

def task_func(numbers=list(range(1, 3))):
    permutations = list(itertools.permutations(numbers))
    sum_mults = 0

    for perm in permutations:
        perm = list(perm)
        shuffle(perm)
        mults = [perm[i] * perm[i+1] for i in range(len(perm)-1)]
        sum_mults += sum(mults)

    return sum_mults
"""
            ),
        ),
    )
]


def clean_resp(resp: str, schema_cls: str) -> str:
    schema_use = f"{schema_cls}("
    segment_pat = r"(.*?)```python(.*?)```"

    pair_chunks = []
    last_end = 0

    for match in re.finditer(segment_pat, resp, re.DOTALL):
        _, last_end = match.span()
        text = match.group(1).strip()
        code = match.group(2).strip()
        pair_chunks.append((text, code))

    # append remaining text after last code block
    if last_end < len(resp):
        text = resp[last_end:].strip()
        pair_chunks.append((text, ""))

    out = ""
    for text, code in pair_chunks:
        out += f"{text}\n"

        # remove code lines before schema usage
        # remove code blocks without schema usage
        start = code.find(schema_use)
        if start >= 0:
            out += f"```python\n{code[start:]}\n```\n"

    return out
