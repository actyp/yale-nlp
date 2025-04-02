from typing import Annotated, Optional
import pyglove as pg


class Step(pg.Object):
    """One solution step."""

    city_name: Annotated[
        Optional[str], "The city name."
    ]

    arrival_day: Annotated[
        Optional[int], "The day you arrive in the city."
    ]

    departure_day: Annotated[
        Optional[int], "The day you depart from the city."
    ]

    duration: Annotated[
        Optional[int], "The number of days spent in the city."
    ]


class Solution(pg.Object):
    """Solution containing many steps."""

    steps: Annotated[
        list[Step | None], "The steps leading to the solution."
    ]


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
