from typing import Annotated, Optional
import langfun as lf
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


@pg.use_init_args(['request', 'input'])
class SamplePrompt(lf.structured.Mapping):
    """
    Sample prompt class.

    {{ preamble }}

    {% if examples -%}
    {{ example_title }}

    {% for example in examples -%}
    {{ mapping_template.render(example=example) }}
    {% endfor %}

    {% endif -%}

    {{ mapping_template.render(example=mapping_request) }}
    """

    mapping_template = lf.Template(
        """
        {{ input_title }}:
        {{ example.input_repr(protocol, verbose=False, compact=False) | indent(2, True) }}

        {% if not example.has_output -%}
        {{ request | indent(2, True) }}
        {% endif -%}

        {% if example.schema -%}
        {{ schema_title }}:
        {{ example.schema_repr(protocol) | indent(2, True) }}
        {% endif -%}

        {%- if example.has_output %}
        {{ output_title }}:
        {{ example.output_repr(protocol, verbose=False, compact=False) | indent(2, True) }}
        {% endif -%}
        """
    )

    input_title = 'TASK'
    schema_title = 'SOLUTION_TYPE'
    output_title = 'SOLUTION'
    example_title = "Here are a few example tasks and solutions:"


@pg.use_init_args(['request', 'solution'])
class VerifyPrompt(lf.structured.Mapping):
    """
    Verify prompt class.

    {{ preamble }}

    {% if examples -%}
    {{ example_title }}

    {% for example in examples -%}
    {{ mapping_template.render(example=example) }}
    {% endfor %}

    {% endif -%}

    {{ request }}

    {{ solution_title }}:
    {{ solution_repr }}
    """

    mapping_template = lf.Template(
        """
        {{ input_title }}:
        {{ example.input_repr(protocol, verbose=False, compact=False) | indent(2, True) }}

        {% if example.schema -%}
        {{ schema_title }}:
        {{ example.schema_repr(protocol) | indent(2, True) }}
        {% endif -%}

        {%- if example.has_output %}
        {{ output_title }}:
        {{ example.output_repr(protocol, verbose=False, compact=False) | indent(2, True) }}
        {% endif -%}
        """
    )

    input = ""
    input_title = 'TASK'
    schema_title = 'SOLUTION_TYPE'
    output_title = 'SOLUTION'
    example_title = "Here are a few example tasks and solutions:"
    solution_title = 'PROPOSED SOLUTION'
    solution: Solution

    @property
    def solution_repr(self) -> str:
        return lf.MappingExample(
            input=self.solution
        ).input_repr(self.protocol, verbose=False, compact=False)


@pg.use_init_args(['request', 'input', 'solution', 'analysis'])
class CorrectPrompt(lf.structured.Mapping):
    """
    Correct prompt class.

    {{ preamble }}

    {% if examples -%}
    {{ example_title }}

    {% for example in examples -%}
    {{ mapping_template.render(example=example) }}
    {% endfor %}

    {% endif -%}

    {{ request }}

    {{ input_title }}:
    {{ input }}

    {{ solution_title }}:
    {{ solution_repr }}

    {{ analysis_title }}:
    {{ analysis }}
    """

    mapping_template = lf.Template(
        """
        {{ input_title }}:
        {{ example.input_repr(protocol, verbose=False, compact=False) | indent(2, True) }}

        {%- if not example.has_output %}
        {{ example.output_repr(protocol, verbose=False, compact=False) | indent(2, True) }}
        {% endif -%}

        {% if example.schema -%}
        {{ schema_title }}:
        {{ example.schema_repr(protocol) | indent(2, True) }}
        {% endif -%}

        {%- if example.has_output %}
        {{ output_title }}:
        {{ example.output_repr(protocol, verbose=False, compact=False) | indent(2, True) }}
        {% endif -%}
        """
    )

    input_title = 'TASK'
    schema_title = 'SOLUTION_TYPE'
    output_title = 'SOLUTION'
    example_title = "Here are a few example tasks and solutions:"
    solution_title = 'PROPOSED SOLUTION'
    solution: Solution
    analysis_title = 'Analysis'
    analysis: str

    @property
    def solution_repr(self) -> str:
        return lf.MappingExample(
            input=self.solution
        ).input_repr(self.protocol, verbose=False, compact=False)
