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


class Response(pg.Object):
    "Response containing analytical constraints and final solution."

    analysis: Annotated[
        str, "List all the constraints in the problem."
    ]

    solution: Annotated[
        Solution, "The final solution that satisfies all the constraints."
    ]


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
    {{ mapping_request.input_repr(protocol, verbose=False, compact=False) }}
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

    input_title = 'TASK'
    schema_title = 'SOLUTION_TYPE'
    output_title = 'SOLUTION'
    solution_title = 'PROPOSED SOLUTION'
    example_title = "Here are a few example tasks and solutions:"
