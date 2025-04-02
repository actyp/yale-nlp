from schema import Solution
import langfun as lf
import pyglove as pg


class CommonPrompt(lf.structured.Mapping):
    """
    Common base prompt class.
    """

    mapping_template = lf.Template(
        """
        {{ input_title }}:
        {{ example.input_repr(protocol, verbose=False, compact=False)
           | indent(2, True) }}

        {% if example.schema -%}
        {{ schema_title }}:
        {{ example.schema_repr(protocol) | indent(2, True) }}
        {% endif -%}

        {%- if example.has_output %}
        {{ output_title }}:
        {{ example.output_repr(protocol, verbose=False, compact=False)
            | indent(2, True) }}
        {% endif -%}
        """
    )

    input_title = 'TASK'
    schema_title = 'SOLUTION_TYPE'
    output_title = 'SOLUTION'
    example_title = "Here are a few example tasks and solutions:"

    preamble = (
        "You are an expert at planning trips. "
        "You are given a few constraints regarding the cities to visit and "
        "the durations of staying at each city. "
        "You are also given the flight information between the cities."
    )

    def mapping_input_repr(self, input) -> str:
        return lf.MappingExample(
            input=input
        ).input_repr(self.protocol, verbose=False, compact=False)


@pg.use_init_args(['input'])
class SamplePrompt(CommonPrompt):
    """
    Sample prompt class.

    {{ preamble }}

    {% if examples -%}
    {{ example_title }}

    {% for example in examples -%}
    {{ mapping_template.render(example=example) }}
    {% endfor %}

    {% endif -%}

    {{ input_title }}:
    {{ input }}

    {{ request_template.render() }}
    """

    request_template = lf.Template(
        "Please first list all the constraints in the problem and "
        "then output a final solution that satisfies all the constraints."
    )


@pg.use_init_args(['solution'])
class VerifyPrompt(CommonPrompt):
    """
    Verify prompt class.

    {{ preamble }}

    {% if examples -%}
    {{ example_title }}

    {% for example in examples -%}
    {{ mapping_template.render(example=example) }}
    {% endfor %}

    {% endif -%}

    {{ request_template.render(
        input_title=input_title, solution_title=solution_title) }}

    {{ solution_title }}:
    {{ mapping_input_repr(solution) }}
    """

    input = ""
    solution_title = 'PROPOSED SOLUTION'
    solution: Solution

    request_template = lf.Template(
        "You are an expert at planning trips. "
        "You are given a {{ input_title }} of Trip Planning request, "
        "and a {{ solution_title }}. Your job is to:\n"
        "1. List all constraints in the TASK.\n"
        "2. Verify if the PROPOSED SOLUTION satisfies each of the constraints "
        "with justifications.\n"
        "3. Write a line of the form \"The proposed solution is correct\" "
        "or \"The proposed solution is incorrect\" at the end of your "
        "response based on your analysis."
    )


@pg.use_init_args(['input', 'solution', 'analysis'])
class CorrectPrompt(CommonPrompt):
    """
    Correct prompt class.

    {{ preamble }}

    {% if examples -%}
    {{ example_title }}

    {% for example in examples -%}
    {{ mapping_template.render(example=example) }}
    {% endfor %}

    {% endif -%}

    {{ request_template.render(input_title=input_title) }}

    {{ input_title }}:
    {{ input }}

    {{ solution_title }}:
    {{ mapping_input_repr(solution) }}

    {{ analysis_title }}:
    {{ analysis }}
    """

    solution: Solution
    analysis: str
    solution_title = 'PROPOSED SOLUTION'
    analysis_title = 'ANALYSIS'

    request_template = lf.Template(
        "You are an expert at planning trips. "
        "You are given a {{ input_title }} of Trip Planning request. "
        "You are also given pairs of "
        "({{ solution_title }}, {{ analysis_title }})."
        "Your job is to outline your step-by-step thought process for "
        "deriving a new solution."
    )
