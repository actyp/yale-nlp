from schema import Step, Solution, AnalyticalResponse, CorrectionResponse
from prompt import SamplePrompt, VerifyPrompt, CorrectPrompt
from local_models import Qwen25_7B_Instruct
import langfun as lf


task = ("""\
You plan to visit 6 European cities for 16 days in total. You only take \
direct flights to commute between cities. On the last day of your visit to \
each city, you can take a direct flight to the next city and arrive on the \
same day. Both the day you arrive and the day you depart count toward the \
total number of days spent in each city. You would like to visit Lyon for \
3 days. You plan to stay in Athens for 4 days. You would like to visit \
Dubrovnik for 2 days. You plan to stay in Porto for 4 days. You have to \
attend a workshop in Porto between day 11 and day 14. You would like to \
visit Helsinki for 5 days. From day 5 to day 9, there is a annual show you \
want to attend in Helsinki. You would like to visit Milan for 3 days.

Here are the cities that have direct flights:
 Athens and Milan, Milan and Porto, Porto and Lyon, Athens and Dubrovnik, \
 Dubrovnik and Helsinki, Helsinki and Milan.

Find a trip plan of visiting the cities for 16 days by taking direct flights \
to commute between them.\
""")

solution = Solution(steps=[
    Step(
        city_name='Rome',
        arrival_day=1,
        departure_day=3,
        duration=3
    ),
    Step(
        city_name='Barcelona',
        arrival_day=3,
        departure_day=6,
        duration=4
    ),
])

examples = [
    lf.MappingExample(
        input='Input of first example of trip planning',
        schema=AnalyticalResponse,
        output=AnalyticalResponse(
            analysis="Analysis of first input example",
            solution=solution
        )
    )
]

analysis = "Verification analysis"

sample_prompt = SamplePrompt(
    examples=examples,
    input=task,
)

verify_prompt = VerifyPrompt(
    examples=examples,
    solution=solution,
)

correct_prompt = CorrectPrompt(
    examples=examples,
    input=task,
    solution=solution,
    analysis=analysis,
)

print(
    lf.query_prompt(
        prompt=sample_prompt,
        schema=AnalyticalResponse,
        lm=Qwen25_7B_Instruct(),
        default=None
    ),
    lf.query_prompt(
        prompt=verify_prompt,
        lm=Qwen25_7B_Instruct(),
        default=None
    ),
    lf.query_prompt(
        prompt=correct_prompt,
        schema=CorrectionResponse,
        lm=Qwen25_7B_Instruct(),
        default=None
    ),
    sep='\n\n' + 100 * '=' + '\n\n',
)
