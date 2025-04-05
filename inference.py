from schema import Solution, AnalyticalResponse, CorrectionResponse
from prompt import SamplePrompt, VerifyPrompt, CorrectPrompt
from local_models import Qwen25_7B_Instruct
from schema import examples, solution
import langfun as lf
import logging
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] (%(thread)d) %(message)s',
    handlers=[
        logging.FileHandler("inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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


def log_sample_prompts():
    analysis = "Verification analysis"
    lm = Qwen25_7B_Instruct()

    logger.debug(
        lf.query_prompt(
            prompt=SamplePrompt(
                examples=examples,
                input=task,
            ),
            schema=AnalyticalResponse,
            lm=lm,
            default=None
        ),
    )

    logger.debug(
        lf.query_prompt(
            prompt=VerifyPrompt(
                examples=examples,
                input=task,
                solution=solution,
            ),
            lm=lm,
            default=None
        )
    )

    logger.debug(
        lf.query_prompt(
            prompt=CorrectPrompt(
                examples=examples,
                input=task,
                solution=solution,
                analysis=analysis,
            ),
            schema=CorrectionResponse,
            lm=lm,
            default=None
        )
    )


def sample(
    task: str,
    lm: lf.LanguageModel
) -> AnalyticalResponse | None:

    return lf.query(
        prompt=SamplePrompt(
            examples=examples,
            input=task,
        ),
        lm=lm,
        schema=AnalyticalResponse,
        default=None
    )


def verify(
    task: str,
    solution: Solution,
    lm: lf.LanguageModel
) -> str | None:

    analysis = lf.query(
        prompt=VerifyPrompt(
            examples=examples,
            input=task,
            solution=solution,
        ),
        lm=lm,
        default=None
    )
    success = VerifyPrompt.is_successful_analysis(analysis)

    return analysis, success


def correct(
    task: str,
    solution: Solution,
    analysis: str,
    lm: lf.LanguageModel
) -> CorrectionResponse | None:

    return lf.query(
        prompt=CorrectPrompt(
            examples=examples,
            input=task,
            solution=solution,
            analysis=analysis,
        ),
        schema=CorrectionResponse,
        lm=lm,
        default=None
    )


def majority_vote_or_random(
    schema: any,
    schema_title: str,
    schema_items: list[any],
    lm: lf.LanguageModel,
):
    if not schema_items:
        return None

    vote = lf.query(
        prompt='What is the majority {{schema_title}} from {{schema_items}}',
        schema=schema,
        schema_title=schema_title,
        schema_items=schema_items,
        lm=lm,
        default=None,
    )

    if vote is not None:
        return vote
    else:
        return random.choice(schema_items)


def sample_verify_correct_one(
    task: str,
    lm: lf.LanguageModel,
    num_retries: int
) -> (Solution | None, int):
    logger.info("Starting sample_verify_correct")

    attempts = 0
    aresp = sample(task, lm)
    logger.debug(f"AnalyticalResponse: {aresp}")

    if aresp is None:
        logger.critical("Found None analytical response")
        return None, attempts

    solution = aresp.solution

    analysis, success = verify(task, solution, lm)
    logger.debug(f"Verification success: {success}, analysis: {analysis}")

    if analysis is None:
        logger.critical("Found None verification analysis")
        return None, attempts

    elif success:
        logger.info(f"Verified solution immediately: {solution}")
        return solution, attempts

    else:
        for attempts in range(1, num_retries + 1):
            logger.debug(f"correct-verify attempt: {attempts}")

            cresp = correct(task, solution, analysis, lm)
            logger.debug(f"CorrectionResponse: {cresp}")

            if cresp is None:
                logger.critical("Found None correction response")
                return None, attempts

            solution = cresp.solution

            analysis, success = verify(task, solution, lm)
            logger.debug(f"Verification success: {success}, "
                         f"analysis: {analysis}")

            if analysis is None:
                logger.critical("Found None verification analysis")
                return None, attempts

            elif success:
                logger.info(f"Verified solution in {attempts} attempts")
                return solution, attempts

        return None, attempts


def sample_verify_correct(
    task: str,
    lm: lf.LanguageModel,
    num_samples: int,
    num_retries: int,
) -> list[(Solution | None, int)]:
    task_id = task[:20]

    logger.info(f"Start sample_verify_correct for task {task_id}")

    map_iterator = lf.concurrent_map(
        func=lambda task: sample_verify_correct_one(task, lm, num_retries),
        parallel_inputs=[task] * num_samples,
        show_progress=True,
    )

    sols = []
    for _, output, error in map_iterator:
        sols.append(output)
        if error:
            logger.warning(f"Error response: {error}")

    logger.info(f"End sample_verify_correct for task {task_id}")

    vote = majority_vote_or_random(
        schema=Solution,
        schema_title='Solution',
        schema_items=list(filter(None, sols)),
        lm=lm,
    )

    return vote, sols


if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)

    log_sample_prompts()

    lm = Qwen25_7B_Instruct()
    vote, sols = sample_verify_correct(task, lm, num_samples=3, num_retries=3)

    logger.info(f"Solutions: {sols}\n\nMajority Vote: {vote}")
