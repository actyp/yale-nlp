from schema import *
from prompt import *
from local_models import Qwen25_7B_Instruct
import langfun as lf
import unittest
import logging
import random

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(thread)d) %(message)s",
    handlers=[logging.FileHandler("inference.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def log_sample_prompts():
    lm = Qwen25_7B_Instruct()
    solution = examples[0].output.solution
    analysis = examples[0].output.analysis

    logger.debug(
        lf.query_prompt(
            prompt=SamplePrompt(
                examples=examples,
                input=task,
            ),
            schema=AnalyticalResponse,
            lm=lm,
            default=None,
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
            default=None,
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
            default=None,
        )
    )

    logger.debug(
        lf.query_prompt(
            prompt=MultiChoicePrompt(
                examples=examples,
                input=task,
                solutions=[solution, solution],
            ),
            schema=MultiChoiceResponse,
            lm=lm,
            default=None,
        )
    )


def sample(task: str, lm: lf.LanguageModel) -> AnalyticalResponse | None:

    return lf.query(
        prompt=SamplePrompt(
            examples=examples,
            input=task,
        ),
        lm=lm,
        schema=AnalyticalResponse,
        default=None,
    )


def verify(task: str, solution: Solution, lm: lf.LanguageModel) -> str | None:

    analysis = lf.query(
        prompt=VerifyPrompt(
            examples=examples,
            input=task,
            solution=solution,
        ),
        lm=lm,
        default=None,
    )
    success = VerifyPrompt.is_successful_analysis(analysis)

    return analysis, success


def correct(
    task: str, solution: Solution, analysis: str, lm: lf.LanguageModel
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
        default=None,
    )


def majority_vote_or_random(
    schema: any,
    schema_title: str,
    schema_items: list[any],
    lm: lf.LanguageModel,
) -> Solution | None:
    if not schema_items:
        return None

    vote = lf.query(
        prompt="What is the majority {{schema_title}} from {{schema_items}}",
        schema=schema,
        schema_title=schema_title,
        schema_items=schema_items,
        lm=lm,
        default=None,
    )

    if vote is not None:
        logger.info("Got majority vote")
        return vote
    else:
        logger.info("Picking majority vote in random")
        return random.choice(schema_items)


def multi_choice(
    task: str,
    solutions: list[Solution],
    lm: lf.LanguageModel,
) -> Solution | None:
    if not schema_items:
        return None

    return lf.query(
        prompt=MultiChoicePrompt(
            input=task,
            examples=examples,
            solutions=solutions,
        ),
        schema=MultiChoiceResponse,
        lm=lm,
        default=None,
    )


def sample_verify_correct(
    task: str, lm: lf.LanguageModel, num_retries: int
) -> (Solution | None, bool, int):
    logger.info("Starting sample_verify_correct")

    attempts = 0
    aresp = sample(task, lm)
    logger.debug(f"AnalyticalResponse: {aresp}")

    if aresp is None:
        logger.critical("Found None analytical response")
        return None, False, attempts

    solution = aresp.solution

    analysis, success = verify(task, solution, lm)
    logger.debug(f"Verification success: {success}, analysis: {analysis}")

    if analysis is None:
        logger.critical("Found None verification analysis")
        return solution, success, attempts

    elif success:
        logger.info(f"Verified solution immediately: {solution}")
        return solution, success, attempts

    else:
        for attempts in range(1, num_retries + 1):
            logger.debug(f"correct-verify attempt: {attempts}")

            cresp = correct(task, solution, analysis, lm)
            logger.debug(f"CorrectionResponse: {cresp}")

            if cresp is None:
                logger.critical("Found None correction response")
                return solution, success, attempts

            solution = cresp.solution

            analysis, success = verify(task, solution, lm)
            logger.debug(f"Verification success: {success}, " f"analysis: {analysis}")

            if analysis is None:
                logger.critical("Found None verification analysis")
                return solution, success, attempts

            elif success:
                logger.info(f"Verified solution in {attempts} attempts")
                return solution, success, attempts

        return solution, success, attempts


def sample_eval(
    task: str,
    lm: lf.LanguageModel,
    num_samples: int,
) -> Solution | None:
    task_id = task[:20]

    logger.info(f"Start sample_eval for task {task_id}")

    map_iterator = lf.concurrent_map(
        func=lambda task: sample(task, lm),
        parallel_inputs=[task] * num_samples,
        show_progress=True,
    )

    sols = []
    for _, output, error in map_iterator:
        sols.append(output)
        if error:
            logger.warning(f"Error response: {error}")

    logger.info(f"End sample_eval for task {task_id}")

    schema_items = sols
    choice = multi_choice(sols, lm)
    logger.debug(f"MultiChoiceResponse: {choice}")

    return choice.solution


def sample_vote(
    task: str,
    lm: lf.LanguageModel,
    num_samples: int,
) -> Solution | None:
    task_id = task[:20]

    logger.info(f"Start sample_vote for task {task_id}")

    map_iterator = lf.concurrent_map(
        func=lambda task: sample(task, lm),
        parallel_inputs=[task] * num_samples,
        show_progress=True,
    )

    sols = []
    for _, output, error in map_iterator:
        sols.append(output)
        if error:
            logger.warning(f"Error response: {error}")

    logger.info(f"End sample_vote for task {task_id}")

    schema_items = sols
    vote = majority_vote_or_random(
        schema=Solution,
        schema_title="Solution",
        schema_items=schema_items,
        lm=lm,
    )

    return vote


def sample_verify(
    task: str,
    lm: lf.LanguageModel,
    num_samples: int,
    num_retries: int,
) -> Solution | None:
    task_id = task[:20]

    logger.info(f"Start sample_verify for task {task_id}")

    map_iterator = lf.concurrent_map(
        func=lambda task: sample_verify_correct(task, lm, num_retries),
        parallel_inputs=[task] * num_samples,
        show_progress=True,
    )

    sols = []
    for _, output, error in map_iterator:
        sols.append(output)
        if error:
            logger.warning(f"Error response: {error}")

    logger.info(f"End sample_verify for task {task_id}")

    verified = []
    non_verified = []
    for sol, success, _ in sols:
        if success:
            verified.append(sol)
        else:
            non_verified.append(sol)

    schema_items = verified or non_verified

    vote = majority_vote_or_random(
        schema=Solution,
        schema_title="Solution",
        schema_items=schema_items,
        lm=lm,
    )

    return vote


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)

    log_sample_prompts()
