from schema import *
from prompt import *
import langfun as lf
import logging
import random

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(thread)d): <%(funcName)s> %(message)s",
    handlers=[logging.FileHandler("inference.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


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


def multiple_choice(
    task: str,
    solutions: list[Solution],
    lm: lf.LanguageModel,
) -> MultipleChoiceResponse | None:
    if not solutions:
        return None

    with lm.sampling_options.override(temperature=0):
        return lf.query(
            prompt=MultipleChoicePrompt(
                examples=examples,
                input=task,
                solutions=solutions,
            ),
            schema=MultipleChoiceResponse,
            lm=lm,
            default=None,
        )


def majority_vote_or_random(
    solutions: list[Solution],
    lm: lf.LanguageModel,
) -> Solution | None:
    if not solutions:
        return None

    vote = lf.query(
        prompt="What is the majority {{schema_title}} from {{schema_items}}",
        schema=Solution,
        schema_title='Solution',
        schema_items=solutions,
        lm=lm,
        default=None,
    )

    if vote is not None:
        logger.info("Got majority vote")
        return vote
    else:
        logger.info("Randomly choosing majority vote")
        return random.choice(solutions)


def sample_verify_correct(
    task_id: str,
    task: str,
    lm: lf.LanguageModel,
    num_retries: int
) -> (Solution | None, bool, int):
    logger.info(f"Start task {task_id}")

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
            logger.debug(f"Correct-Verify attempt: {attempts}")

            cresp = correct(task, solution, analysis, lm)
            logger.debug(f"CorrectionResponse: {cresp}")

            if cresp is None:
                logger.critical("Found None correction response")
                return solution, success, attempts

            solution = cresp.solution

            analysis, success = verify(task, solution, lm)
            logger.debug(f"Verification success: {success}, analysis: {analysis}")

            if analysis is None:
                logger.critical("Found None verification analysis")
                return solution, success, attempts

            elif success:
                logger.info(f"Verified solution in {attempts} attempts: {solution}")
                return solution, success, attempts

        return solution, success, attempts


def sample_once(
    task_id: str,
    task: str,
    lm: lf.LanguageModel,
) -> Solution | None:
    logger.info(f"Start task {task_id}")
    aresp = sample(task, lm)

    logger.info(f"End task {task_id}: {aresp}")
    return aresp.solution if aresp is not None else None


def sample_vote(
    task_id: str,
    task: str,
    lm: lf.LanguageModel,
    num_samples: int,
) -> Solution | None:
    logger.info(f"Start task {task_id}")

    map_iterator = lf.concurrent_map(
        func=lambda task: sample(task, lm),
        parallel_inputs=[task] * num_samples,
        show_progress=True,
    )

    sols = []
    for _, output, error in map_iterator:
        if error:
            logger.warning(f"Error response: {error}")
        elif output is not None:
            sols.append(output)

    logger.info(f"Solutions before majority vote: {sols}")

    vote = majority_vote_or_random(sols, lm)

    logger.info(f"End task {task_id}: {vote}")
    return vote


def sample_eval(
    task_id: str,
    task: str,
    lm: lf.LanguageModel,
    num_samples: int,
) -> Solution | None:
    logger.info(f"Start task {task_id}")

    map_iterator = lf.concurrent_map(
        func=lambda task: sample(task, lm),
        parallel_inputs=[task] * num_samples,
        show_progress=True,
    )

    sols = []
    for _, output, error in map_iterator:
        if error:
            logger.warning(f"Error response: {error}")
        elif output is not None:
            sols.append(output)

    mresp = multiple_choice(task, sols, lm)

    logger.info(f"End task {task_id}: {mresp}")
    return mresp.solution if mresp is not None else None


def sample_verify(
    task_id: str,
    task: str,
    lm: lf.LanguageModel,
    num_samples: int,
    num_retries: int,
) -> Solution | None:
    logger.info(f"Start task {task_id}")

    map_iterator = lf.concurrent_map(
        func=lambda tinfo: sample_verify_correct(*tinfo, lm, num_retries),
        parallel_inputs=[(task_id, task)] * num_samples,
        show_progress=True,
    )

    sols = []
    for _, output, error in map_iterator:
        if error:
            logger.warning(f"Error response: {error}")
        elif output is not None:
            sols.append(output)

    verified = []
    unverified = []
    for sol, success, _ in sols:
        if success:
            verified.append(sol)
        else:
            unverified.append(sol)

    logger.info(f"Verified Solutions before majority vote: {verified}")
    logger.info(f"Unverified Solutions before majority vote: {unverified}")

    mcsols = verified or unverified
    vote = majority_vote_or_random(mcsols, lm)

    logger.info(f"End task {task_id}: {vote}")
    return vote
