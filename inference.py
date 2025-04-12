from prompt import SamplePrompt, VerifyPrompt, CorrectPrompt, MultipleChoicePrompt
from schema import Solution, AnalyticalResponse, CorrectionResponse
from schema import MultipleChoiceResponse, examples
import langfun as lf
import traceback
import logging
import random

logger = logging.getLogger(__name__)


def query_with_exception(*args, **kwargs):
    try:
        return lf.query(*args, **kwargs)
    except Exception:
        logger.warning(f"Exception during query: {traceback.format_exc()}")
        return None


def sample(task: str, lm: lf.LanguageModel) -> AnalyticalResponse | None:

    return query_with_exception(
        prompt=SamplePrompt(
            examples=examples,
            input=task,
        ),
        lm=lm,
        schema=AnalyticalResponse,
        response_postprocess=AnalyticalResponse.clean_resp,
        default=None,
    )


def verify(task: str, solution: Solution, lm: lf.LanguageModel) -> str | None:

    analysis = query_with_exception(
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

    return query_with_exception(
        prompt=CorrectPrompt(
            examples=examples,
            input=task,
            solution=solution,
            analysis=analysis,
        ),
        schema=CorrectionResponse,
        response_postprocess=CorrectionResponse.clean_resp,
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

    if len(solutions) == 1:
        logger.info("Only one provided solution")
        return MultipleChoiceResponse(
            thought="Only one provided solution",
            solution=solutions[0],
        )

    with lm.sampling_options.override(temperature=0):
        choice = query_with_exception(
            prompt=MultipleChoicePrompt(
                examples=examples,
                input=task,
                solutions=solutions,
            ),
            schema=MultipleChoiceResponse,
            response_postprocess=MultipleChoiceResponse.clean_resp,
            lm=lm,
            default=None,
        )

    if choice is not None:
        logger.info("Got multiple choice")
        return choice
    else:
        logger.info("Randomly choosing multiple choice")
        return MultipleChoiceResponse(
            thought="Randomly chosen solution",
            solution=random.choice(solutions),
        )


def majority_vote(
    solutions: list[Solution],
    lm: lf.LanguageModel,
) -> Solution | None:
    if not solutions:
        return None

    if len(solutions) == 1:
        logger.info("Only one provided solution")
        return solutions[0]

    vote = query_with_exception(
        prompt="What is the majority {{schema_title}} from {{schema_items}}",
        schema=Solution,
        schema_title='Solution',
        schema_items=solutions,
        response_postprocess=Solution.clean_resp,
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
    task: str,
    lm: lf.LanguageModel,
    num_retries: int
) -> (Solution | None, bool, int):
    attempts = -1
    aresp = sample(task, lm)
    logger.debug(f"AnalyticalResponse: {aresp}")

    if aresp is None:
        logger.critical("Found None analytical response")
        return None, False, attempts

    solution = aresp.solution

    analysis, success = verify(task, solution, lm)
    logger.debug(f"Verification success: {success}, analysis: {analysis}")
    attempts += 1

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
            logger.debug(
                f"Verification success: {success}, analysis: {analysis}"
            )

            if analysis is None:
                logger.critical("Found None verification analysis")
                return solution, success, attempts

            elif success:
                logger.info(
                    f"Verified solution in {attempts} attempts: {solution}"
                )
                return solution, success, attempts

        return solution, success, attempts


def sample_once(
    task_id: str,
    task: str,
    lm: lf.LanguageModel,
) -> dict[str, Solution | None]:
    logger.info(f"Start task {task_id}")
    aresp = sample(task, lm)

    logger.info(f"End task {task_id}: {aresp}")
    return {
        'solution': aresp.solution if aresp is not None else None,
    }


def sample_vote(
    task_id: str,
    task: str,
    lm: lf.LanguageModel,
    num_samples: int,
) -> dict[str, Solution | None]:
    logger.info(f"Start task {task_id}")

    map_iterator = lf.concurrent_map(
        func=lambda task: sample(task, lm),
        parallel_inputs=[task] * num_samples,
    )

    sols = []
    for _, aresp, error in map_iterator:
        if error:
            logger.warning(f"Error response: {error}")
        elif aresp is not None:
            sols.append(aresp.solution)

    logger.info(f"Solutions before majority vote ({len(sols)}): {sols}")

    sol = majority_vote(sols, lm)

    logger.info(f"End task {task_id}: {sol}")
    return {
        'solution': sol,
    }


def sample_eval(
    task_id: str,
    task: str,
    lm: lf.LanguageModel,
    num_samples: int,
) -> dict[str, Solution | None]:
    logger.info(f"Start task {task_id}")

    map_iterator = lf.concurrent_map(
        func=lambda task: sample(task, lm),
        parallel_inputs=[task] * num_samples,
    )

    sols = []
    for _, aresp, error in map_iterator:
        if error:
            logger.warning(f"Error response: {error}")
        elif aresp is not None:
            sols.append(aresp.solution)

    logger.info(f"Solutions before multiple choice ({len(sols)}): {sols}")
    mresp = multiple_choice(task, sols, lm)

    logger.info(f"End task {task_id}: {mresp}")
    return {
        'solution': mresp.solution if mresp is not None else None,
    }


def sample_veco(
    task_id: str,
    task: str,
    lm: lf.LanguageModel,
    num_samples: int,
    num_retries: int,
) -> dict[str, Solution | None]:
    logger.info(f"Start task {task_id}")

    map_iterator = lf.concurrent_map(
        func=lambda task: sample_verify_correct(task, lm, num_retries),
        parallel_inputs=[task] * num_samples,
    )

    ver_sols = []
    unver_sols = []
    details = []
    for _, triple, error in map_iterator:
        if error:
            logger.warning(f"Error response: {error}")
        sol, success, attempts = triple

        details.append((success, attempts))

        if sol is None:
            continue

        if success:
            ver_sols.append(sol)
        else:
            unver_sols.append(sol)

    logger.info(f"Verified Solutions before majority vote "
                f"({len(ver_sols)}): {ver_sols}")

    logger.info(f"Unverified Solutions before majority vote "
                f"({len(unver_sols)}): {unver_sols}")

    sols = ver_sols or unver_sols
    sol = majority_vote(sols, lm)

    logger.info(f"End task {task_id}: {sol}")
    return {
        'solution': sol,
        'details': details,
    }
