from schema import Solution, AnalyticalResponse, CorrectionResponse
from prompt import SamplePrompt, VerifyPrompt, CorrectPrompt
from schema import examples, task, test_case_str
from local_models import Qwen25_7B_Instruct
import langfun as lf
import unittest
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
        logger.info("Got majority vote")
        return vote
    else:
        logger.info("Picking majority vote in random")
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

    schema_items = [p[0] for p in sols if p is not None]
    vote = majority_vote_or_random(
        schema=Solution,
        schema_title='Solution',
        schema_items=schema_items,
        lm=lm,
    )

    return vote, sols


def eval_func(sol, test_case_str):
    full_code = sol.source + '\n' + test_case_str
    global_ns = {}
    exec(full_code, global_ns)

    test_cases = global_ns['TestCases']
    logger.info(type(test_cases))

    suite = unittest.TestLoader().loadTestsFromTestCase(test_cases)
    unittest.TextTestRunner().run(suite)


if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)

    log_sample_prompts()

    lm = Qwen25_7B_Instruct()
    sol, sols = sample_verify_correct(task, lm, num_samples=3, num_retries=3)

    logger.info(f"Solutions: {sols}")
    logger.info(f"Majority Vote: {sol}")

    eval_func(sol, test_case_str)
