from local_models import Qwen25_7B_Instruct
import langfun as lf
import pyglove as pg


question = (
    'Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. '
    'She sells the remainder at the farmers\' market daily for $2 per fresh duck egg. '
    'How much in dollars does she make every day at the farmers\' market?')


class Step(pg.Object):
    description: str
    step_output: float


class Solution(pg.Object):
    steps: list[Step]
    final_answer: int


r = lf.query(prompt=question, schema=Solution, lm=Qwen25_7B_Instruct())
print(r)
