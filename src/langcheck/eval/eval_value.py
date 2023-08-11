from dataclasses import dataclass
from typing import List, Optional

@dataclass
class EvalValue:
    '''A rich object that is the output of any langcheck.eval function.'''
    metric_name: str
    prompts: Optional[List[str]]
    generated_outputs: List[str]
    metric_values: List[float]