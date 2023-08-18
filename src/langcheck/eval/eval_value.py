from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar

# Metrics usually take on float or integer values
MetricValue = TypeVar('MetricValue')


@dataclass
class EvalValue(Generic[MetricValue]):
    '''A rich object that is the output of any langcheck.eval function.'''
    metric_name: str
    prompts: Optional[List[str]]
    generated_outputs: List[str]
    metric_values: List[MetricValue]