from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Optional, TypeVar


@dataclass
class MetricTokenUsage:
    """
    A dataclass that contains the token usage information for a metric.

    Args:
        input_token_count: The number of input tokens used.
        output_token_count: The number of output tokens used.
        input_token_cost: The cost of the input tokens.
        output_token_cost: The cost of the output tokens.
    """

    input_token_count: int | None = None
    output_token_count: int | None = None
    input_token_cost: float | None = None
    output_token_cost: float | None = None

    def __add__(self, other: MetricTokenUsage) -> MetricTokenUsage:
        """
        Add two MetricTokenUsage objects together.

        Args:
            other: The MetricTokenUsage object to add to the current object.

        Returns:
            A new MetricTokenUsage object with the sum of the input and output token counts and costs.
        """

        def add_or_keep(a, b):
            if a is not None and b is not None:
                return a + b
            return a if a is not None else b

        if other is None:
            return self

        return MetricTokenUsage(
            add_or_keep(self.input_token_count, other.input_token_count),
            add_or_keep(self.output_token_count, other.output_token_count),
            add_or_keep(self.input_token_cost, other.input_token_cost),
            add_or_keep(self.output_token_cost, other.output_token_cost),
        )


T = TypeVar("T")


class ResponsesWithTokenUsage(list[Optional[T]], Generic[T]):
    """
    A backward-compatible list subclass that carries additional token
    usage information.

    This class extends the built-in `list` to preserve existing behavior for
    callers that expect a plain list, ensuring backward compatibility after the
    function's return type is expanded.

    The motivation is to allow returning both the original list data and
    token usage information without breaking existing code that iterates over or
    mutates the list.

    Example:
        >>> responses = fn()
        >>> responses.append("new_item")   # still works like a list
        >>> responses.token_usage          # token usage information is available
    """

    token_usage: MetricTokenUsage | None = None

    def __init__(
        self,
        response_texts: list[T | None],
        token_usage: MetricTokenUsage | None,
    ):
        super().__init__(response_texts)
        self.token_usage = token_usage
