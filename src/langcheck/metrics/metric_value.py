from __future__ import annotations

import operator
import warnings
from dataclasses import dataclass, fields
from statistics import mean
from typing import Generic, TypeVar, Union

import pandas as pd

from langcheck.metrics.metric_inputs import MetricInputs

# Metrics take on float or integer values
# Some metrics may return `None` values when the score fails to be computed
NumericType = TypeVar(
    "NumericType", float, int, Union[float, None], Union[int, None]
)


@dataclass
class MetricValue(Generic[NumericType]):
    """A rich object that is the output of any langcheck.metrics function."""

    metric_name: str
    metric_values: list[NumericType]

    # Input of the metrics such as prompts, generated outputs... etc
    metric_inputs: MetricInputs

    # An explanation can be None if the metric could not be computed
    explanations: list[str | None] | None
    language: str | None

    def to_df(self) -> pd.DataFrame:
        """Returns a DataFrame of metric values for each data point."""
        input_df = self.metric_inputs.to_df()
        output_df = pd.DataFrame(
            {
                "explanations": self.explanations,
                "metric_values": self.metric_values,
            }
        )

        # Return the concatenation of the input and output DataFrames
        return pd.concat([input_df, output_df], axis=1)

    def __str__(self) -> str:
        """Returns a string representation of an
        :class:`~langcheck.metrics.metric_value.MetricValue` object.
        """
        return f"Metric: {self.metric_name}\n" f"{self.to_df()}"

    def __repr__(self) -> str:
        """Returns a string representation of an
        :class:`~langcheck.metrics.metric_value.MetricValue` object.
        """
        return str(self)

    def _repr_html_(self) -> str:
        """Returns an HTML representation of an
        :class:`~langcheck.metrics.metric_value.MetricValue`, which is
        automatically called by Jupyter notebooks.
        """
        return (
            f"Metric: {self.metric_name}<br>" f"{self.to_df()._repr_html_()}"  # type: ignore
        )

    def __lt__(self, threshold: float | int) -> MetricValueWithThreshold:
        """Allows the user to write a `metric_value < 0.5` expression."""
        all_fields = {f.name: getattr(self, f.name) for f in fields(self)}
        return MetricValueWithThreshold(
            **all_fields, threshold=threshold, threshold_op="<"
        )

    def __le__(self, threshold: float | int) -> MetricValueWithThreshold:
        """Allows the user to write a `metric_value <= 0.5` expression."""
        all_fields = {f.name: getattr(self, f.name) for f in fields(self)}
        return MetricValueWithThreshold(
            **all_fields, threshold=threshold, threshold_op="<="
        )

    def __gt__(self, threshold: float | int) -> MetricValueWithThreshold:
        """Allows the user to write a `metric_value > 0.5` expression."""
        all_fields = {f.name: getattr(self, f.name) for f in fields(self)}
        return MetricValueWithThreshold(
            **all_fields, threshold=threshold, threshold_op=">"
        )

    def __ge__(self, threshold: float | int) -> MetricValueWithThreshold:
        """Allows the user to write a `metric_value >= 0.5` expression."""
        all_fields = {f.name: getattr(self, f.name) for f in fields(self)}
        return MetricValueWithThreshold(
            **all_fields, threshold=threshold, threshold_op=">="
        )

    def __eq__(self, threshold: float | int) -> MetricValueWithThreshold:
        """Allows the user to write a `metric_value == 0.5` expression."""
        all_fields = {f.name: getattr(self, f.name) for f in fields(self)}
        return MetricValueWithThreshold(
            **all_fields, threshold=threshold, threshold_op="=="
        )

    def __ne__(self, threshold: float | int) -> MetricValueWithThreshold:
        """Allows the user to write a `metric_value != 0.5` expression."""
        all_fields = {f.name: getattr(self, f.name) for f in fields(self)}
        return MetricValueWithThreshold(
            **all_fields, threshold=threshold, threshold_op="!="
        )

    def all(self) -> bool:
        """Equivalent to all(metric_value.metric_values). This is mostly useful
        for binary metric functions.
        """
        return all(self.metric_values)

    def any(self) -> bool:
        """Equivalent to any(metric_value.metric_values). This is mostly useful
        for binary metric functions.
        """
        return any(self.metric_values)

    def __bool__(self):
        raise ValueError(
            "A MetricValue cannot be used as a boolean. "
            "Try an expression like `metric_value > 0.5`, "
            "`metric_value.all()`, or `metric_value.any()` instead."
        )

    def __getattr__(self, name: str):
        """If the attribute is not found in the MetricValue object, we try to
        proxy the attribute to the MetricInputs object.
        """
        try:
            return self.metric_inputs.get_input_list(name)
        except ValueError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    @property
    def is_scatter_compatible(self) -> bool:
        """Checks if the metric value is compatible with the scatter plot
        method. It is only available for metric values with only non-pairwise
        metric values used from initial release (generated_outputs, prompts,
        reference_outputs and sources)
        """
        allowed_inputs = [
            "generated_outputs",
            "prompts",
            "reference_outputs",
            "sources",
        ]
        return len(self.metric_inputs.pairwise_inputs) == 0 and all(
            input_name in allowed_inputs
            for input_name in self.metric_inputs.individual_inputs
        )

    def scatter(self, jupyter_mode: str = "inline") -> None:
        """Shows an interactive scatter plot of all data points in MetricValue.
        Intended to be used in a Jupyter notebook.

        This is a convenience function that calls
        :func:`langcheck.plot.scatter()`.
        """
        from langcheck.plot import scatter as plot_scatter

        # Type ignore because a Self type is only valid in class contexts
        return plot_scatter(
            self,  # type: ignore[reportGeneralTypeIssues]
            jupyter_mode=jupyter_mode,
        )

    def histogram(self, jupyter_mode: str = "inline") -> None:
        """Shows an interactive histogram of all data points in MetricValue.
        Intended to be used in a Jupyter notebook.

        This is a convenience function that calls
        :func:`langcheck.plot.histogram()`.
        """
        from langcheck.plot import histogram as plot_histogram

        # Type ignore because a Self type is only valid in class contexts
        return plot_histogram(
            self,  # type: ignore[reportGeneralTypeIssues]
            jupyter_mode=jupyter_mode,
        )


@dataclass
class MetricValueWithThreshold(MetricValue):
    """A rich object that is the output of comparing an
    :class:`~langcheck.metrics.metric_value.MetricValue` object,
    e.g. `metric_value >= 0.5`.
    """

    threshold: float | int
    threshold_op: str  # One of '<', '<=', '>', '>=', '==', '!='

    def __post_init__(self) -> None:
        """Computes self.pass_rate and self.threshold_results based on the
        constructor arguments.
        """
        operators = {
            "<": operator.lt,
            "<=": operator.le,
            ">": operator.gt,
            ">=": operator.ge,
            "==": operator.eq,
            "!=": operator.ne,
        }

        if self.threshold_op not in operators:
            raise ValueError(f"Invalid threshold operator: {self.threshold_op}")

        if self.threshold is None:
            raise ValueError("A threshold of `None` is not supported.")

        if None in self.metric_values:
            warnings.warn(
                "The threshold result for `None` values in `metric_values` will"
                " always be `False`."
            )

        # Set the result to `False` if the metric value is `None`
        self._threshold_results = [
            operators[self.threshold_op](x, self.threshold)
            if x is not None
            else False
            for x in self.metric_values
        ]

        self._pass_rate = mean(self._threshold_results)

    @property
    def pass_rate(self) -> float:
        """Returns the proportion of data points that pass the threshold."""
        return self._pass_rate

    @property
    def threshold_results(self) -> list[bool]:
        """Returns a list of booleans indicating whether each data point passes
        the threshold.
        """
        return self._threshold_results

    def to_df(self) -> pd.DataFrame:
        """Returns a DataFrame of metric values for each data point."""
        dataframe = super().to_df()

        dataframe["threshold_test"] = [
            f"{self.threshold_op} {self.threshold}" for _ in self.metric_values
        ]
        dataframe["threshold_result"] = self.threshold_results

        return dataframe

    def __str__(self) -> str:
        """Returns a string representation of an
        :class:`~langcheck.metrics.metric_value.MetricValue`.
        """
        return (
            f"Metric: {self.metric_name}\n"
            f"Pass Rate: {round(self.pass_rate*100, 2)}%\n"
            f"{self.to_df()}"
        )

    def __repr__(self) -> str:
        """Returns a string representation of an
        :class:`~langcheck.metrics.metric_value.MetricValue` object.
        """
        return str(self)

    def _repr_html_(self) -> str:
        """Returns an HTML representation of an
        :class:`~langcheck.metrics.metric_value.MetricValue`, which is
        automatically called by Jupyter notebooks.
        """
        return (
            f"Metric: {self.metric_name}<br>"
            f"Pass Rate: {round(self.pass_rate*100, 2)}%<br>"
            f"{self.to_df()._repr_html_()}"  # type: ignore
        )

    def all(self) -> bool:
        """Returns True if all data points pass the threshold."""
        return all(self.threshold_results)

    def any(self) -> bool:
        """Returns True if any data points pass the threshold."""
        return any(self.threshold_results)

    def __bool__(self) -> bool:
        """Allows the user to write an `assert metric_value > 0.5` or
        `if metric_value > 0.5:` expression. This is shorthand for
        `assert (metric_value > 0.5).all()`.
        """
        return self.all()
