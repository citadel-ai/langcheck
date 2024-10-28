from __future__ import annotations

from typing import Union

import pandas as pd
from jinja2 import Environment, meta

# You need "Union" to declare a type in Python < 3.10
IndividualInputType = Union[str, list[str], None]


def _map_pairwise_input_to_list(
    input: tuple[IndividualInputType, IndividualInputType],
) -> tuple[list[str] | None, list[str] | None]:
    return (
        _map_individual_input_to_list(input[0]),
        _map_individual_input_to_list(input[1]),
    )


def _map_individual_input_to_list(
    input: IndividualInputType,
) -> list[str] | None:
    if input is None:
        return None
    elif isinstance(input, str):
        return [input]
    else:
        return input


class MetricInputs:
    """A helper class to handle the inputs for the metric in a consistent way."""

    def __init__(
        self,
        individual_inputs: dict[str, IndividualInputType],
        pairwise_inputs: dict[
            str, tuple[IndividualInputType, IndividualInputType]
        ]
        | None = None,
        required_params: list[str] | None = None,
        optional_params: list[str] | None = None,
        input_name_to_prompt_var_mapping: dict[str, str] | None = None,
    ):
        """Initialize the MetricInputs object.

        Args:
            individual_inputs: A dictionary of individual inputs. The keys are
                the parameter names and the values are the input lists.
            pairwise_inputs: A dictionary of pairwise inputs. The keys are the
                parameter names and the values are tuples of two input lists.
            required_params: A list of required parameters.
            optional_params: A list of optional parameters.
            input_name_to_prompt_var_mapping: A dictionary that maps the input
                names to the variable names in the prompt template. The values
                should therefore correspond with the keys returned from the
                `get_inputs_for_prompt_template` method.
        """
        # Instantiate the parameter lists if None
        self.required_params = required_params or []
        self.optional_params = optional_params or []

        self.individual_inputs = {
            key: _map_individual_input_to_list(value)
            for key, value in individual_inputs.items()
        }

        if pairwise_inputs is None:
            self.pairwise_inputs = {}
        else:
            self.pairwise_inputs = {
                key: _map_pairwise_input_to_list(value)
                for key, value in pairwise_inputs.items()
            }

        self.input_name_to_prompt_var_mapping = (
            input_name_to_prompt_var_mapping or {}
        )

        all_input_keys = list(self.individual_inputs.keys()) + list(
            self.pairwise_inputs.keys()
        )
        # Check that all the required parameters are present
        missing_required_params = set(self.required_params) - set(
            all_input_keys
        )
        if missing_required_params:
            raise ValueError(
                f"Missing required parameters: {missing_required_params}"
            )

        for input_key in all_input_keys:
            if input_key not in self.input_name_to_prompt_var_mapping:
                # Add mapping for the input key itself
                self.input_name_to_prompt_var_mapping[input_key] = input_key

        # Do the validation of parameters
        # Validate that individual_inputs and pairwise_inputs are disjoint
        individual_input_keys = set(self.individual_inputs.keys())
        pairwise_input_keys = set(self.pairwise_inputs.keys())
        if not individual_input_keys.isdisjoint(pairwise_input_keys):
            overlap_keys = individual_input_keys.intersection(
                pairwise_input_keys
            )
            raise ValueError(
                "Individual input keys and pairwise input keys should be disjoint."
                f" Overlapping keys: {overlap_keys}"
            )

        # Validate the individual inputs
        for individual_input_key in individual_input_keys:
            individual_input = self.individual_inputs[individual_input_key]
            if individual_input_key in self.required_params:
                if individual_input is None:
                    raise ValueError(
                        f"Required parameter '{individual_input_key}' is None."
                    )
            elif individual_input_key not in self.optional_params:
                raise ValueError(f"Unknown parameter '{individual_input_key}'")

        # Validate the pairwise inputs
        for pairwise_input_key in pairwise_input_keys:
            pairwise_input_a, pairwise_input_b = self.pairwise_inputs[
                pairwise_input_key
            ]
            if pairwise_input_key in self.required_params:
                if pairwise_input_a is None or pairwise_input_b is None:
                    raise ValueError(
                        f"Required parameter '{pairwise_input_key}' is None."
                    )
            elif pairwise_input_key not in self.optional_params:
                raise ValueError(f"Unknown parameter '{pairwise_input_key}'")

            # If to_df is called, each key is mapped into two columns: key_a and
            # key_b. Check that the key is not already used.
            df_key_a = pairwise_input_key + "_a"
            if df_key_a in all_input_keys:
                raise ValueError(
                    f"Key '{df_key_a} will be added as a dataframe column, but it is already used as a input key."
                )
            df_key_b = pairwise_input_key + "_b"
            if df_key_b in all_input_keys:
                raise ValueError(
                    f"Key '{df_key_b} will be added as a dataframe column, but it is already used as a input key."
                )

        # Validate the lengths of the inputs
        input_lengths: set[int] = set()
        for key in self.individual_inputs:
            individual_input = self.individual_inputs[key]
            if individual_input is not None:
                input_lengths.add(len(individual_input))

        for key in self.pairwise_inputs:
            pairwise_input_a, pairwise_input_b = self.pairwise_inputs[key]
            if pairwise_input_a is not None:
                input_lengths.add(len(pairwise_input_a))
            if pairwise_input_b is not None:
                input_lengths.add(len(pairwise_input_b))

        if len(input_lengths) > 1:
            individual_input_lengths = "\n".join(
                f"{key}: {len(value)}"
                for key, value in self.individual_inputs.items()
                if value is not None
            )

            pairwise_input_lengths = "\n".join(
                f"{key}: ({len(value[0])}, {len(value[1])})"
                for key, value in self.pairwise_inputs.items()
                if value[0] is not None and value[1] is not None
            )

            raise ValueError(
                f"All inputs should have the same length.\n{individual_input_lengths}\n{pairwise_input_lengths}"
            )

        if not input_lengths:
            raise ValueError("No inputs provided.")
        self.input_length = input_lengths.pop()
        if self.input_length == 0:
            raise ValueError("All inputs should have at least one element.")

        # Validate the mapping to prompt variables
        self.prompt_var_to_input_name_mapping = {}

        for individual_input_key in individual_input_keys:
            prompt_var = self.input_name_to_prompt_var_mapping[
                individual_input_key
            ]
            if prompt_var in self.prompt_var_to_input_name_mapping:
                raise ValueError(
                    f"Prompt variable '{prompt_var}' is mapped from multiple arguments: "
                    f"{self.prompt_var_to_input_name_mapping[prompt_var]} and {individual_input_key}"
                )

            self.prompt_var_to_input_name_mapping[prompt_var] = (
                individual_input_key
            )

        for pairwise_input_key in pairwise_input_keys:
            prompt_var_individual = self.input_name_to_prompt_var_mapping[
                pairwise_input_key
            ]
            prompt_vars = [
                prompt_var_individual + "_a",
                prompt_var_individual + "_b",
            ]

            for prompt_var in prompt_vars:
                if prompt_var in self.prompt_var_to_input_name_mapping:
                    raise ValueError(
                        f"Prompt variable '{prompt_var}' is mapped from multiple arguments: "
                        f"{self.prompt_var_to_input_name_mapping[prompt_var]} and {pairwise_input_key}"
                    )

                self.prompt_var_to_input_name_mapping[prompt_var] = (
                    pairwise_input_key
                )

    def get_inputs_for_prompt_template(
        self, swap_pairwise: bool = False
    ) -> list[dict[str, str | None]]:
        """Get the inputs that can be used as arguments for the prompt
        template.
        Each item is a dictionary where the keys are the prompt variables
        specified in the `input_name_to_prompt_var_mapping` and the values are
        the input values, which are corresponding elements from the input lists.
        For pairwise inputs, the values for the first list and the second list
        are stored in the attributes with the suffixes "_a" and "_b".

        Args:
            swap_pairwise: If True, swap the pairwise inputs.
        """
        inputs_for_prompt_template: list[dict[str, str | None]] = []
        for i in range(self.input_length):
            # Create the inputs for the prompt template for the i-th input
            single_instance_inputs = {}
            for individual_key in self.individual_inputs:
                individual_input = self.individual_inputs[individual_key]
                individual_prompt_var = self.input_name_to_prompt_var_mapping[
                    individual_key
                ]
                if individual_input is None:
                    single_instance_inputs[individual_prompt_var] = None
                else:
                    single_instance_inputs[individual_prompt_var] = (
                        individual_input[i]
                    )

            for pairwise_key in self.pairwise_inputs:
                pairwise_input_a, pairwise_input_b = self.pairwise_inputs[
                    pairwise_key
                ]
                if swap_pairwise:
                    pairwise_input_a, pairwise_input_b = (
                        pairwise_input_b,
                        pairwise_input_a,
                    )
                pairwise_prompt_var_a = (
                    self.input_name_to_prompt_var_mapping[pairwise_key] + "_a"
                )
                if pairwise_input_a is None:
                    single_instance_inputs[pairwise_prompt_var_a] = None
                else:
                    single_instance_inputs[pairwise_prompt_var_a] = (
                        pairwise_input_a[i]
                    )
                pairwise_prompt_var_b = (
                    self.input_name_to_prompt_var_mapping[pairwise_key] + "_b"
                )
                if pairwise_input_b is None:
                    single_instance_inputs[pairwise_prompt_var_b] = None
                else:
                    single_instance_inputs[pairwise_prompt_var_b] = (
                        pairwise_input_b[i]
                    )
            inputs_for_prompt_template.append(single_instance_inputs)

        return inputs_for_prompt_template

    def to_df(self) -> pd.DataFrame:
        """Convert the inputs to a DataFrame."""
        input_lists = {}
        for individual_key in self.individual_inputs:
            individual_input = self.individual_inputs[individual_key]
            if individual_input is None:
                input_lists[individual_key] = [None] * self.input_length
            else:
                input_lists[individual_key] = individual_input

        for pairwise_key in self.pairwise_inputs:
            pairwise_input_a, pairwise_input_b = self.pairwise_inputs[
                pairwise_key
            ]
            if pairwise_input_a is None:
                input_lists[pairwise_key + "_a"] = [None] * self.input_length
            else:
                input_lists[pairwise_key + "_a"] = pairwise_input_a

            if pairwise_input_b is None:
                input_lists[pairwise_key + "_b"] = [None] * self.input_length
            else:
                input_lists[pairwise_key + "_b"] = pairwise_input_b

        return pd.DataFrame(input_lists)

    def get_input_list(
        self, key: str
    ) -> tuple[list[str] | None, list[str] | None] | list[str] | None:
        """Get the input list for the key."""
        if key in self.individual_inputs:
            return self.individual_inputs[key]
        elif key in self.pairwise_inputs:
            return self.pairwise_inputs[key]
        else:
            raise ValueError(f"Unknown key: {key}")

    def validate_template(self, template_src: str):
        """Validate that the given prompt template string is compatible with
        the input parameters.

        Args:
            template_src: The prompt template string.
        """
        # Validate the expected parameters in the prompt template
        env = Environment()
        expected_params = meta.find_undeclared_variables(
            env.parse(template_src)
        )

        allowed_params = self.prompt_var_to_input_name_mapping.keys()
        assert all(
            param in allowed_params for param in expected_params
        ), f"The prompt template contains invalid parameters. The allowed parameters are {allowed_params} but the prompt template expects the parameters {expected_params}"

        for param in expected_params:
            arg_key = self.prompt_var_to_input_name_mapping[param]
            if arg_key in self.individual_inputs:
                assert (
                    self.individual_inputs[arg_key] is not None
                ), f'The prompt template expects the parameter "{param}" but it is not provided.'
            else:
                pairwise_inputs_a, pairwise_inputs_b = self.pairwise_inputs[
                    arg_key
                ]
                assert (
                    pairwise_inputs_a is not None
                ), f'The prompt template expects the parameter "{param}_a" but it is not provided.'
                assert (
                    pairwise_inputs_b is not None
                ), f'The prompt template expects the parameter "{param}_b" but it is not provided.'

    def get_required_individual_input(self, key: str) -> list[str]:
        """Get the list of a required parameter in individual_inputs.
        Mainly used for metrics without eval clients.
        """
        if (key not in self.individual_inputs) or (
            key not in self.required_params
        ):
            raise ValueError(f"Unknown key: {key}")

        individual_input = self.individual_inputs[key]

        # It is already validated that the input is not None
        assert isinstance(individual_input, list)

        return individual_input


def get_metric_inputs(
    *,
    generated_outputs: IndividualInputType
    | tuple[IndividualInputType, IndividualInputType] = None,
    prompts: IndividualInputType
    | tuple[IndividualInputType, IndividualInputType] = None,
    sources: IndividualInputType
    | tuple[IndividualInputType, IndividualInputType] = None,
    reference_outputs: IndividualInputType
    | tuple[IndividualInputType, IndividualInputType] = None,
    additional_inputs: dict[str, IndividualInputType] | None = None,
    additional_input_name_to_prompt_var_mapping: dict[str, str] | None = None,
    required_params: list[str],
) -> MetricInputs:
    """Create a metric inputs object with the standard parameters
    (i.e. generated_outputs, prompts, sources, reference_outputs) and the
    specified additional parameters.

    Args:
        generated_outputs: The generated outputs.
        prompts: The prompts.
        sources: The sources.
        reference_outputs: The reference outputs.
        additional_inputs: Additional inputs other than the standard ones.
        additional_input_name_to_prompt_var_mapping: A dictionary that maps the
            additional input names to the variable names in the prompt template.
        required_params: A list of required parameters.
    Returns:
        A MetricInputs object.
    """
    if additional_inputs is None:
        additional_inputs = {}
    if additional_input_name_to_prompt_var_mapping is None:
        additional_input_name_to_prompt_var_mapping = {}

    allowed_params = [
        "generated_outputs",
        "prompts",
        "sources",
        "reference_outputs",
    ] + list(additional_inputs.keys())
    for param in required_params:
        if param not in allowed_params:
            raise ValueError(f"Unknown parameter: {param}")

    optional_params = list(set(allowed_params) - set(required_params))
    all_inputs = {
        "generated_outputs": generated_outputs,
        "prompts": prompts,
        "sources": sources,
        "reference_outputs": reference_outputs,
        **additional_inputs,
    }
    # Split individual and pairwise inputs
    individual_inputs = {
        key: value
        for key, value in all_inputs.items()
        if not isinstance(value, tuple)
    }
    pairwise_inputs = {
        key: value
        for key, value in all_inputs.items()
        if isinstance(value, tuple)
    }
    return MetricInputs(
        individual_inputs=individual_inputs,
        pairwise_inputs=pairwise_inputs,
        required_params=required_params,
        optional_params=optional_params,
        input_name_to_prompt_var_mapping={
            "generated_outputs": "gen_output",
            "prompts": "user_query",
            "sources": "src",
            "reference_outputs": "ref_output",
            **additional_input_name_to_prompt_var_mapping,
        },
    )


def get_metric_inputs_with_required_lists(
    *,
    generated_outputs: IndividualInputType
    | tuple[IndividualInputType, IndividualInputType] = None,
    prompts: IndividualInputType
    | tuple[IndividualInputType, IndividualInputType] = None,
    sources: IndividualInputType
    | tuple[IndividualInputType, IndividualInputType] = None,
    reference_outputs: IndividualInputType
    | tuple[IndividualInputType, IndividualInputType] = None,
    additional_inputs: dict[str, IndividualInputType] | None = None,
    additional_input_name_to_prompt_var_mapping: dict[str, str] | None = None,
    required_params: list[str],
) -> tuple[MetricInputs, list[list[str]]]:
    """Create a metric inputs object with the standard parameters
    (i.e. generated_outputs, prompts, sources, reference_outputs) and the
    specified additional parameters. This function also returns the list of
    required parameters as raw lists, which is useful for metrics without eval
    clients.

    Args:
        generated_outputs: The generated outputs.
        prompts: The prompts.
        sources: The sources.
        reference_outputs: The reference outputs.
        additional_inputs: Additional inputs other than the standard ones.
        additional_input_name_to_prompt_var_mapping: A dictionary that maps the
            additional input names to the variable names in the prompt template.
        required_params: A list of required parameters.

    Returns:
        A MetricInputs object and the required lists.
    """
    metric_inputs = get_metric_inputs(
        generated_outputs=generated_outputs,
        prompts=prompts,
        sources=sources,
        reference_outputs=reference_outputs,
        additional_inputs=additional_inputs,
        additional_input_name_to_prompt_var_mapping=additional_input_name_to_prompt_var_mapping,
        required_params=required_params,
    )

    required_lists = [
        metric_inputs.get_required_individual_input(param)
        for param in required_params
    ]
    return metric_inputs, required_lists
