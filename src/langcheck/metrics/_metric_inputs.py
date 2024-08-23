from __future__ import annotations

from typing import Union

SingleInputType = Union[str, list[str], None]


def _map_pairwise_input_to_list(
    input: tuple[SingleInputType, SingleInputType],
) -> tuple[list[str] | None, list[str] | None]:
    return (
        _map_single_input_to_list(input[0]),
        _map_single_input_to_list(input[1]),
    )


def _map_single_input_to_list(input: SingleInputType) -> list[str] | None:
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
        single_inputs: dict[str, SingleInputType],
        pairwise_inputs: dict[str, tuple[SingleInputType, SingleInputType]]
        | None,
        required_params: list[str] | None = None,
        optional_params: list[str] | None = None,
        input_record_mapping: dict[str, str] | None = None,
    ):
        """TODO"""
        # Instantiate the paramater lists if None
        self.required_params = required_params or []
        self.optional_params = optional_params or []

        self.single_inputs = {
            key: _map_single_input_to_list(value)
            for key, value in single_inputs.items()
        }

        if pairwise_inputs is None:
            self.pairwise_inputs = {}
        else:
            self.pairwise_inputs = {
                key: _map_pairwise_input_to_list(value)
                for key, value in pairwise_inputs.items()
            }

        self.input_record_mapping = input_record_mapping or {}

        all_input_keys = list(self.single_inputs.keys()) + list(
            self.pairwise_inputs.keys()
        )
        for input_key in all_input_keys:
            if input_key not in self.input_record_mapping:
                # Add mapping for the input key itself
                self.input_record_mapping[input_key] = input_key

        self.input_records: list[dict[str, str | None]] | None = None

    def validate(self):
        """TODO"""
        # Validate that single_inputs and pairwise_inputs are disjoint
        single_input_keys = set(self.single_inputs.keys())
        pairwise_input_keys = set(self.pairwise_inputs.keys())
        if not single_input_keys.isdisjoint(pairwise_input_keys):
            overlap_keys = single_input_keys.intersection(pairwise_input_keys)
            raise ValueError(
                "Single input keys and pairwise input keys should be disjoint."
                f" Overlapping keys: {overlap_keys}"
            )

        all_input_keys = list(self.single_inputs.keys()) + list(
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

        # Dictionary to check the collision in the prompt variable mapping
        input_record_to_arg = {}

        # Validate the single inputs
        for single_input_key in single_input_keys:
            single_input = self.single_inputs[single_input_key]
            if (
                single_input_key in self.required_params
                and single_input is None
            ):
                raise ValueError(
                    f"Required parameter '{single_input_key}' is None."
                )
            elif single_input_key not in self.optional_params:
                raise ValueError(f"Unknown parameter '{single_input_key}'")

            input_record_name = self.input_record_mapping[single_input_key]
            if input_record_name in input_record_to_arg:
                raise ValueError(
                    f"Prompt variable '{input_record_name}' is mapped from multiple arguments: "
                    f"{input_record_to_arg[input_record_name]} and {single_input_key}"
                )

            input_record_to_arg[input_record_name] = single_input_key

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
            elif pairwise_input_key in self.optional_params:
                # Raise an error if only one of the inputs is None
                if (pairwise_input_a is None) ^ (pairwise_input_b is None):
                    raise ValueError(
                        f"Both inputs of '{pairwise_input_key}' should be None or not None."
                    )
            else:
                raise ValueError(f"Unknown parameter '{pairwise_input_key}'")

            input_record_name_single = self.input_record_mapping[
                single_input_key
            ]
            input_record_names = [
                input_record_name_single + "_a",
                input_record_name_single + "_b",
            ]

            for input_record_name in input_record_names:
                if input_record_name in input_record_to_arg:
                    raise ValueError(
                        f"Prompt variable '{input_record_name}' is mapped from multiple arguments: "
                        f"{input_record_to_arg[input_record_name]} and {pairwise_input_key}"
                    )

                input_record_to_arg[input_record_name] = pairwise_input_key

        # Validate the lengths of the inputs
        input_lengths: set[int] = set()
        for key in self.single_inputs:
            single_input = self.single_inputs[key]
            if single_input is not None:
                input_lengths.add(len(single_input))

        for key in self.pairwise_inputs:
            pairwise_input_a, pairwise_input_b = self.pairwise_inputs[key]
            if pairwise_input_a is not None:
                input_lengths.add(len(pairwise_input_a))
            if pairwise_input_b is not None:
                input_lengths.add(len(pairwise_input_b))

        if len(input_lengths) > 1:
            single_input_lengths = "\n".join(
                f"{key}: {len(value)}"
                for key, value in self.single_inputs.items()
                if value is not None
            )

            pairwise_input_lengths = "\n".join(
                f"{key}: ({len(value[0])}, {len(value[1])})"
                for key, value in self.pairwise_inputs.items()
                if value[0] is not None and value[1] is not None
            )

            raise ValueError(
                f"All inputs should have the same length. {single_input_lengths}\n{pairwise_input_lengths}"
            )

        input_length = input_lengths.pop()
        if input_length == 0:
            raise ValueError("All inputs should have at least one element.")

        # Validate the mapping to prompt variables
        input_record_to_arg = {}

        for single_input_key in single_input_keys:
            input_record_name = self.input_record_mapping[single_input_key]
            if input_record_name in input_record_to_arg:
                raise ValueError(
                    f"Prompt variable '{input_record_name}' is mapped from multiple arguments: "
                    f"{input_record_to_arg[input_record_name]} and {single_input_key}"
                )

            input_record_to_arg[input_record_name] = single_input_key

        for pairwise_input_key in pairwise_input_keys:
            input_record_name_single = self.input_record_mapping[
                single_input_key
            ]
            input_record_names = [
                input_record_name_single + "_a",
                input_record_name_single + "_b",
            ]

            for input_record_name in input_record_names:
                if input_record_name in input_record_to_arg:
                    raise ValueError(
                        f"Prompt variable '{input_record_name}' is mapped from multiple arguments: "
                        f"{input_record_to_arg[input_record_name]} and {pairwise_input_key}"
                    )

                input_record_to_arg[input_record_name] = pairwise_input_key

        self.input_records = []
        for i in range(input_length):
            input_record = {}
            for single_key in self.single_inputs:
                single_input = self.single_inputs[single_key]
                single_var_key = self.input_record_mapping[single_key]
                if single_input is None:
                    input_record[single_var_key] = None
                else:
                    input_record[single_var_key] = single_input[i]

            for pairwise_key in self.pairwise_inputs:
                pairwise_input_a, pairwise_input_b = self.pairwise_inputs[
                    pairwise_key
                ]
                input_record_key_a = (
                    self.input_record_mapping[pairwise_key] + "_a"
                )
                if pairwise_input_a is None:
                    input_record[input_record_key_a] = None
                else:
                    input_record[input_record_key_a] = pairwise_input_a[i]
                input_record_key_b = (
                    self.input_record_mapping[pairwise_key] + "_b"
                )
                if pairwise_input_b is None:
                    input_record[input_record_key_b] = None
                else:
                    input_record[input_record_key_b] = pairwise_input_b[i]

    def get_input_records(self) -> list[dict[str, str | None]]:
        if self.input_records is None:
            raise ValueError(
                "Please run `validate` before calling this method."
            )

        return self.input_records
