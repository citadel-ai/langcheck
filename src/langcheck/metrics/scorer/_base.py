from __future__ import annotations

from typing import Generic, TypeVar

# Define a type variable for token type.
# This type is used to represent the list of tokens returned by the
# _tokenize method. We do not use `list` type because the token type
# can be a list, dict, or any other type.
_TokenType = TypeVar('_TokenType')


class BaseSingleScorer(Generic[_TokenType]):
    '''Base class for single input scorers.
    '''

    def __init__(self, validation_mode: str = 'raise'):
        if validation_mode not in ['raise', 'null']:
            raise ValueError(f'Invalid validation mode: {validation_mode}')

        self.validation_mode = validation_mode

    def _validate_input(self, single_input: str) -> bool:
        '''Method to validate input. If the input needs any validation, it
        should be implemented in the subclass. By default, it returns True.
        False means the input is invalid.
        '''
        return True

    def _tokenize(self, inputs: list[str]) -> _TokenType:
        '''Tokenize the inputs. The returned type should be defined in the
        subclass.
        '''
        raise NotImplementedError

    def _validate_tokens(self, tokens: _TokenType) -> list[bool]:
        '''Validate the tokens. The returned list should have the same length
        as the tokens. Each element in the list should be True if the token is
        valid, False otherwise.
        '''
        raise NotImplementedError

    def _score_tokens(self, tokens: _TokenType) -> list[float]:
        '''Score the tokens. The returned list should have the same length as
        the tokens. Each element in the list should be the score of the token.
        '''
        raise NotImplementedError
