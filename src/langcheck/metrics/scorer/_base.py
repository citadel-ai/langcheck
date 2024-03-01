from __future__ import annotations

from typing import Generic, TypeVar, Optional
from langcheck.utils.progess_bar import tqdm_wrapper

# Define a type variable for token type.
# This type is used to represent the list of tokens returned by the
# _tokenize method. We do not use `list` type because the token type
# can be a list, dict, or any other type.
_TokenType = TypeVar('_TokenType')


class BaseSingleScorer(Generic[_TokenType]):
    '''Base class for single input scorers.
    '''
    BASE_BATCH_SIZE = 8

    def __init__(self, validation_mode: str = 'raise'):
        if validation_mode not in ['raise', 'null']:
            raise ValueError(f'Invalid validation mode: {validation_mode}')

        self.validation_mode = validation_mode

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

    def _slice_tokens(self, tokens: _TokenType,
                      indices: list[int]) -> _TokenType:
        '''Slice the tokens. The returned type should be the same as the tokens.
        '''
        raise NotImplementedError

    def score(self, inputs: list[str]) -> list[Optional[float]]:
        '''Score the inputs. Basically subclasses should not override this.
        '''

        tokens = self._tokenize(inputs)
        valid_tokens = self._validate_tokens(tokens)

        if self.validation_mode == 'raise':
            if not all(valid_tokens):
                raise ValueError('Invalid tokens')

        input_length = len(inputs)

        scores: list[Optional[float]] = [None] * input_length
        for i in tqdm_wrapper(range(0, input_length, self.BASE_BATCH_SIZE),
                              total=(input_length + self.BASE_BATCH_SIZE - 1) //
                              self.BASE_BATCH_SIZE):
            batch_indices = list(
                range(i, min(i + self.BASE_BATCH_SIZE, input_length)))

            # Only apply _score_tokens to valid tokens
            valid_batch_token_indices = [
                j for j, valid in zip(batch_indices,
                                      valid_tokens[i:i + self.BASE_BATCH_SIZE])
                if valid
            ]
            valid_batch_tokens = self._slice_tokens(tokens,
                                                    valid_batch_token_indices)
            valid_batch_scores = self._score_tokens(valid_batch_tokens)

            for j, score in zip(valid_batch_token_indices, valid_batch_scores):
                scores[j] = score

        return scores


