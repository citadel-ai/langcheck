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

    def _tokenize(self, inputs: list[str]) -> _TokenType:
        '''Tokenize the inputs. The returned type should be defined in the
        subclass.
        '''
        raise NotImplementedError

    def _score_tokens(self, tokens: _TokenType) -> list[Optional[float]]:
        '''Score the tokens. The returned list should have the same length as
        the tokens. Each element in the list should be the score of the token.
        '''
        raise NotImplementedError

    def _slice_tokens(self, tokens: _TokenType, start_idx: int,
                      end_idx: int) -> _TokenType:
        '''Slice the tokens. The returned type should be the same as the tokens.
        It is equivalent to tokens[start_idx:end_idx] for slicable data types
        such as list.
        '''
        raise NotImplementedError

    def score(self, inputs: list[str]) -> list[Optional[float]]:
        '''Score the inputs. Basically subclasses should not override this.
        '''

        tokens = self._tokenize(inputs)

        input_length = len(inputs)

        scores: list[Optional[float]] = []
        for i in tqdm_wrapper(range(0, input_length, self.BASE_BATCH_SIZE),
                              total=(input_length + self.BASE_BATCH_SIZE - 1) //
                              self.BASE_BATCH_SIZE):

            batch_tokens = self._slice_tokens(
                tokens, i, min(i + self.BASE_BATCH_SIZE, input_length))

            scores.extend(self._score_tokens(batch_tokens))

        return scores
