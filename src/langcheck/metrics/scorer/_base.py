from __future__ import annotations

from typing import Generic, Optional, TypeVar

import torch
from sentence_transformers import util
from torch import Tensor

from langcheck.utils.progess_bar import tqdm_wrapper

# Define a type variable for token type.
# This type is used to represent the list of tokens returned by the
# _tokenize method. We do not use `list` type because the token type
# can be a list, dict, or any other type.
_TokensType = TypeVar('_TokensType')


class BaseSingleScorer(Generic[_TokensType]):
    '''Base class for single input scorers.
    '''

    def __init__(self) -> None:
        self.batch_size = 8

    def _tokenize(self, inputs: list[str]) -> _TokensType:
        '''Tokenize the inputs. The returned type should be defined in the
        subclass.
        '''
        raise NotImplementedError

    def _score_tokens(self, tokens: _TokensType) -> list[Optional[float]]:
        '''Score the tokens. The returned list should have the same length as
        the tokens. Each element in the list should be the score of the token.
        '''
        raise NotImplementedError

    def _slice_tokens(self, tokens: _TokensType, start_idx: int,
                      end_idx: int) -> _TokensType:
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
        for i in tqdm_wrapper(range(0, input_length, self.batch_size),
                              total=(input_length + self.batch_size - 1) //
                              self.batch_size):

            batch_tokens = self._slice_tokens(
                tokens, i, min(i + self.batch_size, input_length))

            scores.extend(self._score_tokens(batch_tokens))

        return scores


class BaseSimilarityScorer:
    '''Base class for similarity score calculators, which calculate the
    similarity score between two inputs.
    '''

    def __init__(self) -> None:
        self.batch_size = 8

    def _embed(self, inputs: list[str]) -> Tensor:
        '''Embed the inputs. The returned type should be defined in the
        subclass.
        '''
        raise NotImplementedError

    def _get_similarity_score(self, embedding1: Tensor,
                              embedding2: Tensor) -> list[float]:
        '''Calculate the similarity score between the two embeddings. The
        returned list should have the same length as the embeddings. Each
        element in the list should be the similarity score of the two
        embeddings.
        '''
        cosine_scores = util.pairwise_cos_sim(embedding1, embedding2)
        # Numerical instability can cause the dot product of almost identical
        # vectors to exceed 1.0 slightly, so we clip the outputs.
        cosine_scores = torch.clamp(cosine_scores, -1.0, 1.0)
        return cosine_scores.tolist()

    def score(self, inputs1: list[str], inputs2: list[str]) -> list[float]:
        '''Score the similarity between the inputs. Basically subclasses should
        not override this.
        '''
        input_length = len(inputs1)

        embeddings1 = []
        embeddings2 = []

        # Wrap the encoding process in a progress bar.
        for i in tqdm_wrapper(range(0, input_length, self.batch_size),
                              total=(input_length + self.batch_size - 1) //
                              self.batch_size,
                              desc='Getting embeddings'):
            batch_inputs1 = inputs1[i:min(i + self.batch_size, input_length)]
            batch_inputs2 = inputs2[i:min(i + self.batch_size, input_length)]

            embeddings1.append(self._embed(batch_inputs1))
            embeddings2.append(self._embed(batch_inputs2))

        # Concatenate the embeddings
        embedding1 = torch.cat(embeddings1, dim=0)
        embedding2 = torch.cat(embeddings2, dim=0)

        scores: list[float] = []
        for i in tqdm_wrapper(range(0, input_length, self.batch_size),
                              total=(input_length + self.batch_size - 1) //
                              self.batch_size,
                              desc='Computing semantic similarity'):
            start_idx = i
            end_idx = min(i + self.batch_size, input_length)
            batch_embedding1 = embedding1[start_idx:end_idx]
            batch_embedding2 = embedding2[start_idx:end_idx]

            scores.extend(
                self._get_similarity_score(batch_embedding1, batch_embedding2))

        return scores
