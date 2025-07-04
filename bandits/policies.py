from abc import ABC
from bandits.quality_function import QualityFunction, QualitySoftmaxGrad
from bandits.utils import softmax

import numpy as np

class Policy(ABC):
    def choice(self) -> int:
        raise NotImplementedError

    def update(self, i: int, r: float) -> None:
        raise NotImplementedError

class GreedyPolicy(Policy):
    def __init__( self, quality: QualityFunction ):
        self.quality = quality

    def choice(self) -> int:
        return np.argmax( self.quality.get_parameters() )

    def update( self, i: int, r: float ) -> None:
        self.quality.update_parameters( i, r )

class EpsilonGreedyPolicy(Policy):
    def __init__( self, epsilon: float, quality: QualityFunction ):
        assert 0 <= epsilon <= 1, f"{epsilon} not in [0, 1]"
        self.eps     = epsilon
        self.quality = quality

    def choice(self) -> int:
        if np.random.uniform() < self.eps:
            return np.random.randint( self.quality.k )
        return np.argmax( self.quality.get_parameters() )

    def update( self, i: int, r: float ) -> None:
        self.quality.update_parameters( i, r )

class SoftmaxStochasticPolicy(Policy):
    def __init__( self, quality: QualityFunction ):
        assert isinstance( quality, QualitySoftmaxGrad )
        self.quality = quality

    def choice( self ) -> int:
        return np.random.choice( self.quality.k, p=softmax( self.quality.get_parameters() ) )

    def update( self, i: int, r: float ) -> None:
        self.quality.update_parameters( i, r )