from abc import ABC
from bandits.quality_function import QualityFunction

import numpy as np

class Policy(ABC):
    def choice(self) -> int:
        raise NotImplementedError
    
    def update(self, i: int, r: float) -> None:
        raise NotImplementedError

class GreedyPolicy(Policy): 
    def __init__( self, bt: QualityFunction ): 
        self.bt = bt 

    def choice(self) -> int: 
        return np.argmax( self.bt.get_parameters() )
    
    def update( self, i: int, r: float ) -> None:
        self.bt.update_parameters( i, r )

class EpsilonGreedyPolicy(Policy): 
    def __init__( self, epsilon: float, bt: QualityFunction ): 
        assert 0 <= epsilon <= 1, f"{epsilon} not in [0, 1]"
        self.eps        = epsilon
        self.bt = bt

    def choice(self) -> int:
        if np.random.uniform() < self.eps: 
            return np.random.randint( self.bt.k )
        return np.argmax( self.bt.get_parameters() )
    
    def update( self, i: int, r: float ) -> None:
        self.bt.update_parameters( i, r )