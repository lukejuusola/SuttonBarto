from abc import ABC
from typing import List

from rewards import RewardGenerator

import numpy as np

class KArmedBandit(ABC):
    def __init__( self, rewards: List[RewardGenerator] ):
        self.rewards = rewards

    def __len__( self ) -> int:
        return len( self.rewards )

    def step( self ) -> None:
        [r.step() for r in self.rewards]

    def query_arm( self, i: int ) -> float:
        return self.rewards[i].get_reward()

    def query_arm_at( self, i: int, t: float ) -> float:
        return self.rewards[i].reward_at(t)
    
    def optimal_arm( self ) -> float:
        rewards = [ r.get_value() for r in self.rewards ]
        return np.argmax( rewards )

    def optimal_arm_at( self,t: float ) -> float:
        rewards = [ r.get_value_at(t) for r in self.rewards ]
        return np.argmax( rewards )