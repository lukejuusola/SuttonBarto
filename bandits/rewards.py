from abc import ABC

import numpy as np

class RewardGenerator(ABC):
    def step(self) -> None:
        # For discrete bandit reward sampling
        raise NotImplementedError

    def get_reward( self ) -> float:
        # Get current reward
        raise NotImplementedError

    def reward_at(self, t: float) -> float:
        # For continuous bandit reward sampling
        raise NotImplementedError

    def __add__( self, other ):
        return SumRewardGenerator( self, other )

class SumRewardGenerator(RewardGenerator):
    def __init__( self, r1: RewardGenerator, r2: RewardGenerator ):
        self.leftReward = r1
        self.rightReward = r2

    def step( self ) -> None:
        self.leftReward.step()
        self.rightReward.step()

    def get_reward(self) -> float:
        return self.leftReward.get_reward() + self.rightReward.get_reward()

    def reward_at(self, t: float):
        return self.leftReward.reward_at(t) + self.rightReward.reward_at(t)
    
class ValueNoiseRewardGenerator(RewardGenerator): 
    def __init__( self, valueGen: RewardGenerator, noiseGen: RewardGenerator ): 
        self.valueGen = valueGen
        self.noiseGen = noiseGen
        self.rewardGen = valueGen + noiseGen

    def step( self ) -> None: 
        self.rewardGen.step()

    def get_reward(self) -> float:
        assert self.valueGen.get_reward() + self.noiseGen.get_reward() == self.rewardGen.get_reward() 
        return self.rewardGen.get_reward()
    
    def get_value( self ) -> float: 
        return self.valueGen.get_reward()
    
    def get_noise( self ) -> float: 
        return self.noiseGen.get_reward() 

    def reward_at(self, t: float) -> float:
        assert self.valueGen.reward_at(t) + self.noiseGen.reward_at(t) == self.rewardGen.reward_at(t) 
        return self.rewardGen.reward_at(t)


class ConstantRewardGenerator(RewardGenerator):
    def __init__(self, constReward: float):
        self.constReward = constReward

    def step( self ) -> None:
        return

    def get_reward(self) -> float:
        return self.constReward

    def reward_at(self, t: float):
        return self.constReward

class NormalNoiseRewardGenerator( RewardGenerator ):
    def __init__( self, mean: float, std: float, periodRecomp: int = 1000 ):
        assert periodRecomp > 0
        self.mean = mean
        self.std  = std
        self.n = periodRecomp
        self.rewards = np.random.normal( self.mean, self.std, self.n )
        self.i = 0

    def step( self ) -> None:
        self.i += 1
        if self.i % self.n == 0:
            self.rewards = np.random.normal( self.mean, self.std, self.n )
            self.i -= self.n

    def get_reward( self ) -> float:
        return self.rewards[self.i]

    def reward_at(self, t: float) -> float:
        # Should have a small problem where we can query 1050 and 2050 without
        # doing a regeneration so this has definitionally detectable periodicity
        raise NotImplementedError
        return self.reward_at[self.i]
    
class BrownianMotionRewardGenerator( RewardGenerator ): 
    def __init__( self, std: float, periodRecomp: int = 1000 ): 
        assert periodRecomp > 0
        self.std = std
        self.n = periodRecomp
        self.rewards = np.hstack( [[0], np.random.normal( 0, std, self.n - 1 )] ).cumsum()
        self.i = 0

    def step( self ) -> None: 
        self.i += 1
        if self.i % self.n == 0: 
            start = self.rewards[-1]
            self.rewards = np.random.normal( 0, self.std, self.n ).cumsum() + start
            self.i -= self.n
    
    def get_reward( self ) -> float: 
        return self.rewards[self.i]
    
    def reward_at(self, t):
        # Should have a small problem where we can query 1050 and 2050 without
        # doing a regeneration so this has definitionally detectable periodicity
        raise NotImplementedError
        return self.reward_at[self.i]

