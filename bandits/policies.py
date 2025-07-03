from abc import ABC

from typing import Tuple, Optional

import numpy as np

class Policy(object):
    def choice(self) -> int:
        raise NotImplementedError
    
    def update(self, i: int, r: float) -> None:
        raise NotImplementedError
    
# class GreedyPolicy(Policy): 
#     def __init__( self, k: int ): 
#         self.sumRewards = np.zeros(k)
#         self.nSampled   = np.zeros(k)

#     def choice(self) -> int: 
#         meanRewards = self.sumRewards.copy()
#         noSampleMask = self.nSampled == 0
#         meanRewards[noSampleMask] = np.inf
#         meanRewards[~noSampleMask] /= self.nSampled[~noSampleMask]
#         return np.argmax( meanRewards )
    
#     def update( self, i: int, r: float ) -> None:
#         self.sumRewards[i] += r
#         self.nSampled[i] += 1

# class EpsilonGreedyPolicy(Policy): 
#     def __init__( self, k: int, epsilon: float ): 
#         self.sumRewards = np.zeros(k)
#         self.nSampled   = np.zeros(k)
#         assert 0 <= epsilon <= 1, f"{epsilon} not in [0, 1]"
#         self.eps        = epsilon

#     def choice(self) -> int:
#         if np.random.uniform() < self.eps: 
#             return np.random.randint( self.sumRewards.shape[0] )
#         meanRewards = self.sumRewards.copy()
#         noSampleMask = self.nSampled == 0
#         meanRewards[noSampleMask] = np.inf
#         meanRewards[~noSampleMask] /= self.nSampled[~noSampleMask]
#         return np.argmax( meanRewards )
    
#     def update( self, i: int, r: float ) -> None:
#         self.sumRewards[i] += r
#         self.nSampled[i] += 1

class BanditTracker(ABC):
    def __init__(self, k: int, prior: Optional[np.array] ):
        self.k = k
        self.set_prior( prior )

    def set_prior( self, prior ):
        self.prior = prior
        if prior is None:
            self.prior = np.zeros( self.k )
        assert self.prior.shape[0] == self.k

    def get_parameters( self ) -> np.array:
        raise NotImplementedError
    
    def update_parameters( self, arm: int, r: float ) -> np.array:
        raise NotImplementedError
    
class BTSampleMean(BanditTracker): 
    def __init__( self, k: int, prior: Optional[np.array] = None ): 
        super().__init__( k, prior )
        self.mean = self.prior.copy()
        self.nSampled = np.zeros( k )

    def get_parameters(self) -> np.array: 
        # Careful, this is definitely returned by reference
        return self.mean
    
    def update_parameters( self, arm: int, r: float ) -> None:
        self.nSampled[arm] += 1
        self.mean[arm] += ( r - self.mean[arm] ) / self.nSampled[arm]

class BTEMAConst(BanditTracker): 
    def __init__( self, k: int, alpha: float, prior: Optional[np.array] = None ): 
        super().__init__(k, prior)
        assert 0 <= alpha <= 1
        self.alpha = alpha
        self.mean = self.prior.copy()

    def get_parameters(self) -> np.array: 
        # Careful, this is definitely returned by reference
        return self.mean
    
    def update_parameters( self, arm: int, r: float, prior: Optional[np.array] = None ) -> None:
        self.mean[arm] *= (1 - self.alpha)
        self.mean[arm] += self.alpha * r

class BTUCB(BanditTracker): 
    def __init__( self, quality: BanditTracker, c:float, prior: Optional[np.array] = None ): 
        super().__init__( quality.k, prior )
        self.quality = quality
        self.c = c
        self.totalUpdates = 0
        self.nUpdates = np.zeros( self.k )
    
    def get_parameters( self ) -> np.array:
        q = self.quality.get_parameters()
        conf = np.ones( self.k ) * np.inf
        if self.totalUpdates == 0: 
            return conf
        mask = self.nUpdates != 0
        conf[mask] = np.sqrt( np.log( self.totalUpdates ) / self.nUpdates[mask] )
        if self.totalUpdates > self.k:
            assert np.all( conf < np.inf )
        return q + self.c * conf
    
    def update_parameters(self, arm, r) -> None:
        self.totalUpdates += 1
        self.nUpdates[arm] += 1
        self.quality.update_parameters(arm, r)

class BTEMA(BanditTracker): 
    def __init__( self, k: int, alpha: float, prior: Optional[np.array] = None ): 
        super().__init__(k, prior)
        assert 0 <= alpha <= 1
        self.alpha = alpha
        self.mean = self.prior.copy()
        self.i = 0
        self.lastUpdate = np.zeros(k)

    def get_parameters(self) -> np.array: 
        # Careful, this is definitely returned by reference
        return self.mean
    
    def update_parameters( self, arm: int, r: float ) -> None:
        self.i += 1
        decay = np.power( 1-self.alpha, self.i - self.lastUpdate[arm] )
        self.mean[arm] *= decay
        self.mean[arm] += (1 - decay) * r
        self.lastUpdate[arm] = self.i

class GreedyPolicy(Policy): 
    def __init__( self, bt: BanditTracker ): 
        self.bt = bt 

    def choice(self) -> int: 
        return np.argmax( self.bt.get_parameters() )
    
    def update( self, i: int, r: float ) -> None:
        self.bt.update_parameters( i, r )

class EpsilonGreedyPolicy(Policy): 
    def __init__( self, epsilon: float, bt: BanditTracker ): 
        assert 0 <= epsilon <= 1, f"{epsilon} not in [0, 1]"
        self.eps        = epsilon
        self.bt = bt

    def choice(self) -> int:
        if np.random.uniform() < self.eps: 
            return np.random.randint( self.bt.k )
        return np.argmax( self.bt.get_parameters() )
    
    def update( self, i: int, r: float ) -> None:
        self.bt.update_parameters( i, r )