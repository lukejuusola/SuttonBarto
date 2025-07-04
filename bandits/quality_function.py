from abc import ABC
from typing import Optional

import numpy as np

class QualityFunction(ABC):
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
    
class QualitySampleMean(QualityFunction): 
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

class QualityEMAConst(QualityFunction): 
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

class QualityUCB(QualityFunction): 
    def __init__( self, quality: QualityFunction, c:float, prior: Optional[np.array] = None ): 
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

class QualityEMA(QualityFunction): 
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