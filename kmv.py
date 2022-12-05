# -*- encoding: utf-8 -*-
'''
Author: Lee Tung
Created: 2022/12/2
'''

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')


class KMV():
    """
    Moody's KMV.

    KMV fits a KMV model with several parameters to minimize the non-linear function value. 

    Parameters
    ----------
    n_iter : int, default=300
        The maximum number of calls to the funciton.

    tol : float, default=1.0e-06
        The calculation will terminate if the relative error between two
        consecutive iterates is at most `tol`.

    initValue : list, default=[1.1, 1.1]
        The starting estimate for the root of ``func(x) = 0``.

    Returns
    -------
    self : object
        Initialize KMV model.

    Notes
    -----
    This ``KMV`` is just basic KMV function with non-linear solution.

    Examples
    --------
    Find a default distance from the company.
    - riskFreeRate = 0.012
    - liabilities = 712788068
    - marketValue = 13793196
    - marketValueVolatility = 0.260344488
    - duration = 1

    >>> from kmv import KMV
    >>> model = KMV()
    >>> model.fit(riskFreeRate, liabilities, marketValue, marketValueVolatility, duration)
    >>> res = model.result()
    >>> res
    (52.060369711520856, 0.005001069830276989)
    >>> summary = model.summary()
    Asset Coefficient  Asset Market Value  Asset Volatility  Default Distance KMV  Default Distance Merton  ND1  ND2
    0           52.06037        7.180789e+08          0.005001              1.473288                 3.875728  1.0  1.0
    """
    
    def __init__(self, n_iter=300, tol=1.0e-06, initValue=[1.1, 1.1]):
        self.n_iter = n_iter
        self.tol = tol
        self.initValue = initValue
        
    def fit(self, riskFreeRate, liabilities, marketValue, marketValueVolatility, duration):
        """Fit the model according to the given parameters.

        Parameters
        ----------
        riskFreeRate : 
        liabilities : 
        marketValue : 
        marketValueVolatility :
        duration : 

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

            .. versionadded:: 0.17
               parameter *sample_weight* support to LinearRegression.

        Returns
        -------
        self : object
            Fitted Estimator.
        """
        self.riskFreeRate = float(riskFreeRate)
        self.liabilities = float(liabilities)
        self.marketValue = float(marketValue)
        self.marketValueVolatility = float(marketValueVolatility)
        self.duration = float(duration)

    def _func(self, x):
        MtoL = self.marketValue / self.liabilities
        d1 = (np.lib.scimath.log(x[0] * MtoL) + (self.riskFreeRate + 0.5 * x[1]**2) * self.duration) / (x[1] * np.lib.scimath.sqrt(self.duration))
        d2 = d1 - x[1] * np.lib.scimath.sqrt(self.duration)
        nd1 = stats.norm.cdf(d1)
        nd2 = stats.norm.cdf(d2)
        return [np.real(x[0] * nd1 - np.exp(-self.riskFreeRate*self.duration)*nd2/MtoL - 1), np.real(nd1*x[0]*x[1]-self.marketValueVolatility)]

    def _solve(self):
        x0 = self.initValue
        root = fsolve(self._func, x0, xtol=self.tol, maxfev=self.n_iter)
        if np.isclose(self._func(root), np.zeros((1,2), dtype=float), atol=1).any():
            return root
        else:
            raise Exception('No locally optimal solution found!')

    @staticmethod
    def defaultDistance_KMV(VA, sigma_A, D):
        default_distance_kmv = (VA-D)/(VA*sigma_A)
        return default_distance_kmv

    @staticmethod
    def defaultDistance_Merton(VA, sigma_A, D, r):
        default_distance_Merton = (np.lib.scimath.log(
            VA/D) + r.real-0.5*sigma_A**2)/sigma_A
        return np.real(default_distance_Merton)

    @staticmethod
    def nd1(VA, sigma_A, E, D, r, T):
        d1 = (np.lib.scimath.log(VA * (E/D)) + (r + 0.5 * sigma_A**2)
              * T) / (sigma_A * np.lib.scimath.sqrt(T))
        return np.real(stats.norm.cdf(d1))

    @staticmethod
    def nd2(VA, sigma_A, E, D, r, T):
        d1 = (np.lib.scimath.log(VA * (E/D)) + (r.real + 0.5 * sigma_A**2)
              * T) / (sigma_A * np.lib.scimath.sqrt(T))
        d2 = d1 - sigma_A * np.lib.scimath.sqrt(T)
        return np.real(stats.norm.cdf(d2))

    def result(self):
        """
        The roots of the function.

        Returns
        -------
        assetCoefficient : float
            The first number of solution list.

        assetVolatility : float
            The second number of solution list.
        """
        assetCoefficient, assetVolatility = self._solve()
        return assetCoefficient, assetVolatility

    def summary(self):
        """
        The roots of the function.

        Returns
        -------
        df : pd.Dataframe
            The summary of KMV model. It includes 
            - Asset Coefficient
            - Asset Market Value
            - Asset Volatility
            - Default Distance KMV
            - Default Distance Merton
            - ND1
            - ND2
        """
        assetCoefficient, assetVolatility = self._solve()
        assetValue = assetCoefficient*self.marketValue
        defaultDistanceKMV = self.defaultDistance_KMV(assetCoefficient*self.marketValue, assetVolatility, self.liabilities)
        defaultDistanceMerton= self.defaultDistance_Merton(assetCoefficient*self.marketValue, assetVolatility, self.liabilities, self.riskFreeRate)
        nd1 = self.nd1(assetCoefficient*self.marketValue, assetVolatility, self.marketValue, self.liabilities, self.riskFreeRate, self.duration)
        nd2 = self.nd2(assetCoefficient*self.marketValue, assetVolatility, self.marketValue, self.liabilities, self.riskFreeRate, self.duration)
        data = {'Asset Coefficient': [assetCoefficient], 
                'Asset Market Value': [assetCoefficient*self.marketValue],
                'Asset Volatility': [assetVolatility],
                'Default Distance KMV': [defaultDistanceKMV],
                'Default Distance Merton': [defaultDistanceMerton],
                'ND1': [nd1],
                'ND2': [nd2]
                }
        df = pd.DataFrame(data=data).T
        return df
   

        



    
    
        