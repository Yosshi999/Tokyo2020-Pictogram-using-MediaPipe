#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
同数のデータ点数を持つS, Tに対し、点ごとの対応が既知であるとして
点群間の平行移動・回転・スケーリングを推定する
"""
from typing import Tuple
from dataclasses import dataclass
import numpy as np

__all__ = ["MatchingResult", "minL2"]

@dataclass
class MatchingResult:
    cost: float
    offsetX: float
    offsetY: float
    angle: float
    scale: float
    movingCenterX: float
    movingCenterY: float

def minL2(S: np.ndarray, T: np.ndarray) -> MatchingResult:
    r"""Find (s, R, t) \in Sim(2) which minimizes sum_i || sRS_i + t - T_i ||^2.
    
    Parameters
    ==========
    S: (N, 2) array_like
        Moving pointcloud.
    T: (N, 2) array_like
        Reference pointcloud.

    Returns
    =======
    result: MatchingResult
    """
    Smean = np.mean(S, axis=0)
    Tmean = np.mean(T, axis=0)
    S_ = S - Smean
    T_ = T - Tmean
    S_F2 = (S_ ** 2).sum()
    T_F2 = (T_ ** 2).sum()
    offset = Tmean - Smean
    U, s, V = np.linalg.svd(S_.T @ T_)
    rot = V @ U.T
    angle = np.arctan2(rot[1,0], rot[0,0])
    trS = np.sum(s)
    scale = trS / S_F2
    cost = T_F2 - trS ** 2 / S_F2
    return MatchingResult(
        cost, offset[0], offset[1], angle, scale,
        Smean[0], Smean[1]
    )
