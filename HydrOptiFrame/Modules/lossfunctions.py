import matplotlib.pyplot as plt

from Modules import constants
import numpy as np
from numba import njit


@njit
def fat_water_ratio(M):
    fatband = constants.FATBAND * constants.CORRCOEF
    s1 = constants.FATFREQ - fatband
    s2 = constants.FATFREQ + fatband

    waterband = 20 * constants.CORRCOEF

    s3 = -waterband
    s4 = waterband

    d_f = 2*constants.F / constants.NF

    start = int((constants.F + s1) / d_f)
    stop = int((constants.F + s2) / d_f)

    startw = int((constants.F + s3) / d_f)
    stopw = int((constants.F + s4) / d_f)

    fatSignal = np.mean(M[start:stop])
    waterSignal = np.mean(M[startw:stopw])

    loss = fatSignal/waterSignal
    return loss


def water_fat_ratio(M):
    fatband = 20
    s1 = constants.FATFREQ - fatband
    s2 = constants.FATFREQ + fatband

    waterband = 50

    s3 = -waterband
    s4 = waterband

    d_f = 2*constants.F / constants.NF

    start = int((constants.F + s1) / d_f)
    stop = int((constants.F + s2) / d_f)

    startw = int((constants.F + s3) / d_f)
    stopw = int((constants.F + s4) / d_f)

    fatSignal = np.mean(M[start:stop])
    waterSignal = np.mean(M[startw:stopw])

    loss = waterSignal/fatSignal
    return loss


def derivative_fat(M):
    s1 = -540
    s2 = -340
    d_f = 2*constants.F / constants.NF

    start = int((constants.F + s1) / d_f)
    stop = int((constants.F + s2) / d_f)

    dfilter = [-1, 0, 1]
    derivativeSum = np.convolve(M[start:stop], dfilter, mode="same")

    loss = (derivativeSum**2).mean()
    return loss

@njit
def mean_water(M):
    waterband = 20 * constants.CORRCOEF
    s3 = -waterband
    s4 = waterband

    d_f = 2*constants.F / constants.NF

    startw = int((constants.F + s3) / d_f)
    stopw = int((constants.F + s4) / d_f)
    return np.mean(M[startw:stopw])


@njit
def mean_fat(M):
    fatband = 20
    s3 = constants.FATFREQ - fatband
    s4 = constants.FATFREQ + fatband

    d_f = 2*constants.F / constants.NF

    start = int((constants.F + s3) / d_f)
    stop = int((constants.F + s4) / d_f)

    return np.mean(M[start:stop])


@njit
def l2_fat(M):
    fatband = constants.FATBAND * constants.CORRCOEF
    s3 = constants.FATFREQ - fatband
    s4 = constants.FATFREQ + fatband

    d_f = 2*constants.F / constants.NF

    start = int((constants.F + s3) / d_f)
    stop = int((constants.F + s4) / d_f)

    offset_band = 0
    offset_indice = int(offset_band / d_f)

    weight_fltr = np.ones((stop-start+offset_indice))
    weight_fltr[int((stop-start)/2) - 5:int((stop-start)/2) + 5] = 20
    weight_fltr[int((stop-start)/2) - 2:int((stop-start)/2) + 2] = 140
    M1 = M[start:stop + offset_indice]*weight_fltr

    return np.mean(M1**2)


def l2_water(M):
    watband = 50
    s3 = - watband
    s4 = watband

    d_f = 2*constants.F / constants.NF

    start = int((constants.F + s3) / d_f)
    stop = int((constants.F + s4) / d_f)

    offset_band = 0
    offset_indice = int(offset_band / d_f)

    weight_fltr = np.ones((stop-start+offset_indice))
    weight_fltr[int((stop-start)/2) - 5:int((stop-start)/2) + 5] = 20
    weight_fltr[int((stop-start)/2) - 2:int((stop-start)/2) + 2] = 80
    M1 = M[start:stop + offset_indice]*weight_fltr

    return np.mean(M1**2)


#@njit
def composed_loss(M, verbose=0):
    M_xy = np.abs(M[:, constants.NT-1, 0] + 1j * M[:, constants.NT - 1, 1])

    mean_w = mean_water(M_xy)
    mean_l2_f = l2_fat(M_xy)
    contrast_f_w = fat_water_ratio(M_xy)

    if verbose == 1:
        print("mean water = ", constants.L2 / mean_w)
        print("L2 fat = ", constants.L1 * mean_l2_f)
        print("fat_water_ratio = ", contrast_f_w)

    return contrast_f_w + constants.L2 / mean_w + constants.L1 * mean_l2_f

