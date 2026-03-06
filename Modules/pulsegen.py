import numpy as np
from numba import njit
from Modules import constants
import scipy.interpolate
import matplotlib.pyplot as plt

@njit
def rectpulse(flip=20):

    ntau = int(constants.T / constants.DT)

    da = (flip / 180 * np.pi) / (ntau)

    b1_amp = da*np.ones(ntau)
    b1_phase = np.zeros(ntau)

    return b1_amp, b1_phase


@njit
def binomial2(freq1, freq2, phi1, phi2, flip=20):


    tau = (int(constants.T)+0.014)/2
    #tau = constants.T / 2
    ntau = int(tau / constants.DT)
    ninter = int((constants.NT - 2 * ntau))

    ninter = 6
    ntau = int((constants.NT - ninter)/2)
    tau = ntau*constants.DT


    da = (flip / 180 * np.pi) / (2 * tau / constants.DT)

    ramp = np.linspace(-0.5, 0.5, ntau)
    inter = np.zeros(ninter)
    const = np.ones(ntau)

    b1_amp = np.concatenate((da * const, inter, da * const))
    b1_phase = np.concatenate((phi1 + freq1 * ramp, inter, phi2 + freq2 * ramp))

    return b1_amp, b1_phase


@njit
def binomial3(phi1, phi2, phi3, phi4, phi5, phi6, flip=20):

    ninter = 6
    ntau = int((constants.NT - 2*ninter)/4)
    tau = ntau*constants.DT

    b1 = (flip / 180 * np.pi) / (4 * tau / constants.DT)

    ramp1 = np.linspace(-0.5, 0.5, ntau)
    ramp2 = np.linspace(-0.5, 0.5, 2*ntau)
    inter = np.zeros(ninter)
    const1 = np.ones(ntau)
    const2 = np.ones(ntau*2)

    b1_amp = np.concatenate((const1 * b1, inter, const2 * b1, inter, const1 * b1))
    b1_phase = np.concatenate((phi4 + phi1 * ramp1, inter, phi5 + phi2 * ramp2, inter, phi6 + phi3 * ramp1))

    return b1_amp, b1_phase


def randBsplineN(a, p, flip=30, order =2):

    t = np.linspace(0, constants.T, len(a))
    da = (flip / 180 * np.pi)
    k = order

    pulse_amp_spline = scipy.interpolate.make_interp_spline(t, a, k)#, bc_type="clamped")
    pulse_phase_spline = scipy.interpolate.make_interp_spline(t, p, k)#, bc_type="clamped")

    b1_amp = da * pulse_amp_spline(constants.TSEQ).T / np.sum(np.abs(pulse_amp_spline(constants.TSEQ).T))
    b1_phase = pulse_phase_spline(constants.TSEQ).T

    return b1_amp, b1_phase


@njit
def double_sinc(start1, stop1, start2, stop2, startphase1, stopphase1, startphase2, stopphase2):
    flip = 60

    xsinc1 = np.linspace(start1, stop1, int(constants.NT / 2))
    xsinc2 = np.linspace(start2, stop2, int(constants.NT / 2))

    alpha = (flip / 180 * np.pi) / constants.NT*4

    b1_amp = np.concatenate((np.abs(alpha*np.sinc(xsinc1)), np.abs(alpha*np.sinc(xsinc2))))
    b1_phase = np.concatenate((np.linspace(startphase1, stopphase1, int(constants.NT/2)),
                               np.linspace(startphase2, stopphase2, int(constants.NT/2))))

    return b1_amp, b1_phase


def sinc(start1, stop1, phi1, phi2):
    flip = 60

    xsinc1 = np.linspace(start1, stop1, int(constants.NT))
    alpha = (flip / 180 * np.pi) / constants.NT

    b1_amp = alpha*np.sinc(xsinc1)
    b1_phase = np.linspace(phi1, phi2, int(constants.NT))

    return b1_amp, b1_phase