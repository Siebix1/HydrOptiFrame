import numpy as np
from numba import njit
from Modules import constants, pulsegen, bsim
from joblib import Parallel, delayed


@njit
def pulse_offset_relax(B1_amp, B1_phase, dF):
    M0 = np.array([0, 0, 1], dtype="float")
    M = np.zeros((constants.NF, constants.NT, 3))

    for j in range(constants.NF):
        a, b = bsim.freeprecess(constants.DT, constants.T1, constants.T2, 0)

        for i in range(constants.NT):
            rth = bsim.throtoffres(B1_amp[i], B1_phase[i], dF[j])

            if i == 0:
                M[j, i, :] = (a @ rth @ M0.reshape(3, 1) + b).reshape(3,)

            else:
                M[j, i, :] = (a @ rth @ M[j, i - 1, :].reshape(3, 1) + b).reshape(3,)

    return M



@njit
def b1_sim_off(amp, phi, dF):
    flip = np.linspace(constants.FLIPMIN, constants.FLIPMAX, constants.NFLIP)
    M = np.zeros((constants.NFLIP, constants.NF, constants.NT, 3))

    for i in range(constants.NFLIP):
        B1_amp, B1_phase = pulsegen.randBsplineN(amp, phi, flip[i])
        M[i, :, :, :] = pulse_offset_relax(B1_amp, B1_phase, dF)
    return M
