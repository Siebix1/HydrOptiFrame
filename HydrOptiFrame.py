########################################################################################################################
#                                       HydrOptiFrame: Water Excitation RF pulse design                                #
########################################################################################################################
# Numerical Optimisation of RF pulse to find the best WE RF pulse
# Optimization of nodes and interpolation with  
#
#

#                                                   Import stuff                                                       #
import numpy as np
from Modules.PulsePTAGen import pta_gen
from Modules import constants, lossfunctions, pulsegen, plots, simulations
import optuna

#%%#                                               Const generate                                                      #

Tseq = np.linspace(0, constants.T, constants.NT)
dF = np.linspace(-constants.F, constants.F, constants.NF)
#%%#                                              Optimisation loop                                                    #

def binom_opti(trial):
    phi = []
    amp = []

    for i in range(Npoints):
        if i == 0:
            phi = [trial.suggest_float('phi1', -phiLim, phiLim, step=0.001)]
            amp = [trial.suggest_float('amp1', ampLim1, ampLim2, step=0.001)]
        else:
            phi = np.concatenate([phi, [trial.suggest_float('phi'+str(i+1), -phiLim, phiLim, step=0.001)]])
            amp = np.concatenate([amp, [trial.suggest_float('amp'+str(i+1), ampLim1, ampLim2, step=0.001)]])

    if setEdgesToZero:
        amp = np.concatenate([[0], amp, [0]])
        phi = np.concatenate([[0], phi, [0]])

    b1_amp, b1_phase = pulsegen.randBsplineN(amp, phi) # interpolate the suggested points to create the waveform
    M1 = simulations.pulse_offset_relax(b1_amp, b1_phase, dF) # Bloch equations that uses small angle approx
    loss = lossfunctions.composed_loss(M1) #compute the loss function from the final magnetisation vector

    return loss

#%%#                                                 Optimize                                                          #
# Optimization parameters
setEdgesToZero =0
Npoints = 15
pi = 3.142
ampLim1 = 0.01
ampLim2 = 1
phiLim = 4*pi
X0 = []
N_epochs = 1000

sampler = optuna.samplers.CmaEsSampler(n_startup_trials=np.round(N_epochs/10), sigma0=1/25)
study = optuna.create_study(sampler=sampler)
study.optimize(binom_opti, n_trials=N_epochs)

#%%#                                                Plot  Results                                                          #

best_p = study.best_params

nparr = np.array(list(best_p.items()))
a = nparr[:,1]
amp = a[1::2]
phi = a[::2]

B1_amp, B1_phase = pulsegen.randBsplineN(amp, phi, 30)
M = simulations.pulse_offset_relax(B1_amp, B1_phase, dF)
M_xy = np.abs(M[:, :, 0] + 1j * M[:, :, 1])

title = "spl_" + str(int(constants.SYSFIELD * 10)) + "T_" + str(Npoints) + "p_" + str(int(
        constants.T * 1000)) + "ms_" + str(int(constants.L2*10000))+"L2_"+str(constants.FATBAND)+"FB"

plots.pulse_amp_phase_freq_spl(M_xy, B1_amp, B1_phase, amp, phi, title)
#plots.plot_3d_vect(M, title)
plots.amp_phase_paramno(B1_amp, B1_phase, title)

MB1 = simulations.b1_sim_off(amp, phi, dF)

plots.b1_map(MB1)
#%%                                              Generate pulse file

pta_gen(title, B1_amp[:], -B1_phase[:], constants.T, "fatfreq = "+str(constants.FATFREQ)+" l1 = "+str(constants.L1))
