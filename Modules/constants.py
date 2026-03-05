import numpy as np

T = 1.024                          # Simulation time in [ms]
NT = 256                            # Number of simulation steps
DT = T/NT                           # Time steps in [ms]
GYR = 42577478.518                  # Gyromagnetic constant for H1 in Hz/T
SYSFIELD = 3.0                    # System field strength
SYSFREQ = GYR*SYSFIELD
DELTAFREQFAT = -0.0000034


# Sequence param
TE = 4.78
TR = 2.35
T1 = 1480
T2 = 50

NF = 200
NZ = 100

FATFREQ = SYSFREQ*DELTAFREQFAT
CORRCOEF = 1
F = 1000

# B1 map sim parameters
FLIPMIN = 1
FLIPMAX = 90
NFLIP = 200

# Z Gradient sim parameters
FOV = 0.44
DF = np.linspace(-F, F, NF)
TSEQ = np.linspace(0, T, NT)
DZ = np.linspace(-FOV, FOV, NZ)
SLICETHICKNESS = FOV/3.5 

# Loss function weights
L1 = 100
L2 = 5
L3 = 30

# Loss function bands
FATBAND = 50
WATBAND = 10