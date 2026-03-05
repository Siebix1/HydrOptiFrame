import numpy as np


def pta_gen(title, B1_amp, B1_phase, T, comment="you didn't add any comment you idiot"):
    if len(B1_amp) != len(B1_phase):
        return -1
    gamma = 42577478.518
    refgrad = 1000*5.12 / (gamma * 0.01*T)

    B1_amp= B1_amp/np.max(B1_amp)

    B1_norm = (B1_amp)*np.exp(1j*B1_phase)
    ampInt = np.abs(np.sum(B1_norm))
    powerInt = np.sum(np.abs((B1_norm) ** 2))
    absInt = np.sum(np.abs(B1_norm))

    with open("PulsesPTAfiles/" + title + '.pta', 'w') as f:
        f.write('PULSENAME:\t' + title + '\r\n')
        f.write('COMMENT:\t' + comment + '\r\n')
        f.write('REFGRAD:\t' + str(refgrad) + '\r\n')
        f.write('MINSLICE:\t10' + '\r\n')
        f.write('MAXSLICE:\t500\r\n')
        f.write('AMPINT:\t' + str(ampInt) + '\r\n')
        f.write('POWERINT:\t' + str(powerInt) + '\r\n')
        f.write('ABSINT:\t' + str(absInt) + '\r\n')
        f.write('\t' + '\r\n')
        for i in range(len(B1_amp)):
            f.write(str(np.round(B1_amp[i], 8)) + '\t' + str(np.round((B1_phase[i] + np.pi) % (2 * np.pi) - np.pi, 8)) + '\t' + ';\t(' + str(i) + ')\r\n')

    return print("Pulse file done")

