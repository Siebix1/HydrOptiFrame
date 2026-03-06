import numpy as np
import matplotlib.pyplot as plt
from Modules import constants
import matplotlib.patches as patches

def pulse_amp_phase_freq_spl(M_xy, B1_amp, B1_phase, amp, phi, title):
    Tseq = np.linspace(0, constants.T, constants.NT)
    dF = np.linspace(-constants.F, constants.F, constants.NF)
    d_f = 2 * constants.F / constants.NF

    freqFatStart = int((constants.F + constants.FATFREQ - constants.FATBAND) / d_f)
    freqFatStop = int((constants.F + constants.FATFREQ + constants.FATBAND) / d_f)
    freqWaterStart = int((constants.F - constants.FATBAND) / d_f)
    freqWaterStop = int((constants.F + constants.FATBAND) / d_f)
    freqFat = int((constants.F + constants.FATFREQ) / d_f)

    plotBand = 2*constants.FATBAND
    freqFatplotstart = int((constants.F + constants.FATFREQ - plotBand) / d_f)
    freqFatplotstop = int((constants.F + constants.FATFREQ + plotBand) / d_f)

    waterFatRatio = np.mean(M_xy[freqWaterStart:freqWaterStop, constants.NT - 1]) / np.mean(
                            M_xy[freqFatStart:freqFatStop, constants.NT - 1])

    fatMean = np.mean(M_xy[freqFatStart:freqFatStop, constants.NT - 1])
    rect = np.ones(constants.NT)

    ntau = int(constants.T / constants.DT)
    da = (30 / 180 * np.pi) / (ntau)

    pulse_amp_tesla = B1_amp*1000/(constants.GYR*constants.T)
    print("SAR = ", np.sum(B1_amp**2)/np.sum((da*rect)**2))

    fig, axs = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle('Optipulse'+title)#+ str(amp) + str(phi))

    tspace = np.linspace(0, constants.T, len(amp))
    da = amp.astype(float)*(30 / 180 * np.pi)
    pts_tesla = np.array(da * 1000 / (constants.GYR * constants.DT))

    indexes = np.linspace(0, constants.NT-1, len(amp)).astype(int)

    axs[0][0].plot(Tseq, pulse_amp_tesla)
    axs[0][0].plot(tspace, pulse_amp_tesla[indexes], "ro")
    axs[0][0].set_xlabel("Time [ms]")
    axs[0][0].set_ylabel("B1 amplitude [T]")
    axs[0][0].set_title("RF Pulse")
    axs[0][0].set_xlim((0, constants.T))
    axs[0][0].set_ylim((0, 1.1*np.max(pulse_amp_tesla)))

    axs[0][1].plot(Tseq, B1_phase)
    axs[0][1].plot(tspace, phi.astype(float), "ro")
    axs[0][1].set_xlabel("Time [ms]")
    axs[0][1].set_ylabel("Phase [rad]")
    axs[0][1].set_title("RF Pulse")
    axs[0][1].set_xlim((0, constants.T))


    textPosMax = np.max(M_xy[:, constants.NT - 1])

    axs[1][0].plot(dF, M_xy[:, constants.NT - 1])
    axs[1][0].vlines([constants.FATFREQ], 0, 1.1 * np.max(M_xy[:, constants.NT - 1]), linestyles='dashed', colors='red',
                     label="Fat freq. " + str(constants.FATFREQ))
    axs[1][0].set_xlabel("Frequency shift [Hz]")
    axs[1][0].set_ylabel("Amplitude [a.u.]")
    axs[1][0].text(250, 0.4 * textPosMax, "Amp ratio = " + str(np.round(waterFatRatio, 4)))
    axs[1][0].text(250, 0.3 * textPosMax, "Fat mean = " + str(np.round(fatMean, 4)))
    axs[1][0].text(250, 0.2 * textPosMax,
                   "Amp at 0Hz= = " + str(np.round(M_xy[int(constants.NF / 2), constants.NT - 1], 4)))
    axs[1][0].set_title("|Mxy|")
    axs[1][0].text(200, 0.1 * textPosMax,
                   "Amp at 215Hz= " + str(np.round(M_xy[int(freqFat), constants.NT - 1], 4)))
    axs[1][1].plot(dF[freqFatplotstart:freqFatplotstop], M_xy[freqFatplotstart:freqFatplotstop, constants.NT - 1])
    axs[1][1].vlines([constants.FATFREQ], 0, 1.1 * np.max(M_xy[freqFatStart:freqFatStop, constants.NT - 1]),
                     linestyles='dashed', colors='red', label="Fat freq. 440")
    axs[1][1].set_xlabel("Frequency shift [Hz]")
    axs[1][1].set_ylabel("Amplitude [a.u.]")
    axs[1][1].set_title("|Mxy| zoom")
    plt.savefig("Figures/Pulse_" + title + str(int(constants.T)) + "ms.png")
    plt.show()

    return print("2x2 plots done")


def amp_phase_paramno(B1_amp, B1_phase, title):
    #plt.rcParams.update({'font.size': 16})
    Tseq = np.linspace(0, constants.T, constants.NT)
    fig, ax1 = plt.subplots(figsize=(5,4))
    color = 'tab:red'
    ax1.set_xlabel("Time [ms]")
    ax1.set_ylabel("B1 amplitude [T]", color=color)
    ax1.plot(Tseq[:], B1_amp[:] / (constants.GYR * constants.DT / 1000), color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xlim((0, constants.T))
    ax1.set_ylim((0, 1.1*B1_amp.max()/ (constants.GYR * constants.DT / 1000) ))
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel("Phase [rad]", color=color)  # we already handled the x-label with ax1
    ax2.plot(Tseq[:], -B1_phase[:], color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_xlim((0, constants.T))
    #ax2.set_ylim((-3, 3))
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig("Figures/pulse_amp_phase" + title)
    plt.show()


    return print("Amplitude/phase plotted!")


def amp_phase_normed_plot(B1_amp, B1_phase, title="Title"):
    #plt.rcParams.update({'font.size': 16})
    Tseq = np.linspace(0, constants.T, constants.NT)

    fig, ax1 = plt.subplots(figsize=(5,4))
    color = 'tab:red'
    ax1.set_xlabel("Time [ms]")
    ax1.set_ylabel("B1 amplitude [T]", color=color)
    ax1.plot(Tseq[:], B1_amp / np.max(B1_amp), color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xlim((0, constants.T))
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel("Phase [rad]", color=color)  # we already handled the x-label with ax1
    ax2.plot(Tseq[:], -B1_phase/np.max(B1_phase), color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_xlim((0, constants.T))
    #ax2.set_ylim((-3, 3))
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig("Figures/pulse_amp_phase" + title)
    plt.show()


    return print("Amplitude/phase plotted!")

def b1_map(m_b1):

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(m_b1, extent=[-constants.F, constants.F, constants.FLIPMIN, constants.FLIPMAX], aspect="auto",
                   origin='lower')
    ax.set_ylabel("Flip angle [°]")
    ax.set_xlabel("RF offset frequency [Hz]")
    cbar = ax.figure.colorbar(im)
    cbar.outline.set_visible(False)
    cbar.ax.set_ylabel("|Mxy| [a.u.]", rotation=90, va="bottom")
    fig.tight_layout()
    plt.show()
    plt.savefig("Figures/B1_121_" + str(int(constants.T)) + "ms")

    return print("B1_map plotted!")


def loss_plot(loss, N_epoch):

    plt.figure(figsize=(12, 8))
    plt.plot(np.linspace(0,N_epoch-1, N_epoch), loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xlim((0,N_epoch-1))
    plt.ylim((0,np.max(loss*1.1)))
    plt.tight_layout()
    plt.show()
    plt.savefig("Figures/loss_" + str(int(constants.T)) + "ms")

    return print("Loss plotted!")



def plot_amp(B1_amp, amp, i):
    B1_amp_tesla = B1_amp * 1000000 / (constants.GAMMA * constants.DT/1000)
    tspace = np.linspace(0, constants.T, len(amp))
    indexes = np.linspace(0, constants.NT-1, len(amp)).astype(int)

    plt.figure(figsize=(4.5, 3))
    plt.plot(constants.TSEQ, B1_amp_tesla)
    plt.plot(tspace, B1_amp_tesla[indexes], "ro")
    plt.xlabel("Time [ms]")
    plt.ylabel("B1 Amplitude [µT]")
    plt.title("Pulse Amplitude")
    plt.xlim((0, np.max(constants.TSEQ)))
    plt.ylim((0, 10))
    plt.tight_layout()
    plt.savefig("Figures/amp/amp_"+str(i)+".png")
    plt.show()


    return

def plot_phase(B1_phase, phi, i):

    tspace = np.linspace(0, constants.T, len(phi))

    plt.figure(figsize=(4.5, 3))
    plt.plot(constants.TSEQ, B1_phase)
    plt.plot(tspace, phi.astype(float), "ro")
    plt.xlabel("Time [ms]")
    plt.ylabel("B1 Phase [rad]")
    plt.title("Pulse Phase")
    plt.xlim((0, np.max(constants.TSEQ)))
    plt.ylim((-1.2*np.pi, 1.2*np.pi))
    plt.tight_layout()
    plt.savefig("Figures/phase/phase_"+str(i)+".png")
    plt.show()

    return

def plot_3d_vect(M,title):

    for i in range(constants.NT):
        if i % 8 == 0:

            d_f = 2 * constants.F / constants.NF
            fatFreqIndex = int((constants.F + constants.FATFREQ) / d_f)

            v_z = np.array([0.00, 0.00, 0.00])
            Mf = np.concatenate((v_z, M[int(constants.NF / 2), i, :]))
            Mw = np.concatenate((v_z, M[fatFreqIndex, i, :]))

            vectors = np.stack((Mf,Mw), axis=0)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            k = 0
            for vector in vectors:
                if k == 0:
                    col = "g"
                else:
                    col = "r"
                v = np.array([vector[3], vector[4], vector[5]])
                vlength = np.linalg.norm(v)
                ax.quiver(vector[0], vector[1], vector[2], vector[3], vector[4], vector[5], color= col)
                #pivot='tail', length=vlength, arrow_length_ratio=0.2 / vlength
                k = k+1
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([0, 1])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            fig.tight_layout()
            plt.savefig("Figures/vect_" + title + str(i) + ".png")
            plt.show()
    return

def amp_phase_normed_plot(B1_amp, B1_phase, amp, phi,i):
    tspace = np.linspace(0, constants.T, len(phi))
    B1_amp_tesla = B1_amp * 1000000 / (constants.GAMMA * constants.DT/1000)
    indexes = np.linspace(0, constants.NT-1, len(amp)).astype(int)


    fig, ax1 = plt.subplots(figsize=(4.5,3))
    color = 'tab:red'
    ax1.set_xlabel("Time [ms]")
    ax1.set_ylabel("B1 amplitude [µT]", color=color)
    ax1.plot(constants.TSEQ, B1_amp_tesla, color)
    ax1.plot(tspace, B1_amp_tesla[indexes], "ro")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xlim((0, np.max(constants.TSEQ)))
    ax1.set_ylim((0, 10))
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel("B1 Phase [rad]", color=color)  # we already handled the x-label with ax1
    ax2.plot(constants.TSEQ, B1_phase, color)
    ax2.plot(tspace, phi.astype(float), "bo")
    ax2.set_xlim((0, np.max(constants.TSEQ)))
    ax2.set_ylim((-1.2*np.pi, 1.2*np.pi))
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig("Figures/ampphi/ampphi" + str(i) + ".png")
    plt.show()

    return

