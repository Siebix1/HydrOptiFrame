# functions to simulate Bloch equations (from Hargreaves online course RAD229)

import numpy as np
from numba import njit
from Modules import constants


@njit
def zrot(phi):
    Rz = np.array(((np.cos(phi), np.sin(phi), 0),
                    (-np.sin(phi), np.cos(phi), 0),
                    (0,0,1)))
    return Rz


@njit
def xrot(phi):
    Rx = np.array(((1, 0, 0),
                   (0,np.cos(phi), np.sin(phi)),
                   (0,-np.sin(phi),np.cos(phi))))
    return Rx


@njit
def yrot(phi):
    Ry = np.array(((np.cos(phi), 0 , -np.sin(phi)),
                   (0, 1, 0),
                   (np.sin(phi),0,np.cos(phi))))
    return Ry


@njit
def throt(phi, theta):
    Rz = zrot(theta)
    Rz_n = zrot(-theta)
    Rx = xrot(phi)
    Rth = (Rz @ Rx @ Rz_n)
    return Rth


@njit
def throtoffres(phi, theta, domega):

    alpha = 2*np.pi*constants.DT*domega/1000
    alphap = np.sqrt((alpha**2 + phi**2))

    if phi == 0:
        beta = np.pi/2

    else:
        #beta = np.arctan(alpha/(phi))
        beta = np.arctan(alpha / (phi))

    Rz = zrot(theta)
    Rz_n = zrot(-theta)
    Ry = yrot(beta)
    Ry_n = yrot(-beta)
    Rx = xrot(alphap)

    Rth = (Rz @ Ry @ Rx @ Ry_n @ Rz_n)
    return Rth


@njit
def throtoffresgrad(phi, theta, domega, grad):

    alpha = 2*np.pi*constants.DT*domega/1000
    alphap = np.sqrt((alpha**2 + phi**2))

    if phi == 0:
        beta = np.pi/2

    else:
        #beta = np.arctan(alpha/(phi))
        beta = np.arctan(alpha / (phi))

    Rz = zrot((theta+grad))
    Rz_n = zrot(-(theta+grad))
    Ry = yrot(beta)
    Ry_n = yrot(-beta)
    Rx = xrot(alphap)

    Rth = (Rz @ Ry @ Rx @ Ry_n @ Rz_n)
    return Rth


@njit
def xrotoffres(phi, tau, domega):
    alpha = 2*np.pi*tau*domega/1000
    alphap = np.sqrt((alpha**2 + phi**2))

    if phi == 0:
        beta = np.pi/2

    else:
        beta = np.arctan(alpha/(phi))

    r11 = np.cos(beta)**2 + np.cos(alphap)*np.sin(beta)**2
    r12 = np.sin(beta)*np.sin(alphap)
    r13 = np.sin(beta)*np.cos(beta)*(1-np.cos(alphap))
    r21 = -np.sin(beta)*np.sin(alphap)
    r22 = np.cos(alphap)
    r23 = np.cos(beta)*np.sin(alphap)
    r31 = np.sin(beta)*np.cos(beta)*(1-np.cos(alphap))
    r32 = -np.cos(beta)*np.sin(alphap)
    r33 = np.sin(beta)**2 + np.cos(alphap)*np.cos(beta)**2

    rx = [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]]
    return rx


@njit
def freeprecess(T,T1,T2,df):
    E1 = np.exp(-T/T1)
    E2 = np.exp(-T/T2)
    alpha = 2*np.pi*T*df/1000

    A = np.array(((E2,0,0),(0,E2,0),(0,0,E1)))
    Rz = zrot(alpha)
    
    Afp = A @ Rz
    Bfp = np.array(((0),(0),(1-E1)))

    return Afp, Bfp.reshape((3,1))


def sssignal(flip,T1,T2,TE,TR,df):
    Ae, Be = freeprecess(TE, T1, T2, df) 
    Atr, Btr = freeprecess(TR-TE, T1, T2, df) 
    
    Ry = yrot(flip)
    I = np.identity(3)
    
    Inv = np.linalg.inv(I - Ae @ Ry @ Atr)
    Mss = Inv @ (Ae @ Ry @ Btr + Be)
    Msig = Mss[0]+1j*Mss[1]
    return Mss, Msig


def srsignal(flip,T1,T2,TE,TR,df):
    Ae, Be = freeprecess(TE, T1, T2, df) 
    Atr, Btr = freeprecess(TR-TE, T1, T2, df) 
    
    Atr = np.array([[0,0,0],[0,0,0],[0,0,1]]) @ Atr

    Ry = yrot(flip)
    I = np.identity(3)
    
    Inv = np.linalg.inv(I - Ae @ Ry @ Atr)
    Mss =  Inv @ (Ae @ Ry @ Btr + Be)
    Msig = Mss[0]+1j*Mss[1]
    return Mss, Msig

def sesignal(flip,T1,T2,TE,TR,df):
    Ae, Be = freeprecess(TE/2, T1, T2, df) 
    Atr, Btr = freeprecess(TR-TE, T1, T2, df) 
    
    Atr = np.array([[0,0,0],[0,0,0],[0,0,1]]) @ Atr

    Ry = yrot(np.pi/2)
    Rx = xrot(np.pi)
    
    I = np.identity(3)
    
    Inv = np.linalg.inv(I - Ae @ Rx @ Ae @ Ry @ Atr)
    Mss =  Inv @ (Ae @ Rx @ (Be + Ae @ Ry @ Btr) + Be)
    Msig = Mss[0]+1j*Mss[1]
    return Mss, Msig

def fsesignal(T1,T2,TE,TR,ETL):
    Ate2, Bte2 = freeprecess(TE/2, T1, T2, 10) 

    Ry = yrot(np.pi/2)
    Rx = xrot(np.pi)
    
    M0 = [[0],[0],[1]]
    
    Mss = np.zeros([3,ETL+1])
    Mtemp = np.zeros([3,ETL+1])
    Msig = np.zeros([ETL+1], dtype = complex)
    
    M1 = Ry @ M0
    print(M1)
    M2 = Rx @ (Ate2 @ M1 + Bte2)
    
    Mss[:,0] = M2.reshape(3,)
    Msig[0] = M2[0]+1j*M2[1]
    Mtemp[:,0] = M2.reshape(3,)
    
    for i in range(ETL):
        Mt = Ate2 @  Mtemp[:,i].reshape(3,1) + Bte2
        
        Mss[:,i+1] = Mt.reshape(3,)
        Msig[i+1] = Mt[0] + 1j*Mt[1]
        
        Mtemp[:,i+1] = (Rx @ (Ate2 @ Mt + Bte2)).reshape(3,)
        
    return Mss, Msig

def gssignal(flip, T1, T2, TE, TR, df, phi):
    Ry = yrot(flip)
    Rz = zrot(phi)
    
    Ae, Be = freeprecess(TE, T1, T2, df) 
    Atr, Btr = freeprecess(TR-TE, T1, T2, df) 

    I = np.identity(3)
    
    Inv = np.linalg.inv(I - Ae @ Ry @ Rz @ Atr)
    
    Mss =  Inv @ (Ae @ Ry @ Rz @ Btr + Be)
    Msig = Mss[0]+1j*Mss[1]

    return Mss, Msig

def gresignal(flip, T1, T2, TE, TR, df):
    
    nphi = 100
    phi = 4*np.pi*(np.linspace(1,nphi,nphi)/nphi-0.5)
    
    Mss1 = np.zeros([3,nphi])
    
    for i in range(nphi):
        M1, M2 = gssignal(flip, T1, T2, TE, TR, df, phi[i])
        Mss1[:, i] = M1.reshape(3,)
    
    Mss = np.mean(Mss1, axis=1)
    
    return Mss
