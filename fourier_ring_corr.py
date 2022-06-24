# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 12:54:20 2017

@author: sajid

Based on the MATLAB code by Michael Wojcik

M. van Heela, and M. Schatzb, "Fourier shell correlation threshold
criteria," Journal of Structural Biology 151, 250-262 (2005)

"""

#importing required libraries

import numpy as np
import numpy.fft as fft
import spin_average as sa
import matplotlib.pyplot as plt

def FSC(i1,i2,disp=0,SNRt=0.1):
    '''
    Check whether the inputs dimensions match and the images are square
    '''
    if ( np.shape(i1) != np.shape(i2) ) :
        print('input images must have the same dimensions')
    if ( np.shape(i1)[0] != np.shape(i1)[1]) :
        print('input images must be squares')
    I1 = fft.fftshift(fft.fft2(i1))
    I2 = fft.fftshift(fft.fft2(i2))
    '''
    I1 and I2 store the DFT of the images to be used in the calcuation for the FSC
    '''
    C  = sa.spinavej(np.multiply(I1,np.conj(I2)))
    C1 = sa.spinavej(np.multiply(I1,np.conj(I1)))
    C2 = sa.spinavej(np.multiply(I2,np.conj(I2)))
    
    FSC = abs(C)/np.sqrt(abs(np.multiply(C1,C2)))
    
    '''
    T is the SNR threshold calculated accoring to the input SNRt, if nothing is given
    a default value of 0.1 is used.
    
    x2 contains the normalized spatial frequencies
    '''
    r = np.arange(1+np.shape(i1)[0]/2)
    n = 2*np.pi*r
    n[0] = 1
    eps = np.finfo(float).eps
    t1 = np.divide(np.ones(np.shape(n)),n+eps)
    t2 = SNRt + 2*np.sqrt(SNRt)*t1 + np.divide(np.ones(np.shape(n)),np.sqrt(n))
    t3 = SNRt + 2*np.sqrt(SNRt)*t1 + 1
    T = np.divide(t2,t3)
    x1 = np.arange(np.shape(C)[0])/(np.shape(i1)[0]/2)
    x2 = r/(np.shape(i1)[0]/2)    
    '''
    If the disp input is set to 1, an output plot is generated. 
    '''
    if disp != 0 :
        plt.plot(x1,FSC,label = 'FSC')
        plt.plot(x2,T,'--',label = 'Threshold SNR = '+str(SNRt))
        plt.xlim(0,1)
        plt.legend()
        plt.xlabel('Spatial Frequency/Nyquist')
        plt.show()

def compute_frc(
        image_1: np.ndarray,
        image_2: np.ndarray,
        bin_width: int = 2.0
):
    """ Computes the Fourier Ring/Shell Correlation of two 2-D images

    :param image_1:
    :param image_2:
    :param bin_width:
    :return:
    """
    image_1 = image_1 / np.sum(image_1)
    image_2 = image_2 / np.sum(image_2)
    f1, f2 = np.fft.fft2(image_1), np.fft.fft2(image_2)
    af1f2 = np.real(f1 * np.conj(f2))
    af1_2, af2_2 = np.abs(f1)**2, np.abs(f2)**2
    nx, ny = af1f2.shape
    x = np.arange(-np.floor(nx / 2.0), np.ceil(nx / 2.0))
    y = np.arange(-np.floor(ny / 2.0), np.ceil(ny / 2.0))
    distances = list()
    wf1f2 = list()
    wf1 = list()
    wf2 = list()
    for xi, yi in np.array(np.meshgrid(x,y)).T.reshape(-1, 2):
        if abs(xi)>np.sqrt(3)*abs(yi) or (abs(xi)<1e-3 and abs(yi)<1e-3):
            distances.append(np.sqrt(xi**2 + xi**2))
            xi = int(xi)
            yi = int(yi)
            wf1f2.append(af1f2[xi, yi])
            wf1.append(af1_2[xi, yi])
            wf2.append(af2_2[xi, yi])

    bins = np.arange(0, np.sqrt((nx//2)**2 + (ny//2)**2), bin_width)
    f1f2_r, bin_edges = np.histogram(
        distances,
        bins=bins,
        weights=wf1f2
    )
    f12_r, bin_edges = np.histogram(
        distances,
        bins=bins,
        weights=wf1
    )
    f22_r, bin_edges = np.histogram(
        distances,
        bins=bins,
        weights=wf2
    )
    density = f1f2_r / np.sqrt(f12_r * f22_r)
    return density, bin_edges