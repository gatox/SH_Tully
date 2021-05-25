#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 17:08:22 2021

@author: edisonsalazar
"""

# =============================================================================
# Librarys
# =============================================================================


import numpy as np
import math 
import matplotlib.pyplot as plt
#from scipy.linalg import eig, expm
#import random 

# =============================================================================
# Functions
# =============================================================================

def V_11(a, b, x):
    if x > 0:
        v = a*(1-math.exp(-b*x))
    else:
        v = -a*(1-math.exp(b*x))
    return v


def D_11(a, b, x):
    if x > 0:
        v = a*b*(math.exp(-b*x))
    else:
        v = a*b*(math.exp(b*x))
    return v


def V_12(c, d, x):
    v = c*math.exp(-d*(x**2))
    return v

def D_12(c, d, x):
    v = -2*d*c*x*math.exp(-d*(x**2))
    return v

def coupling(coeff, DV, dE):
    dij= np.dot(coeff[:,0], np.dot(DV, coeff[:,1]))
    return dij/dE

# =============================================================================
# Initial data
# =============================================================================

a = 0.01
b = 1.6
c = 0.005
d = 1.0
x = 10



nac_ij = 0.0
pos = []
poten_di = []
poten_adi = []
poten_adi_2 = []
nac_adi = []

for i in np.arange(-x, x, 0.1):
    pos.append(i)
   
    H11 = V_11(a, b, i)
    H22 = -H11
    H12 = V_12(c, d, i)
    
    u_ij_di = np.array([ [H11, H12],
                         [H12, H22] ])
    
    D11 = D_11(a, b, i)
    D22 = -D11
    D12 = D_12(c, d, i)
    
    DH = np.array([ [D11, D12],
                    [D12, D22] ])
    
    #Coupling way one
    #energies, coeff = np.linalg.eigh(u_ij_di)
    #nac_ij = coupling(coeff, DV, energies)
    #u_ij_adi = energies
    
    #Coupling way two
    dE = np.sqrt( (H11 - H22)**2 + 4*H12**2 )
    ei = 0.5*( (H11 + H22) - dE )
    ej = 0.5*( (H11 + H22) + dE )
    u_ij_adi = np.array([ei,ej])
    
    #coefficients with ei
    ci_1 = -(H12/np.sqrt( H12**2 + (H11-ei)**2 ))
    ci_2 = (H11-ei)/np.sqrt( H12**2 + (H11-ei)**2 )
    #coefficients with ej
    cj_1 = -(H12/np.sqrt( H12**2 + (H11-ej)**2 ))
    cj_2 = (H11-ej)/np.sqrt( H12**2 + (H11-ej)**2 )
    
    coeff = np.array([ [ci_1, cj_1],
                       [ci_2, cj_2] ])
    
    nac_ij = coupling(coeff, DH, dE)/50
    
    poten_di.append(u_ij_di)
    poten_adi.append(u_ij_adi)
    nac_adi.append(nac_ij)
    
        


# =============================================================================
# Plot PES_adia and NAC
# =============================================================================

plt.plot(pos,np.array(poten_adi)[:,1], label = 'E_1')
plt.plot(pos,np.array(poten_adi)[:,0], label = 'E_0')
plt.plot(pos,nac_adi,linestyle='--', label = 'NAC')
plt.xlabel('X', fontweight = 'bold', fontsize = 16)
plt.ylabel('E_0 & E_1', fontweight = 'bold', fontsize = 16)
plt.grid(True)
plt.legend()
plt.show()


