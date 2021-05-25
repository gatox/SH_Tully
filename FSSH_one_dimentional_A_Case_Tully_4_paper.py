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
import matplotlib.pyplot as plt
#from scipy.linalg import eig, expm
#import random 

# =============================================================================
# Functions
# =============================================================================

def V_12(b, c, d, x):
    if x < -d:
        v = -b*np.exp(c*(x-d)) + b*np.exp(c*(x+d))
    elif x > d:
        v = b*np.exp(-c*(x-d)) - b*np.exp(-c*(x+d))
    else:
        v = 2*b -b*np.exp(c*(x-d)) - b*np.exp(-c*(x+d)) 
    return v

def D_12(b, c, d, x):
    if x < -d:
        v = -c*b*np.exp(c*(x-d)) + c*b*np.exp(c*(x+d))
    elif x > d:
        v = -c*b*np.exp(-c*(x-d)) + c*b*np.exp(-c*(x+d))
    else:
        v = -c*b*np.exp(c*(x-d)) + c*b*np.exp(-c*(x+d)) 
    return v


def V_11(a, x):
    v = a
    return v

def V_22(a, x):
    v = -a
    return v

def D_11(x):
    v = 0.0
    return v

def D_22(x):
    v = 0.0
    return v

def coupling(coeff, DV, dE):
    dij= np.dot(coeff[:,0], np.dot(DV, coeff[:,1]))
    return dij/dE



# =============================================================================
# Initial data
# =============================================================================

a = 0.0006
b = 0.10
c = 0.90
d = 4
x = 15



nac_ij = 0.0
pos = []
poten_di = []
poten_adi = []
nac_adi = []

for i in np.arange(-x, x, 0.1):
    pos.append(i)
   
    H11 = V_11(a, i)
    H22 = V_22(a, i)
    H12 = V_12(b, c, d, i)
    
    u_ij_di = np.array([ [H11, H12],
                         [H12, H22] ])
    
    D11 = D_11(i)
    D22 = D_22(i)
    D12 = D_12(b, c, d, i)
    
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
    
    #ei
    ci_1 = -(H12/np.sqrt( H12**2 + (H11-ei)**2 ))
    ci_2 = (H11-ei)/np.sqrt( H12**2 + (H11-ei)**2 )
    #ej
    cj_1 = -(H12/np.sqrt( H12**2 + (H11-ej)**2 ))
    cj_2 = (H11-ej)/np.sqrt( H12**2 + (H11-ej)**2 )
    
    coeff = np.array([ [ci_1, cj_1],
                       [ci_2, cj_2] ])
    
    nac_ij = -coupling(coeff, DH, dE)
    
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




