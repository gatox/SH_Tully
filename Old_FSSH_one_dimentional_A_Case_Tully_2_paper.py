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


def V_22(a, b, e0, x):
    v = -a*math.exp(-b*(x**2)) + e0
    return v

def D_22(a, b, x):
    v = 2*a*b*x*math.exp(-b*(x**2))
    return v


def V_12(c, d, x):
    v = c*math.exp(-d*(x**2))
    return v

def D_12(c, d, x):
    v = -2*d*c*x*math.exp(-d*(x**2))
    return v


# =============================================================================
# Initial data
# =============================================================================

a = 0.10
b = 0.28
c = 0.015
d = 0.06
e0 = 0.05
x = 10



nac_ij = 0.0
pos = []
poten_di = []
poten_adi = []
nac_adi = []

for i in np.arange(-x, x, 0.1):
    pos.append(i)
   
    u_ij_di = np.array([0, V_22(a, b, e0, i)])
    E1 = 0.5*( V_22(a, b, e0, i) - math.sqrt((V_22(a, b, e0, i))**2 + 4*(V_12(c, d, i)**2)) ) 
    E2 = 0.5*( V_22(a, b, e0, i) + math.sqrt((V_22(a, b, e0, i))**2 + 4*(V_12(c, d, i)**2)) ) 
    u_ij_adi = np.array([E1, E2])
    #nac_ij = - ( D_22(a, b, i)*V_22(a, b, e0, i)*V_12(c, d, i) + D_12(c, d, i)*V_12(c, d, i)**2 )/(12*( V_12(c, d, i)**2 + V_22(a, b, e0, i)**2 )*(E2-E1))
    nac_ij = - ( D_22(a, b, i) )/(24*(E2-E1))
    poten_di.append(u_ij_di)
    poten_adi.append(u_ij_adi)
    nac_adi.append(nac_ij)
    
        

# =============================================================================
# Plot PES_dia
# =============================================================================

plt.plot(pos,np.array(poten_di)[:,1], label = 'E_1')
plt.plot(pos,np.array(poten_di)[:,0], label = 'E_0')
plt.xlabel('X', fontweight = 'bold', fontsize = 16)
plt.ylabel('E_0 & E_1', fontweight = 'bold', fontsize = 16)
plt.grid(True)
plt.legend()
plt.show()

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

# =============================================================================
# Plot PES_adia PES_dia and NAC
# =============================================================================

plt.plot(pos,np.array(poten_adi)[:,1], label = 'E_2')
plt.plot(pos,np.array(poten_adi)[:,0], label = 'E_1')
plt.plot(pos,np.array(poten_di)[:,1], label = 'V_22')
plt.plot(pos,np.array(poten_di)[:,0], label = 'V_11')
plt.plot(pos,nac_adi,linestyle='--', label = 'NAC')
plt.xlabel('X', fontweight = 'bold', fontsize = 16)
plt.ylabel('Energy', fontweight = 'bold', fontsize = 16)
plt.grid(True)
plt.legend()
plt.show()




