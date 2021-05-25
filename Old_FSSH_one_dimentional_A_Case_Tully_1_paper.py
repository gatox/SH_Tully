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

def hopping_LZ(a,b,c,d,x,v):
    hop_ji_LZ =math.exp(((-2*math.pi*(V_12(c,d,x))**2)/(2*D_11(a,b,x)))*v)
    return hop_ji_LZ


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
nac_adi = []

for i in np.arange(-x, x, 0.1):
    pos.append(i)
   
    u_ij_di = np.array([V_11(a, b, i), -V_11(a, b, i)])
    E = math.sqrt((V_11(a, b, i))**2 + (V_12(c, d, i))**2)
    u_ij_adi = np.array([-E, E])
    nac_ij = D_11(a,b,i)/(50*(2*E))
    
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

# =============================================================================
# Plot Prob L-Z vs K = momentum at x = 0
# =============================================================================

momen = []
hopp_LZ = []

for k in np.arange(0, 35, 0.1):
    momen.append(k)
    hop_ji_LZ = hopping_LZ(a,b,c,d,0,k)
    
    hopp_LZ.append(hop_ji_LZ)

plt.plot(momen,hopp_LZ, label = 'Pro_L-Z')
plt.xlabel('k(a.u)', fontweight = 'bold', fontsize = 16)
plt.ylabel('Prob_L-Z', fontweight = 'bold', fontsize = 16)
plt.grid(True)
plt.legend()
plt.show()


