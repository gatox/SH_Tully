#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 16:00:34 2020

@author: edisonsalazar

Trajectory Surface Hopping (Velocity-Verlet algorithm & Surface hopping: Sharc)

"""

# =============================================================================
# Librarys
# =============================================================================


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

from tully_model_1 import Tully_1
from tully_model_2 import Tully_2
from tully_model_3 import Tully_3
from tully_model_4 import Tully_4

import random 

# =============================================================================
# Functions
# =============================================================================


def position(x, v, a, dt):
	x_new = x + v*dt + 0.5*a*dt*dt
	return x_new


def velocity(v, a_0, a_1, dt):
	v_new = v + 0.5*(a_0 + a_1)*dt
	return v_new

def gradient(x_0, Ene, K_MCH, U):
    """ Constructing the Gradient_MCH matrix and transforming it into 
        diagonal basis.
    """
    G = Tully._gradient(x_0)
    G_MCH = np.diag(np.array([G[0],G[1]])) - (Ene[1]- Ene[0])*K_MCH   
    return np.diag(np.dot(U.T.conj(), np.dot(G_MCH, U)))

def propagator(dia_ij, u_ij, uc_ij, dt):
    p_ij = np.linalg.multi_dot([u_ij, np.diag(np.exp( -1j * dia_ij * dt)), uc_ij])
    return p_ij


# Diagonal hopping (SHARC)
def hopping(c_j_dt, c_j_t, c_i_dt, c_i_t, p_diag_ji, p_diag_ii):
    
    hopp_factor_1 = 1 - np.abs(np.dot(c_i_dt, c_i_dt.conjugate()))/np.abs(np.dot(c_i_t, c_i_t.conjugate())) 
    
    hopp_factor_2_N = ( np.dot(c_j_dt, np.dot(p_diag_ji.conjugate(), c_i_t.conjugate()) ) ).real
    hopp_factor_2_D = ( np.abs(np.dot(c_i_t, c_i_t.conjugate())) - ( np.dot(c_i_dt, np.dot(p_diag_ii.conjugate(), c_i_t.conjugate()) ) ).real)
    
    if hopp_factor_2_D == 0:
        hop_ji = 0.0
    else:
        hop_ji = hopp_factor_1*((hopp_factor_2_N/hopp_factor_2_D))
    return hop_ji


# =============================================================================
# Input parameters
# =============================================================================
print("One-dimentional MD SH: Tully's case 1")



"""initial position"""
x_0 = float(-5) 
"""initial momentum""" 
p_0 = float(20) 
"""mass in atomic units"""
m = 2000.0
"""initial velocity"""
v_0 = p_0 / m
"""number of states"""
nstates = int(2)
"""Ground state"""
c_i = float(0)
"""Excited state"""
c_j = float(1)
"""Initial state"""
if c_j > 0 and c_i < 1:
    state = 1    
else:
    state = 0

"""Tuned factor for coupling"""
F = 30.0

"""Initial time"""
t = 0.0

dt = np.abs(0.05/v_0)
t_max = 2.0*np.abs(x_0/v_0)
md_step = int(t_max / dt)

"""Calling Tully's models:"""

Tully = Tully_1(a = 0.01, b = 1.6, c = 0.005, d = 1.0)
#Tully = Tully_2(a = 0.10, b = 0.28, c = 0.015, d = 0.06, e0 = 0.05)
#Tully = Tully_3(a = 0.0006, b = 0.10, c = 0.90)
#Tully = Tully_4(a = 0.0006, b = 0.10, c = 0.90, d = 4.0)


norm_c_diag_dt = np.zeros([ nstates])

track_state = []
time = []
pos = []
vel = []
cou = []
poten = []
popu = []
norm = []
hopp = []

# =============================================================================
# Velocity Verlet Algorithm
# =============================================================================

while(t <= t_max):
    track_state.append(state)
    
    """Initial coefficient vetor in diagonal basis"""
    c_MCH = np.array([c_i,c_j], dtype=np.complex128)
    cT_MCH = c_MCH.conj()
    rho = np.dot(np.diag(c_MCH),np.diag(cT_MCH))
    
    """ Defining the molecular Coulomb Hamiltonian (MCH) matrix and 
        the nonadiabatic coupling matrix.
    """
    H_MCH = np.diag(Tully._energy(x_0))
    
    K_MCH = F*Tully._a_coupling(x_0)
    

    """ Diagonalising the MCH matrix and total hamiltonian.
    """
    
    Ene, U = np.linalg.eigh(H_MCH)
    
    H_TOTAL = (H_MCH) - 1j*(v_0*K_MCH)

    """ Computing the aceleration at t.
    """
    
    G_diag = gradient(x_0, Ene, K_MCH, U)
    a_0 = - G_diag[state]/m

    
    """ Computing the position and the MCH matrix at t = t + dt
        to get U(t + dt)
    """
    
    x_dt = position(x_0, v_0, a_0, dt)
    
    H_MCH_dt = np.diag(Tully._energy(x_dt))
    
    K_MCH_dt = F*Tully._a_coupling(x_dt)
    
    Ene_dt, U_dt = np.linalg.eigh(H_MCH_dt)
    
    H_TOTAL_dt = (H_MCH_dt) - 1j*(v_0*K_MCH_dt)

    
    """ Average between H_MCH and H_MCH_dt
    
        Three-step-propagator: 
        
            1) Transforamtion c_diag(t) to c_MCH(t), 
            2) Propagating c_MCH(t)) to c_MCH(t+dt), and
            3) Transforming c_MCH(t+dt) to c_diag(t+dt)).
    """
    
    c_diag = np.dot(U.T.conj(),c_MCH)
    
    H_av = 0.5*(H_TOTAL + H_TOTAL_dt)
    
    E_av_eig_dt, U_av_dt = np.linalg.eigh(H_av)
    
    UT_av_dt = U_av_dt.T.conj()
    
    p_MCH_dt = propagator(E_av_eig_dt, U_av_dt, UT_av_dt, dt)
    p_diag_dt = np.dot(U_dt.T.conj() ,np.dot(p_MCH_dt,U))
    p_diag_T_dt = p_diag_dt.T.conj()
    
    c_diag_dt = np.dot(p_diag_dt,c_diag)
    
    
    """Computing density matrix in dt"""
    
    cT_diag_dt = c_diag_dt.conj()
    rho_dt = np.dot(np.diag(c_diag_dt),np.diag(cT_diag_dt))
    
    
    norm_c_diag_dt = np.real(np.diag(rho_dt))
    
    """ Computing hopping from state_i -> state_j and the new aceleration
    """
    
    hop_ij = hopping(c_diag_dt[1 -state], c_diag[1 -state], c_diag_dt[state], c_diag[state], 
                     p_diag_dt[1-state,state], p_diag_dt[state,state])
    if hop_ij < 0:
        hop_ij = 0.0
    
    r = random.uniform(0, 1)
    
    if 0 < r <= hop_ij:
        state = 1 -state
        a_dt = -gradient(x_dt, Ene_dt, K_MCH_dt, U_dt)[state]/m
    else:
        state = state
        a_dt = -gradient(x_dt, Ene_dt, K_MCH_dt, U_dt)[state]/m
        
    """ Computing the new velocity and update the time 
    """

    v_dt = velocity(v_0, a_0, a_dt, dt)
    
    pos.append(x_0)
    vel.append(v_0)
    time.append(t)
    norm.append(norm_c_diag_dt)
    hopp.append(hop_ij)
    poten.append(Ene)
    cou.append(K_MCH[0,1])
    
    x_0 = x_dt
    v_0 = v_dt
    a_t = a_dt
    
    c_i = np.dot(U_dt,c_diag_dt)[0]
    c_j = np.dot(U_dt,c_diag_dt)[1]

    
    
    t = t + dt
    
    
# =============================================================================
# Plots position 
# =============================================================================

plt.plot(time,pos, label = 'x(t)')
plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
plt.ylabel('X(t)', fontweight = 'bold', fontsize = 16)
plt.grid(True)
plt.legend()
plt.show()

# =============================================================================
# Plots velocity
# =============================================================================


plt.plot(time,vel, label = 'v(t)')
plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
plt.ylabel('V(t)', fontweight = 'bold', fontsize = 16)
plt.grid(True)
plt.legend()
plt.show()

# =============================================================================
# Plots energies
# =============================================================================

plt.plot(time,np.array(poten)[:,1], label = 'E_1')
plt.plot(time,np.array(poten)[:,0], label = 'E_0')
plt.plot(time,cou,linestyle='--', label = 'NAC')
plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
plt.ylabel('E_0 & E_1', fontweight = 'bold', fontsize = 16)
plt.grid(True)
plt.legend()
plt.show()

# =============================================================================
# Plots population abs
# =============================================================================

plt.plot(time,np.array(norm)[:,1], label = '|C_j(t)|')
plt.plot(time,np.array(norm)[:,0], label = '|C_i(t)|')
plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
plt.ylabel('Abs|C_i| & Abs|C_j|', fontweight = 'bold', fontsize = 16)
plt.grid(True)
plt.legend()
plt.show()

# =============================================================================
# Plots energies vs position
# =============================================================================

plt.plot(pos,np.array(poten)[:,1], label = 'E_1')
plt.plot(pos,np.array(poten)[:,0], label = 'E_0')
plt.plot(pos,cou,linestyle='--', label = 'NAC')
plt.xlabel('X', fontweight = 'bold', fontsize = 16)
plt.ylabel('E_0 & E_1', fontweight = 'bold', fontsize = 16)
plt.grid(True)
plt.legend()
plt.show()   
    
# =============================================================================
# Printing md_step, time, hopping probability, coupling and current state
# =============================================================================

for i in range(md_step):
    print(i+1," ",time[i]," ",hopp[i]," ",cou[i] , " ",track_state[i])



