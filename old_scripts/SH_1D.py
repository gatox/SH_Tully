#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 16:00:34 2020

@author: edisonsalazar

Trajectory Surface Hopping (Velocity-Verlet algorithm & Tully hopping for model 1)

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


def propagator(h_ij, dt):
    p_ij = expm( -h_ij * dt)
    return p_ij



# Diagonal hopping (SHARC)
def hopping(c_j_dt, c_j_t, c_i_dt, c_i_t, p_diag_ji, p_diag_ii):
    hop_ij = (1 - ( np.abs(np.dot(c_i_dt, c_i_dt.conjugate())) )/\
              ( np.abs(np.dot(c_i_t, c_i_t.conjugate())) ) )*\
              ( ( np.dot(c_j_dt, np.dot(p_diag_ji.conjugate(), c_i_t.conjugate()) ) ).real/\
               ( np.abs(np.dot(c_i_dt, c_i_dt.conjugate())) 
                - ( np.dot(c_i_dt, np.dot(p_diag_ii.conjugate(), c_i_t.conjugate()) ) ).real))
    return hop_ij


# =============================================================================
# Input parameters
# =============================================================================
print("One-dimentional MD SH: Tully's case 1")

"""Initial time"""
t = 0.0
time_step = float(20) #time_step = 20, equivalent to 0.484 fs
md_step = int(1100)
au1 = 0.0242  # 1 atomic unit (a.u.) of time is approximately 0.0242 fs
dt = au1*time_step
t_max = dt*md_step

"""initial position"""
x_0 = float(-4) 
"""initial momentum""" 
p_0 = float(30) 
"""mass in atomic units"""
m = 2000.0
"""initial velocity"""
v_0 = p_0 / m
"""number of states"""
nstates = int(2)
"""Ground state"""
c_i = float(1)
"""Excited state"""
c_j = float(0)
"""Initial state"""
state = 0


"""Calling Tully's models:"""
Tully = Tully_1(a = 0.01, b = 1.6, c = 0.005, d = 1.0)
#Tully = Tully_2(a = 0.10, b = 0.28, c = 0.015, d = 0.06, e0 = 0.05)
#Tully = Tully_3(a = 0.0006, b = 0.10, c = 0.90)
#Tully = Tully_4(a = 0.0006, b = 0.10, c = 0.90, d = 4.0)

"""Initial coefficient vetor in diagonal basis"""
c_diag = np.array([c_i,c_j], dtype=np.complex128)


track_state = []
time = []
pos = []
vel = []
cou = []
poten = []
popu = []
norm_abs = []
hopp = []

# =============================================================================
# Velocity Verlet Algorithm
# =============================================================================

while(t <= t_max):
    
    """ Defining the molecular Coulomb Hamiltonian (MCH) matrix and 
        the nonadiabatic coupling matrix.
    """
    H_MCH = np.diag(Tully._energy(x_0))
    
    K_MCH = Tully._a_coupling(x_0)
    
    """ Diagonalising the MCH matrix.
    """
    
    Ene, U = np.linalg.eigh(H_MCH)
    
    """ Constructing the Gradient_MCH matrix and transforming it into 
        diagonal basis.
    """
    G = Tully._gradient(x_0)
    
    G_MCH = np.diag(G) - (Ene[1]- Ene[0])*K_MCH
    
    G_diag = np.diag(np.dot(U.T.conj(), np.dot(G_MCH, U)))
    
    """ Computing the aceleration at t.
    """
    
    a_t = - G_diag[state]/m
    
    """ Computing the position and the MCH matrix at t = t + dt
        to get U(t + dt)
    """
    
    x_dt = position(x_0, v_0, a_t, dt)
    
    H_MCH_dt = Tully._di_energy(x_dt)
    
    K_MCH_dt = Tully._a_coupling(x_dt)
    
    Ene_dt, U_dt = np.linalg.eigh(H_MCH_dt)

    
    """ Average between H_MCH and H_MCH_dt
    
        Three-step-propagator: 
        
            1) Transforamtion c_diag(t) to c_MCH(t), 
            2) Propagating c_MCH(t)) to c_MCH(t+dt), and
            3) Transforming c_MCH(t+dt) to c_diag(t+dt)).
    """
    
    H_av = 0.5*( 1j*np.diag(Ene + Ene_dt) + v_0*(K_MCH + K_MCH_dt ) )
    
    p_MCH_dt = propagator(H_av, dt)
    
    p_diag_dt = np.dot(U_dt.T.conj(), np.dot(p_MCH_dt, U))
    
    c_diag_dt = np.dot(p_diag_dt,c_diag)
    
    norm_c_diag_dt = np.abs(c_diag_dt)
    
    """ Computing hopping from state_i -> state_j and the new aceleration
    """
    
    hop_ij = hopping(c_diag_dt[1], c_diag[1], c_diag_dt[0], c_diag[0], 
                     p_diag_dt[1,0], p_diag_dt[0,0])
    
    r = random.uniform(0, 1)
    
    if 0 < r <= hop_ij:
        state = 1
        a_dt = - G_diag[state]/m
        #print("Yes hopping at",t, "with r",r, "<=",hop_ij)
    else:
        state = 0
        a_dt = - G_diag[state]/m
        #print("No hopping at",t, "with r",r, "<=",hop_ij)
        
    """ Computing the new velocity and update the time 
    """

    v_dt = velocity(v_0, a_t, a_dt, dt)
    
    pos.append(x_0)
    vel.append(v_0)
    time.append(t)
    norm_abs.append(norm_c_diag_dt)
    hopp.append(hop_ij)
    poten.append(Ene)
    cou.append(K_MCH[0,1])
    track_state.append(state)
    
    x_0 = x_dt
    v_0 = v_dt
    a_t = a_dt
    c_diag = c_diag_dt
    
    
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

plt.plot(time,np.array(norm_abs)[:,1], label = '|C_j(t)|')
plt.plot(time,np.array(norm_abs)[:,0], label = '|C_i(t)|')
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
    



