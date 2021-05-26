#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 16:00:34 2020

@author: edisonsalazar

Trajectory Surface Hopping (Velocity-Verlet algorithm & Surface hopping of Tully)

"""

# =============================================================================
# Librarys
# =============================================================================


import numpy as np
import matplotlib.pyplot as plt
import random 

from tully_model_1 import Tully_1
from tully_model_2 import Tully_2
from tully_model_3 import Tully_3
from tully_model_4 import Tully_4

# =============================================================================
# Functions
# =============================================================================


def position(x, v, a, dt):
	x_new = x + v*dt + 0.5*a*dt*dt
	return x_new


def velocity(v, a_0, a_1, dt):
	v_new = v + 0.5*(a_0 + a_1)*dt
	return v_new


def propagator(dia_ij, u_ij, uc_ij, dt):
    p_ij = np.linalg.multi_dot([u_ij, np.diag(np.exp( -1j * dia_ij * dt)), uc_ij])
    return p_ij


    #hopping of Tully
def hopping(state, rho, h_ij, dt):
    hop_ji = (2.0 * np.imag(rho[state,:] * h_ij[:,state] * dt))/(np.real(rho[state,state])) 
    hop_ji[state] = 0.0
    hop_ji = np.maximum(hop_ji, 0.0)
    return hop_ji



# =============================================================================
# Input parameters
# =============================================================================
print("One-dimentional MD SH: Tully's models")

"""Initial time"""
t = 0.0
time_step = float(20) #time_step = 20, equivalent to 0.484 fs
md_step = int(400)
au1 = 0.0242  # 1 atomic unit (a.u.) of time is approximately 0.0242 fs
dt = au1*time_step
t_max = dt*md_step

# =============================================================================
# Constants
# =============================================================================



"""Calling Tully's models:"""

#Tully = Tully_1(a = 0.01, b = 1.6, c = 0.005, d = 1.0)
#Tully = Tully_2(a = 0.10, b = 0.28, c = 0.015, d = 0.06, e0 = 0.05)
Tully = Tully_3(a = 0.0006, b = 0.10, c = 0.90)
#Tully = Tully_4(a = 0.0006, b = 0.10, c = 0.90, d = 4.0)

# =============================================================================
# Initial conditions
# =============================================================================


#initial position
x_0 = float(-4) 
#initial momentum 
p_0 = float(80) 
# mass atomic units
m = 2000.0
#initial velocity
v_0 = p_0 / m
#number of states
nstates = int(2)
#Ground state
c_i = float(1)
#Excited state
c_j = float(0)
#Tuned factor for coupling
F = 1.0



t = 0.0



#density matrix
rho = np.zeros([ nstates, nstates ], dtype=np.complex128)

#Choose if the initial state is GS or Exc.State
if c_j > 0 and c_i < 1:
    state = 1
    a_HO_t = -Tully._gradient(x_0)[state]/m
    #a_HO_t = -Gra_U_j(a,b,c,d,x_0)/m
    rho[1,1] = c_j
    
else:
    state = 0
    a_HO_t = -Tully._gradient(x_0)[state]/m
    #a_HO_t = -Gra_U_i(a,b,c,d,x_0)/m
    rho[0,0] = c_i
    

c_t = np.array([c_i, c_j])
norm_c_dt = np.zeros([ nstates])
norm_c_dt_abs = np.zeros([ nstates])
norm_c_dt_real = np.zeros([ nstates])

track_state = []
time = []
pos = []
vel = []
cou = []
poten = []
den = []
popu = []
norm_real = []
norm_abs = []
hopp = []



# =============================================================================
# Velocity Verlet Algorithm
# =============================================================================
while(t <= t_max):
    track_state.append(state)
    #Time in t_0
    
    #u_ij = Tully._di_energy(x_0)
    u_ij = np.diag(Tully._energy(x_0))
    vk = F*Tully._a_coupling(x_0)
    
    Ene, U = np.linalg.eigh(u_ij)
    
    H = (u_ij) - 1j*(v_0*vk)
    
    
    #Time in t_0 + dt
    x_dt = position(x_0, v_0, a_HO_t, dt)
    
    
    #u_ij_dt = Tully._di_energy(x_dt)
    u_ij_dt = np.diag(Tully._energy(x_dt))
    vk_dt = F*Tully._a_coupling(x_dt)

    
    H_dt = (u_ij_dt) - 1j*(v_0*vk_dt)
    
    # Averaged Hamiltonian
    H_av= 0.5*(H + H_dt)
    
    E_av_eig_dt, U_av_dt = np.linalg.eigh(H_av)
    
    UT_av_dt = U_av_dt.T.conj()
    

    Prop_dt = propagator(E_av_eig_dt, U_av_dt, UT_av_dt, dt)
    Prop_T_dt = Prop_dt.T.conj()
    
    rho_dt = np.linalg.multi_dot([Prop_dt, rho, Prop_T_dt])
    c_dt = np.diag(rho_dt)
    

    norm_c_dt_abs = np.abs(c_dt) 
    norm_c_dt_real = np.real(c_dt)
    
    
    
    hop_ji = hopping(state, rho_dt, H_av, dt)

    
    r = random.uniform(0, 1)
    acc_prob = np.cumsum(hop_ji)
    hops = np.less(r, acc_prob)
    
    #if 0 < r <= hop_ji:
    if any(hops):
        state = 1 - state
        a_HO_dt = -Tully._gradient(x_0)[state]/m
    else:
        state = state
        a_HO_dt = -Tully._gradient(x_0)[state]/m
        
        
    
    v_dt = velocity(v_0, a_HO_t, a_HO_dt, dt)
    
    pos.append(x_0)
    vel.append(v_0)
    time.append(t)
    norm_real.append(norm_c_dt_real)
    norm_abs.append(norm_c_dt_abs)
    hopp.append(hops)
    poten.append(Ene)
    cou.append(vk[0,1])

    
    x_0 = x_dt
    v_0 = v_dt
    a_HO_t = a_HO_dt
    rho = rho_dt
    c_t = c_dt
    
    
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
# Plots population real
# =============================================================================

plt.plot(time,np.array(norm_real)[:,1], label = '|C_j(t)|')
plt.plot(time,np.array(norm_real)[:,0], label = '|C_i(t)|')
plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
plt.ylabel('Real|C_i| & Real|C_j|', fontweight = 'bold', fontsize = 16)
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

# =============================================================================
# Printing md_step, time, hopping probability, coupling and current state
# =============================================================================

for i in range(md_step):
    print(i+1," ",time[i]," ",hopp[i]," ",cou[i] , " ",track_state[i])